import torch
import utils.sde_lib as SDEs
import utils.losses as losses

class VariationalDiffusionTrainingRoutine():
    def __init__(self,sb : SDEs.VSDM, sampling_sb : SDEs.VSDM,
                 num_iters_dsm_warm_up, num_iters_middle, num_iters_dsm_cool_down,
                 num_iters_forward, num_iters_backward, n_time_pts,
                 opt_f=None, sched_f=None, ema_f=None, opt_b=None, sched_b=None, ema_b=None):
        self.opt_f = opt_f
        self.sched_f = sched_f
        self.ema_f = ema_f
        self.opt_b = opt_b
        self.sched_b = sched_b
        self.ema_b = ema_b
        
        self.sb = sb
        self.sampling_sb = sampling_sb
        self.model_forward = self.sb.At
        self.model_backward = self.sb.backward_score
        if self.sb.is_augmented:
            self.base_sde = SDEs.CLD(T=self.sb.T, delta=self.sb.delta, beta_max=self.sb.beta_max, backward_model=self.model_backward)
        else:
            self.base_sde = SDEs.VP(T=self.sb.T,delta=self.sb.delta,beta_max=self.sb.beta_max,backward_model=self.model_backward)
        self.trajectories = None
        self.frozen_policy = None
        self.time_pts = torch.linspace(0., sb.T-sb.delta,n_time_pts)
        self.num_iters_dsm_warm_up = num_iters_dsm_warm_up
        self.num_iters_middle_stage = num_iters_middle 
        self.num_iters_dsm_cool_down = num_iters_dsm_cool_down
        
        self.adaptive_prior = False
        self.num_iters_forward = num_iters_forward
        self.num_iters_backward = num_iters_backward
        self.refresh_rate = self.num_iters_backward + self.num_iters_forward
        # Parameters for loss fn in backward variational  optimization
        self.loss_times = None
        self.scales = None
        self.stds = None

    def freeze_models(self, optimizing_forward):
        self.model_forward.requires_grad_(optimizing_forward)
        self.model_backward.requires_grad_(not optimizing_forward)
    
    def get_training_stage(self, itr):
        if itr<0:
            return '#'
        if itr < self.num_iters_dsm_warm_up:
            return 'dsm'
        elif itr < self.num_iters_dsm_warm_up  + self.num_iters_middle_stage:
            _itr = itr - self.num_iters_dsm_warm_up
            if _itr%self.refresh_rate < self.num_iters_backward:
                return 'backward'
            else:
                return 'forward'    
        else:
            return 'backward'

    @torch.no_grad()
    def refresh_forward(self, data,cond=None):
        self.time_pts = self.time_pts.to(device=data.device)
        if self.sb.is_augmented:
            data = losses.augment_data(data)
        in_cond = self.sampling_sb.prior_sampling((*data.shape,),device=data.device) if self.adaptive_prior \
                else torch.randn(*data.shape, device=data.device)
        xt, trajectories, frozen_policy = self.sampling_sb.get_trajectories_for_loss(in_cond, self.time_pts,forward=False,cond=cond)
        self.trajectories = trajectories.detach_()
        self.frozen_policy = frozen_policy.detach_()      
        
        self.freeze_models(optimizing_forward=True)    
    
    @torch.no_grad()
    def refresh_backward(self, data):
        self.loss_times = torch.linspace(self.sb.delta, self.sb.T, 100, device=data.device)
        ones = [1] * len(data.shape[1:])
        shaped_t = self.loss_times.reshape(-1,*ones)
        self.scales, self.stds = self.sampling_sb.get_transition_params(torch.empty((self.loss_times.shape[0], *data.shape[1:]),device=data.device), shaped_t)
        self.freeze_models(optimizing_forward=False)    
            
                
    def get_loss(self, itr, data,cond=None):
        prev_stage = self.get_training_stage(itr-1)
        stage = self.get_training_stage(itr)
        if stage == 'dsm':
            if itr == 0:
                self.freeze_models(optimizing_forward=False)
            if self.sb.is_augmented:
                return losses.cld_loss(self.base_sde,data,cond)
            else:
                return losses.dsm_loss(self.base_sde,data,cond)
        elif stage == 'backward':
            aug_data = losses.augment_data(data) if self.sb.is_augmented else data
            if prev_stage != stage or self.loss_times is None:
                self.refresh_backward(aug_data)
            rand_idx = torch.randint(0,self.loss_times.shape[0],(data.shape[0],), device=data.device)
            return losses.linear_sb_loss_given_params(self.sb,aug_data,self.loss_times[rand_idx],self.scales[rand_idx],self.stds[rand_idx],cond)
        elif stage == 'forward':
            if prev_stage != stage:
                self.refresh_forward(data,cond)
            rand_idx = torch.randint(0,data.shape[0],(1000,), device=data.device)
            rand_idx = torch.arange(0,data.shape[0], device=data.device)
            
            return losses.alternate_sb_loss(self.sb,self.trajectories[rand_idx],self.frozen_policy[rand_idx],self.time_pts,optimize_forward=True)

    def training_iteration(self, itr, data,cond=None):
        stage = self.get_training_stage(itr)
        opt, sched, ema = (self.opt_f, self.sched_f, self.ema_f) if \
            stage == 'forward' else (self.opt_b, self.sched_b, self.ema_b)
        model = self.sb.At if stage == 'forward' else self.sb.backward_score 
        opt.zero_grad()
        loss = self.get_loss(itr,data,cond)
        loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        opt.step()
        sched.step()
        ema.update()
        
        return loss

class EvalLossRoutine():
    def __init__(self, sde, loss_fn, opt, sched):
        self.sde = sde
        self.loss_fn = loss_fn
        self.opt = opt
        self.sched = sched
        
    def get_loss(self, itr, data,cond=None):
        return self.loss_fn(self.sde,data,cond)
    
    def training_iteration(self, itr, data,cond=None):
        self.opt.zero_grad()
        loss = self.loss_fn(self.sde, data,cond)
        loss.backward()
        self.opt.step()
        self.sched.step()
        return loss

def get_routine(opts, num_iters, sde,sampling_sde,
                opt_b=None, sched_b=None, ema_backward=None, 
                opt_f=None, sched_f=None, ema_forward=None):
    if isinstance(sde,(SDEs.VP, SDEs.CLD)):
        return EvalLossRoutine(sde=sde, loss_fn=losses.get_loss(opts.sde),
                               opt=opt_b, sched=sched_b)
    else:
        return VariationalDiffusionTrainingRoutine(sde,sampling_sde,\
            opts.dsm_warm_up,num_iters-opts.dsm_warm_up - opts.dsm_cool_down, opts.dsm_cool_down,
            opts.forward_opt_steps, opts.backward_opt_steps,100,
            opt_f, sched_f, ema_forward, opt_b, sched_b, ema_backward)