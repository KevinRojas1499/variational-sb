import torch
import utils.sde_lib as SDEs
import utils.losses as losses

class AlternateTrainingRoutine():
    def __init__(self,sb : SDEs.SchrodingerBridge, sampling_sb : SDEs.SchrodingerBridge,
                 model_forward, model_backward,
                 refresh_rate, n_time_pts):
        self.sb = sb
        self.sampling_sb = sampling_sb
        self.model_forward = model_forward
        self.model_backward = model_backward
        self.refresh_rate = refresh_rate
        self.trajectories = None
        self.frozen_policy = None
        self.time_pts = torch.linspace(0., sb.T,n_time_pts)
    
    def _optimizing_forward(self, itr):
        return (itr//self.refresh_rate)%2 == 1
    
    def refresh(self, itr, data):
        optimizing_forward = self._optimizing_forward(itr)
        refresh = (itr//self.refresh_rate)%2 != ((itr-1)//self.refresh_rate)%2         
        
        if refresh or itr == 0:
            if self.sb.is_augmented:
                data = losses.augment_data(data)
            in_cond = self.sb.prior_sampling((*data.shape,),device=data.device) if optimizing_forward else data
            xt, trajectories, frozen_policy = self.sampling_sb.get_trajectories_for_loss(in_cond, self.time_pts,forward=not optimizing_forward)
            self.trajectories = trajectories.detach_()
            self.frozen_policy = frozen_policy.detach_()      
        
            if optimizing_forward:
                self.model_backward.requires_grad_(False)
                self.model_forward.requires_grad_(True)
            else:
                self.model_forward.requires_grad_(False)
                self.model_backward.requires_grad_(True)
                
    def training_iteration(self, itr, data):
        self.refresh(itr, data)
        return losses.alternate_sb_loss(self.sb,self.trajectories,self.frozen_policy,self.time_pts,self._optimizing_forward(itr))


class VariationalDiffusionTrainingRoutine():
    def __init__(self,sb : SDEs.LinearSchrodingerBridge, sampling_sb : SDEs.LinearSchrodingerBridge,
                 num_iters_dsm_warm_up, num_iters_middle, num_iters_dsm_cool_down,
                 num_iters_forward, num_iters_backward, n_time_pts):
        self.sb = sb
        self.sampling_sb = sampling_sb
        self.model_forward = self.sb.At
        self.model_backward = self.sb.backward_score
        if self.sb.is_augmented:
            self.base_sde = SDEs.CLD(T=self.sb.T, delta=self.sb.delta, beta_max=self.sb.beta_max, model_backward=self.model_backward)
        else:
            self.base_sde = SDEs.VP(T=self.sb.T,delta=self.sb.delta,beta_max=self.sb.beta_max,model_backward=self.model_backward)
        self.trajectories = None
        self.frozen_policy = None
        self.time_pts = torch.linspace(0., sb.T,n_time_pts)
        self.num_iters_dsm_warm_up = num_iters_dsm_warm_up
        self.num_iters_middle_stage = num_iters_middle 
        self.num_iters_dsm_cool_down = num_iters_dsm_cool_down
        
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
        in_cond = self.sb.prior_sampling((*data.shape,),device=data.device)
        xt, trajectories, frozen_policy = self.sampling_sb.get_trajectories_for_loss(in_cond, self.time_pts,forward=False,cond=cond)
        self.trajectories = trajectories.detach_()
        self.frozen_policy = frozen_policy.detach_()      
        
        self.freeze_models(optimizing_forward=True)    
    
    @torch.no_grad()
    def refresh_backward(self, data):
        self.loss_times = torch.linspace(self.sb.delta, self.sb.T, 500, device=data.device)
        ones = [1] * (len(data.shape)-1)
        shaped_t = self.loss_times.reshape(-1,*ones)
        self.scales, self.stds = self.sb.get_transition_params(torch.empty((self.loss_times.shape[0], *data.shape[1:]),device=data.device), shaped_t)
        self.freeze_models(optimizing_forward=False)    
            
                
    def training_iteration(self, itr, data,cond=None):
        prev_stage = self.get_training_stage(itr-1)
        stage = self.get_training_stage(itr)
        if prev_stage != stage:
            print('CHANGED STAGES')
            print(prev_stage, stage)
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
            return losses.alternate_sb_loss(self.sb,self.trajectories,self.frozen_policy,self.time_pts,optimize_forward=True)
class EvalLossRoutine():
    def __init__(self, sde, loss_fn):
        self.sde = sde
        self.loss_fn = loss_fn
    
    def training_iteration(self, itr, data,cond=None):
        return self.loss_fn(self.sde, data,cond)

def get_routine(sde, sampling_sde, opts):
    print(sde)
    if isinstance(sde,(SDEs.VP, SDEs.CLD)):
        return EvalLossRoutine(sde=sde, loss_fn=losses.get_loss(opts.sde, is_alternate_training=False))
    else:
        return VariationalDiffusionTrainingRoutine(sde,sampling_sde,
                                                      opts.dsm_warm_up,opts.num_iters-opts.dsm_warm_up - opts.dsm_cool_down, opts.dsm_cool_down,
                                                      opts.forward_opt_steps, opts.backward_opt_steps,100)