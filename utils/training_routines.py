import torch
import utils.sde_lib as SDEs
import utils.losses as losses

class AlternateTrainingRoutine():
    def __init__(self,sb : SDEs.SchrodingerBridge, sampling_sb : SDEs.SchrodingerBridge,
                 model_forward, model_backward,
                 refresh_rate, n_time_pts, device):
        self.sb = sb
        self.sampling_sb = sampling_sb
        self.model_forward = model_forward
        self.model_backward = model_backward
        self.refresh_rate = refresh_rate
        self.trajectories = None
        self.frozen_policy = None
        self.time_pts = torch.linspace(0., sb.T,n_time_pts,device=device)
    
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
    def __init__(self,sb : SDEs.GeneralLinearizedSB, sampling_sb : SDEs.GeneralLinearizedSB,
                 model_forward, model_backward,
                 refresh_rate, n_time_pts, device):
        self.sb = sb
        self.sampling_sb = sampling_sb
        self.model_forward = model_forward
        self.model_backward = model_backward
        self.refresh_rate = refresh_rate
        self.trajectories = None
        self.frozen_policy = None
        self.time_pts = torch.linspace(0., sb.T,n_time_pts,device=device)
        # Parameters for loss fn in backward variational  optimization
        self.loss_times = None
        self.big_betas = None
        self.Ls = None
    
    def freeze_models(self, optimizing_forward):
        if optimizing_forward:
            self.model_backward.requires_grad_(False)
            self.model_forward.requires_grad_(True)
        else:
            self.model_forward.requires_grad_(False)
            self.model_backward.requires_grad_(True)

    def _optimizing_forward(self, itr):
        return (itr//self.refresh_rate)%2 == 1
    
    def _needs_refresh(self, itr):
        return ((itr//self.refresh_rate)%2 != ((itr-1)//self.refresh_rate)%2) or itr == 0
    
    def refresh_forward(self, itr, data):
        optimizing_forward = self._optimizing_forward(itr)
        refresh = self._needs_refresh(itr)
        
        if refresh:
            if self.sb.is_augmented:
                data = losses.augment_data(data)
            in_cond = self.sb.prior_sampling((*data.shape,),device=data.device) if optimizing_forward else data
            xt, trajectories, frozen_policy = self.sampling_sb.get_trajectories_for_loss(in_cond, self.time_pts,forward=not optimizing_forward)
            self.trajectories = trajectories.detach_()
            self.frozen_policy = frozen_policy.detach_()      
            
            self.freeze_models(optimizing_forward)    
    
    def refresh_backward(self, itr, data):
        optimizing_forward = self._optimizing_forward(itr)
        refresh = self._needs_refresh(itr)
        
        if refresh:
            eps = self.sb.delta
            self.loss_times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * self.sb.T
            shaped_t = self.loss_times.reshape(-1,1,1,1) if len(data.shape) > 2 else self.loss_times.reshape(-1,1)
            _, Ls, big_betas = self.sb.compute_variance(shaped_t)
            self.Ls = Ls.detach().clone()
            self.big_betas = big_betas.detach().clone()
            self.freeze_models(optimizing_forward)    
            
                
    def training_iteration(self, itr, data):
        optimizing_forward = self._optimizing_forward(itr)
        if optimizing_forward:
            self.refresh_forward(itr, data)
            return losses.alternate_sb_loss(self.sb,self.trajectories,self.frozen_policy,self.time_pts,optimize_forward=True)
        else:
            self.refresh_backward(itr,data)
            return losses.linear_sb_loss_given_params(self.sb, data,self.loss_times,self.big_betas,self.Ls)

class EvalLossRoutine():
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
    
    def training_iteration(self, itr, data):
        return self.loss_fn

def get_routine(self, name):
    if name == 'alternate':
        return AlternateTrainingRoutine()