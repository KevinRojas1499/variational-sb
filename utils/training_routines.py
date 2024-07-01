import torch
import utils.sde_lib as SDEs
import utils.losses as losses

class AlternateTrainingLoop():
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
        