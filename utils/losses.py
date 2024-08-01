import torch
import utils.sde_lib as SDEs

from utils.diff import hutch_div

#######################################
#          Diffusion Losses           #
#######################################

def dsm_loss(sde : SDEs.LinearSDE,data, cond=None):
    eps = sde.delta
    times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
    ones = [1] * (len(data.shape)-1)
    
    shaped_t = times.reshape(-1,*ones)
    mean, variance = sde.marginal_prob(data,shaped_t)
    noise = torch.randn_like(mean,device=data.device)
    perturbed_data = mean + variance**.5 * noise
    flatten_error = ((variance**.5 * sde.backward_score(perturbed_data,times,cond) + noise)**2).view(data.shape[0],-1)
    
    return torch.mean(torch.sum(flatten_error,dim=1))

def linear_sb_loss_given_params(sde : SDEs.LinearSDE, data, times, scale, std, cond=None):
    noise = torch.randn_like(data,device=data.device)
    if sde.is_augmented:
        x,v = data.chunk(2,dim=1)
        new_x = scale[...,0,0] * x + scale[...,0,1] * v
        new_v = scale[...,1,0] * x + scale[...,1,1] * v
        
        mean = torch.cat((new_x,new_v),dim=1)

        noise_x, noise_v = noise.chunk(2, dim=1)
        new_noise_x = std[...,0,0] * noise_x
        new_noise_v = std[...,1,0] * noise_x + std[...,1,1] * noise_v
        perturbed_data = mean + torch.cat((new_noise_x, new_noise_v),dim=1)
        flatten_error = ((std[...,1,1]  * sde.backward_score(perturbed_data,times,cond) + noise_v)**2).view(data.shape[0],-1)
    else:
        perturbed_data = scale * data + std * noise
        flatten_error = ((std * sde.backward_score(perturbed_data,times,cond) + noise)**2).view(data.shape[0],-1)
    
    return torch.mean(torch.sum(flatten_error,dim=1))
     
def linear_sb_loss(sde : SDEs.LinearSDE,data, cond=None):
    eps = sde.delta
    times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
    ones = [1] * (len(data.shape)-1)
    
    shaped_t = times.reshape(-1,*ones)
    mean, std = sde.marginal_prob(data,shaped_t)
    noise = torch.randn_like(mean,device=data.device)
    if sde.is_augmented:
        noise_x, noise_v = noise.chunk(2, dim=1)
        new_noise_x = std[...,0,0] * noise_x
        new_noise_v = std[...,1,0] * noise_x + std[...,1,1] * noise_v
        perturbed_data = mean + torch.cat((new_noise_x, new_noise_v),dim=1)
        flatten_error = ((std[...,1,1]  * sde.backward_score(perturbed_data,times,cond) + noise_v)**2).view(data.shape[0],-1)
    else:
        perturbed_data = mean + std * noise
        flatten_error = ((std * sde.backward_score(perturbed_data,times,cond) + noise)**2).view(data.shape[0],-1)
        
    return torch.mean(torch.sum(flatten_error,dim=1))


def cld_loss(sde : SDEs.CLD,data,cond=None):
    eps = sde.delta
    times = (torch.rand((data.shape[0]) ,device=data.device) * (1-eps) + eps) * sde.T
    ones = [1] * (len(data.shape)-1)
    shaped_t = times.reshape(-1,*ones)
    ext_data = torch.cat((data,torch.zeros_like(data)),dim=1) # Add velocity
    mean = sde.marginal_prob_mean(ext_data,shaped_t)
    lxx, lxv, lvv = sde.marginal_prob_std(shaped_t)
    noise = torch.randn_like(mean,device=data.device)
    perturbed_data = mean + sde.multiply_std(noise,shaped_t)
    flatten_error = ((sde.backward_score(perturbed_data,times,cond) * lvv + noise.chunk(2,dim=1)[1])**2).view(data.shape[0],-1)
    
    return torch.mean(torch.sum(flatten_error,dim=1))

#######################################
#         Schrodinger Losses          #
#######################################

def joint_sb_loss(sde : SDEs.SchrodingerBridge, in_cond, time_pts):
    # This corresponds to joint training
    # We assume that time_pts is a uniform discretization of the interval [0,T]
    # We also assume that f has constant divergence
    xt, trajectories, forward_scores = sde.get_trajectories_for_loss(in_cond, time_pts)
    batch_size, d = xt.shape[0], xt.shape[-1]
    dt = time_pts[1]-time_pts[0] 
    
    # We now compute the loss fn, we first need to do some reshaping
    time_pts_shaped = time_pts.repeat(batch_size)
    bt = sde.beta(time_pts_shaped)
    flat_traj = trajectories.reshape(-1,d).requires_grad_(True)
    backward_scores =  bt**.5 * sde.backward_score(flat_traj,time_pts_shaped) # Backward policies as described in the FBSDE paper
    
    if sde.is_augmented:
        backward_scores*= sde.gamma**.5
        aug_backward_score = torch.cat((torch.zeros_like(backward_scores),backward_scores),dim=1)
        div_term = (sde.gamma * bt)**.5 * hutch_div(aug_backward_score,flat_traj,time_pts_shaped) 
    else:
        div_term = bt**.5 * hutch_div(backward_scores,flat_traj,time_pts_shaped) 
        
    loss = .5 * torch.sum((backward_scores + forward_scores.view(-1,backward_scores.shape[-1]))**2)/batch_size \
      + torch.sum(div_term)/batch_size
    loss = dt * loss
    loss += .5 * torch.sum(xt**2)/batch_size
    return loss

def alternate_sb_loss(sde : SDEs.SchrodingerBridge,trajectories, frozen_policy, time_pts, optimize_forward):
    # This corresponds to alternate training
    # We assume that time_pts is a uniform discretization of the interval [0,T]
    # We also assume that f has constant divergence
    # if optimize forward is true we will optimize the forward score, if not we will optimize the backward
    assert not frozen_policy.requires_grad
    
    frozen_policy = frozen_policy.detach()
    batch_size, data_shape = trajectories.shape[0], trajectories.shape[2:]
    dt = time_pts[1]-time_pts[0] 
    
    # We now compute the loss fn, we first need to do some reshaping
    time_pts_shaped = time_pts.repeat(batch_size)
    bt = sde.beta(time_pts_shaped)
    flat_traj = trajectories.reshape(-1,*data_shape).requires_grad_(True)
    
    if optimize_forward:
        opt_policy =  bt**.5 * sde.forward_score(flat_traj,time_pts_shaped) 
    else:
        opt_policy =  bt**.5 * sde.backward_score(flat_traj,time_pts_shaped)
    
    if sde.is_augmented:
        opt_policy*= sde.gamma**.5
        aug_opt_policy = torch.cat((torch.zeros_like(opt_policy),opt_policy),dim=1)
        div_term = (sde.gamma * bt)**.5 * hutch_div(aug_opt_policy,flat_traj,time_pts_shaped) 
    else:
        div_term = bt**.5 * hutch_div(opt_policy,flat_traj,time_pts_shaped) 

    loss = torch.sum(opt_policy * (.5 * opt_policy + frozen_policy.view(-1,*opt_policy.shape[1:])))/batch_size \
      + torch.sum(div_term)/batch_size
        
    loss = dt * loss
    return loss

def standard_sb_loss(sde : SDEs.SchrodingerBridge, data):
    n_times = 100
    time_pts = torch.linspace(0., sde.T,n_times, device=data.device)
    
    if sde.is_augmented:
        v_noise = torch.randn_like(data)
        data = torch.cat((data,v_noise),dim=1)

    return joint_sb_loss(sde,data,time_pts)


def augment_data(data):
    v_noise = torch.randn_like(data)
    return torch.cat((data,v_noise),dim=1)

def standard_alternate_sb_loss(sde : SDEs.SchrodingerBridge, data,optimize_forward, sampling_sde: SDEs.SchrodingerBridge):
    n_times = 100
    time_pts = torch.linspace(0., sde.T,n_times, device=data.device)
    
    if sde.is_augmented:
        data = augment_data(data)
    
    in_cond = sde.prior_sampling((*data.shape,),device=data.device) if optimize_forward else data
    xt, trajectories, frozen_policy = sampling_sde.get_trajectories_for_loss(in_cond, time_pts,forward=not optimize_forward)
    
    return alternate_sb_loss(sde,trajectories,frozen_policy,time_pts,optimize_forward)

def get_loss(sde_name, is_alternate_training):
    if sde_name == 'vp':
        return dsm_loss
    elif sde_name in ('sb','momentum-sb'):
        if is_alternate_training:
            return standard_alternate_sb_loss
        else:
            return standard_sb_loss
    elif sde_name in ('linear-sb','linear-momentum-sb'):
        return linear_sb_loss
    elif sde_name == 'cld':
        return cld_loss
    