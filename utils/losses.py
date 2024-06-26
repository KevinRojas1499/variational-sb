import torch
import utils.sde_lib as SDEs

from utils.diff import batch_div_exact, hutch_div
from utils.misc import batch_matrix_product

#######################################
#          Diffusion Losses           #
#######################################

def dsm_loss(sde : SDEs.SDE,data, model):
    eps = sde.delta
    times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
    shaped_t = times.reshape(-1,1,1,1) if len(data.shape) > 2 else times.reshape(-1,1)
    mean, variance = sde.marginal_prob(data,shaped_t)
    noise = torch.randn_like(mean,device=data.device)
    perturbed_data = mean + variance**.5 * noise
    flatten_error = ((variance**.5 * model(perturbed_data,times) + noise)**2).view(data.shape[0],-1)
    
    return torch.mean(torch.sum(flatten_error,dim=1))

def linear_sb_loss(sde : SDEs.LinearSchrodingerBridge,data, model):
    eps = sde.delta
    times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
    shaped_t = times.reshape(-1,1,1,1) if len(data.shape) > 2 else times.reshape(-1,1)
    mean, L, invL, max_eig = sde.marginal_prob(data,shaped_t)
    noise = torch.randn_like(mean,device=data.device)
    perturbed_data = mean + batch_matrix_product(L, noise) 
    flatten_error = ((batch_matrix_product(invL.mT, noise) + model(perturbed_data,times))**2).view(data.shape[0],-1)
    
    return torch.mean(torch.sum(flatten_error,dim=1))

def cld_loss(sde : SDEs.CLD,data, model):
    eps = sde.delta
    times = (torch.rand((data.shape[0]) ,device=data.device) * (1-eps) + eps) * sde.T
    shaped_t = times.reshape(-1,1,1,1) if len(data.shape) > 2 else times.reshape(-1,1)
    ext_data = torch.cat((data,torch.randn_like(data)),dim=1) # Add velocity
    mean = sde.marginal_prob_mean(ext_data,shaped_t)
    lxx, lxv, lvv = sde.marginal_prob_std(shaped_t)
    noise = torch.randn_like(mean,device=data.device)
    perturbed_data = mean + sde.multiply_std(noise,shaped_t)
    flatten_error = ((model(perturbed_data,times) * lvv + noise.chunk(2,dim=1)[1])**2).view(data.shape[0],-1)
    
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
    time_pts_shaped = time_pts[:-1].repeat(batch_size)
    bt = sde.beta(time_pts_shaped)
    flat_traj = trajectories.reshape(-1,d)
    backward_scores =  bt**.5 * sde.backward_score(flat_traj,time_pts_shaped) # Backward policies as described in the FBSDE paper
    
    if sde.is_augmented:
        backward_scores*= sde.gamma**.5
        aug_backward_score = torch.cat((torch.zeros_like(backward_scores),backward_scores),dim=-1)
        div_term = (sde.gamma * bt)**.5 * hutch_div(aug_backward_score,flat_traj,time_pts_shaped) 
        # div_term = (sde.gamma * bt)**.5 * batch_div_exact(aug_backward_score,flat_traj,time_pts_shaped) 
    else:
        # div_term = bt**.5 * batch_div_exact(backward_scores,flat_traj,time_pts_shaped)
        div_term = bt**.5 * hutch_div(backward_scores,flat_traj,time_pts_shaped) 
        
    loss = .5 * torch.sum((backward_scores + forward_scores.view(-1,backward_scores.shape[-1]))**2)/batch_size \
      + torch.sum(div_term)/batch_size
    loss = dt * loss
    loss += .5 * torch.sum(xt**2)/batch_size
    return loss

def alternate_sb_loss(sde : SDEs.SchrodingerBridge, in_cond, time_pts):
    return 0

def standard_sb_loss(sde : SDEs.SchrodingerBridge, data, model=None):
    n_times = 100
    time_pts = torch.linspace(0., sde.T,n_times, device=data.device)
    
    if sde.is_augmented:
        v_noise = torch.randn_like(data)
        data = torch.cat((data,v_noise),dim=-1)

    return joint_sb_loss(sde,data,time_pts)

    
def get_loss(sde_name):
    if sde_name == 'vp':
        return dsm_loss
    if sde_name == 'sb':
        return standard_sb_loss
    elif sde_name == 'linear-sb':
        return linear_sb_loss
    elif sde_name == 'cld':
        return cld_loss