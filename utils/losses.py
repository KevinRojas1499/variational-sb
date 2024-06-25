import torch
import utils.sde_lib as SDEs
from math import pi, log

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
    # We assume that time_pts is a discretization of the interval [0,T]
    n_time_pts = time_pts.shape[0]
    
    xt = in_cond.detach().clone().requires_grad_(True)
    batch_size, d = xt.shape[0], xt.shape[-1]
    loss = 0
    trajectories = torch.empty((in_cond.shape[0], n_time_pts-1, *in_cond.shape[1:]),device=in_cond.device) 
    # forward_scores = torch.empty((in_cond.shape[0], n_time_pts-1, in_cond.shape[-1]//2),device=in_cond.device) 
    forward_scores = torch.empty_like(trajectories)
    for i, t in enumerate(time_pts):
      if i == n_time_pts - 1:
        break
      t_shape = t.unsqueeze(-1).expand(batch_size,1)
      
      bt = sde.beta(t)
      forward_score = bt * sde.forward_score(xt,t_shape) # beta * fw_score
        
      trajectories[:,i] = xt
      forward_scores[:,i] = forward_score
      
      # First we compute everything, we then take an euler step
      dt = time_pts[i+1] - t
      xt = xt + (-.5 * bt * xt + forward_score) * dt + torch.randn_like(xt) * sde.diffusion(xt,t) * dt.abs().sqrt()
    
    # We now compute the loss fn, we first need to do some reshaping
    time_pts_shaped = time_pts[:-1].repeat(batch_size)
    bt = sde.beta(time_pts_shaped)
    flat_traj = trajectories.reshape(-1,xt.shape[-1])
    backward_scores = bt * sde.backward_score(flat_traj,time_pts_shaped)
    
    # div_term = bt * batch_div_exact(backward_scores,flat_traj,time_pts_shaped) + .5 * d * bt 
    div_term = bt * hutch_div(backward_scores,flat_traj,time_pts_shaped) + .5 * d * bt 
    loss = .5 * torch.sum((backward_scores + forward_scores.view(-1,xt.shape[-1]))**2)/batch_size \
      + torch.sum(div_term)/batch_size
    loss = dt * loss
    loss += .5 * torch.sum(xt**2)/batch_size + .5 * d * log(2*pi)
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