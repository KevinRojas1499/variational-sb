import torch
import utils.sde_lib as SDEs

from utils.misc import batch_matrix_product

def dsm_loss(sde : SDEs.SDE,data, model):
    eps = sde.delta
    times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
    shaped_t = times.reshape(-1,1,1,1) if len(data.shape) > 2 else times.reshape(-1,1)
    mean, variance = sde.marginal_prob(data,shaped_t)
    noise = torch.randn_like(mean,device=data.device)
    perturbed_data = mean + variance**.5 * noise
    flatten_error = ((variance**.5 * model(perturbed_data,times) + noise)**2).view(data.shape[0],-1)
    
    return torch.mean(torch.sum(flatten_error,dim=1))

def standard_sb_loss(sde : SDEs.SchrodingerBridge, data, model=None):
    n_times = 100
    time_pts = torch.linspace(0., sde.T,n_times, device=data.device)

    return sde.eval_sb_loss(data,time_pts)

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
    
def get_loss(sde_name):
    if sde_name == 'vp':
        return dsm_loss
    if sde_name == 'sb':
        return standard_sb_loss
    elif sde_name == 'linear-sb':
        return linear_sb_loss
    elif sde_name == 'cld':
        return cld_loss