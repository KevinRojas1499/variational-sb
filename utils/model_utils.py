import torch.nn as nn
import torch
from typing import Union

from utils.sde_lib import SDE, VP, CLD, LinearMomentumSchrodingerBridge, LinearSchrodingerBridge
from utils.models import MLP, MatrixTimeEmbedding
from utils.unet import ScoreNet

class PrecondVP(nn.Module):
    def __init__(self, net, sde : VP) -> None:
        super().__init__()
        self.net = net
        self.sde = sde
        
    def forward(self, xt,t,cond=None):
        ones = [1] * (len(xt.shape)-1)
        return self.net(xt,t,cond)/self.sde.marginal_prob_std(t).view(-1,*ones)

class PrecondCLD(nn.Module):
    def __init__(self, net, sde : CLD) -> None:
        super().__init__()
        self.net = net
        self.sde = sde
        
    def forward(self, zt,t,cond=None):
        xt,vt = zt.chunk(2,dim=1)
        lxx, lxv, lvv = self.sde.marginal_prob_std(t)
        ones = [1] * (len(xt.shape)-1)
        
        return -vt/(lvv**2+lxv).view(-1,*ones) - self.net(zt,t,cond)/lvv.view(-1,*ones)

class PrecondGeneral(nn.Module):
    def __init__(self, net, sde : Union[LinearSchrodingerBridge,LinearMomentumSchrodingerBridge]) -> None:
        super().__init__()
        self.net = net
        self.sde = sde
        
    def forward(self, zt,t,cond=None):
        ones = [1] * (len(zt.shape)-1)
        with torch.no_grad():
            scale, L = self.sde.get_transition_params(zt,t.view(-1,*ones))

        if isinstance(self.sde, LinearMomentumSchrodingerBridge):
            xt,vt = zt.chunk(2,dim=1)
            lxx, lxv, lvv = L[...,0,0], L[...,0,1], L[...,1,1]
        
            return - self.net(zt,t,cond)/lvv
        
        return - self.net(zt,t,cond)/L

def get_model(name, sde : SDE, device, network_opts=None):
    print(network_opts)
    # Returns model, ema
    augment = sde.is_augmented
    if name == 'mlp':
        return MLP(2,augment).requires_grad_(True).to(device=device), \
            MLP(2, augment).requires_grad_(False).to(device=device)
    elif name == 'linear':
        return MatrixTimeEmbedding(out_shape=network_opts.out_shape).requires_grad_(True).to(device=device), \
            MatrixTimeEmbedding(out_shape=network_opts.out_shape).requires_grad_(False).to(device=device)
    elif name == 'unet':
        model = torch.nn.DataParallel(ScoreNet(in_channels=2 if sde.is_augmented else 1))
        ema = torch.nn.DataParallel(ScoreNet(in_channels=2 if sde.is_augmented else 1))
        return model.requires_grad_(True).to(device=device), ema.requires_grad_(False).to(device=device) 
    
def get_preconditioned_model(model, sde):
    if isinstance(sde, VP):
        return PrecondVP(model,sde)
    elif isinstance(sde,CLD):
        return PrecondCLD(model, sde)
    return PrecondGeneral(model,sde)