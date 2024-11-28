import torch.nn as nn
import torch
from typing import Union

from utils.sde_lib import SDE, VP, CLD, LinearMomentumSchrodingerBridge, VSDM
from utils.models import MLP, MatrixTimeEmbedding
from utils.unet import ScoreNet
from utils.networks import SongUNet, DhariwalUNet
from utils.DiT import DiT_S_2

class PrecondVP(nn.Module):
    def __init__(self, net, sde : VP) -> None:
        super().__init__()
        self.net = net
        self.sde = sde
    def forward(self, xt,t,cond=None):
        ones = [1] * (len(xt.shape)-1)
        sigma = self.sde.marginal_prob_std(t).view(-1,*ones)

        return -self.net(xt,t*999,cond)/sigma
class PrecondCLD(nn.Module):
    def __init__(self, net, sde : CLD) -> None:
        super().__init__()
        self.net = net
        self.sde = sde
        
    def forward(self, zt,t,cond=None):
        xt,vt = zt.chunk(2,dim=1)
        lxx, lxv, lvv = self.sde.marginal_prob_std(t)
        ones = [1] * (len(xt.shape[1:]))
        return -vt/(lvv**2+lxv).view(-1,*ones) - self.net(zt,t,cond)/lvv.view(-1,*ones)

class PrecondGeneral(nn.Module):
    def __init__(self, net, sde : Union[VSDM,LinearMomentumSchrodingerBridge]) -> None:
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
        
            return - self.net(zt,t * 999,cond)/lvv
        
        return - self.net(zt,t,cond)/L

def get_model(name, sde : SDE, device, label_dim=0, network_opts=None):
    print(network_opts)
    # Returns model, ema
    augmented = sde.is_augmented
    if name == 'mlp':
        return MLP(2,augmented).requires_grad_(True).to(device=device)
    elif name == 'linear':
        out_shape = network_opts.out_shape.copy()
        out_shape[0] = out_shape[0]//(2 if augmented else 1)
        gamma = sde.gamma if augmented else None
        return MatrixTimeEmbedding(out_shape,augmented,gamma, under_damp_coeff=network_opts.damp_coef).requires_grad_(True).to(device=device)
    elif name == 'unet':
        in_channels = network_opts.out_shape[0] 
        out_channels = in_channels//(2 if augmented else 1)
        # model = ScoreNet(in_channels=in_channels, out_channels=out_channels)
        model = DhariwalUNet(img_resolution=network_opts.out_shape[-1], in_channels=in_channels, out_channels=out_channels, label_dim=label_dim)
        # model = SongUNet(img_resolution=network_opts.out_shape[-1], in_channels=in_channels, out_channels=out_channels)
        return model.requires_grad_(True).to(device=device)
    elif name == 'DiT':
        in_channels = network_opts.out_shape[0] 
        out_channels = in_channels//(2 if augmented else 1)
        return DiT_S_2(in_channels=in_channels, out_channels=out_channels).to(device=device)
         
    
    
def get_preconditioned_model(model, sde):
    if isinstance(sde, VP):
        return PrecondVP(model,sde)
    elif isinstance(sde,CLD):
        return PrecondCLD(model, sde)
    return PrecondGeneral(model,sde)