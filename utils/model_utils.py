import torch.nn as nn
import torch


from utils.sde_lib import SDE, VP, CLD, GeneralLinearizedSB
from utils.models import MLP, ToyPolicy, LinearMLP, MatrixTimeEmbedding
from utils.misc import batch_matrix_product


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
    def __init__(self, net, sde : GeneralLinearizedSB) -> None:
        super().__init__()
        self.net = net
        self.sde = sde
        
    def forward(self, zt,t,cond=None):
        xt,vt = zt.chunk(2,dim=1)
        cov, L, big_beta = self.sde.compute_variance(t)
        Ltinv = torch.linalg.inv(L.mT)
        score = self.net(zt,t,cond)
        cur_shape = zt.shape
        precond_score = batch_matrix_product(Ltinv,score.view(zt.shape[0],-1)).view(cur_shape)
        return precond_score

def get_model(name, sde : SDE, device):
    # Returns model, ema
    augment = sde.is_augmented
    if name == 'mlp':
        return MLP(2,augment).requires_grad_(True).to(device=device), \
            MLP(2, augment).requires_grad_(False).to(device=device)
    elif name == 'toy':
        return ToyPolicy().requires_grad_(True).to(device=device), \
            ToyPolicy().requires_grad_(False).to(device=device)
    elif name == 'linear':
        return MatrixTimeEmbedding(in_dim=4 if augment else 2, out_dim=2).requires_grad_(True).to(device=device), \
            MatrixTimeEmbedding(in_dim=4 if augment else 2, out_dim=2).requires_grad_(False).to(device=device)
    # elif name == 'linear':
    #     return LinearMLP(2,False).requires_grad_(True).to(device=device), \
    #         LinearMLP(2, False).requires_grad_(False).to(device=device)
            
def get_preconditioned_model(model, sde):
    if isinstance(sde, VP):
        return PrecondVP(model,sde)
    elif isinstance(sde,CLD):
        return PrecondCLD(model, sde)
    elif isinstance(sde,GeneralLinearizedSB):
        return PrecondGeneral(model, sde)