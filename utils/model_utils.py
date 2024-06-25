import torch.nn as nn

from utils.sde_lib import SDE, VP
from utils.models import MLP, ToyPolicy, LinearMLP, MatrixTimeEmbedding


class Precond(nn.Module):
    def __init__(self, net, sde : VP) -> None:
        super().__init__()
        self.net = net
        self.sde = sde
        
    def forward(self, xt,t):
        return self.net(xt,t)/self.sde.marginal_prob_std(t)

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
    return Precond(model,sde)