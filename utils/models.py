import torch
import torch.nn as nn

from utils.sde_lib import LinearSchrodingerBridge
from utils.misc import batch_matrix_product

class MLP(nn.Module):
    def __init__(self, dim, augmented_sde) -> None:
        super().__init__()
        self.dim = dim
        self.true_dim = self.dim + 1
        if augmented_sde:
            self.true_dim += self.dim
        self.sequential = nn.Sequential(
            nn.Linear(self.true_dim,128),
            nn.Sigmoid(),
            nn.Linear(128,128),
            nn.Sigmoid(),
            nn.Linear(128,128),
            nn.SiLU(),
            nn.Linear(128,self.true_dim - 1)
        )
        
    def forward(self,x,t):
        h = torch.cat([x, t.reshape(-1, 1)], dim=1)
        
        return self.sequential(h)
    
class SB_Preconditioning(nn.Module):
    def __init__(self, model, sde : LinearSchrodingerBridge) -> None:
        super().__init__()
        self.model = model
        self.sde = sde
    
    def forward(self, x, t):
        t_shape = t.reshape(-1,1)

        invL = self.sde.compute_variance(t_shape)[2]
        
        denoiser = self.model(x,t_shape)
        return -batch_matrix_product(invL.mT, denoiser)
    
        # t_shape = t.reshape(-1,1)
        # matrix = -.5 * self.sde.int_beta_ds(t_shape)
        # exp = matrix.matrix_exp()
        # inv = (-matrix).matrix_exp()
        
        # invL = self.sde.compute_variance(t_shape)[2]
        
        # denoiser = self.model(batch_matrix_product(inv,x),t_shape)
        # return batch_matrix_product(invL.mT, batch_matrix_product(exp, denoiser) - x)
    