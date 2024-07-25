import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, dim, augmented_sde) -> None:
        super().__init__()
        self.dim = dim
        self.true_dim = self.dim + 1
        if augmented_sde:
            self.true_dim += self.dim
        self.sequential = nn.Sequential(
            nn.Linear(self.true_dim,128),
            nn.SiLU(),
            nn.Linear(128,256),
            nn.SiLU(),
            nn.Linear(256,128),
            nn.SiLU(),
            nn.Linear(128,self.dim)
        )
        
    def forward(self,x,t,cond=None):
        h = torch.cat([x, t.reshape(-1, 1)], dim=1)
        
        return self.sequential(h)

class MatrixTimeEmbedding(nn.Module):
    def __init__(self,out_shape) -> None:
        super().__init__()
        self.diagonal = False
        self.out_shape = out_shape
        self.real_dim = np.prod(out_shape)
        self.sequential = nn.Sequential(
            nn.Linear(1,128,bias=False),
            nn.SiLU(),
            nn.Linear(128,256,bias=False),
            nn.SiLU(),
            nn.Linear(256,128,bias=False),
            nn.SiLU()
        )
        self.out = nn.Linear(128, self.real_dim,bias=False)
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,std=0.01)
             
        
    def forward(self,t):
        t = t.flatten().unsqueeze(-1)
        
        At = self.sequential(t)
        At = self.out(At).view(-1,*self.out_shape)
        return nn.functional.relu(At)
