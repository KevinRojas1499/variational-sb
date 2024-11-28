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
        self.first = nn.Sequential(
            nn.Linear(self.true_dim,256),
        )
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(256,256), nn.SiLU()) for i in range((6))]
        )
        self.out = nn.Linear(256,self.dim)
        
    def forward(self,x,t,cond=None):
        h = torch.cat([x, t.reshape(-1, 1)], dim=1)
        x = self.first(h)
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)

class MatrixTimeEmbedding(torch.nn.Module):
    def __init__(self, out_shape, is_augmented=False, gamma=None, under_damp_coeff=1.):
        super(MatrixTimeEmbedding,self).__init__()
        self.out_shape = out_shape
        self.real_dim = np.prod(out_shape)
        self.gamma = gamma
        self.is_augmented = is_augmented
        self.under_damp_coeff = under_damp_coeff
        self.ones = [-1] * len(self.out_shape)
        self._lambda = nn.Parameter(.01 *torch.ones(self.real_dim))
    
    @property
    def Lambda(self):
        if self.is_augmented:
            lamb = self._lambda.reshape(1, *self.out_shape)
            lamb_v = .5 - 1/self.gamma * (self.under_damp_coeff * (1-2 * self.gamma * lamb)).sqrt()
            lamb_x = (1- 1/self.under_damp_coeff * (self.gamma - 2 * self.gamma * lamb)**2/4)/(2*self.gamma) 
            return torch.cat((lamb_x,lamb),dim=1)
            return torch.cat((lamb,lamb_v),dim=1)
        
        return self._lambda.reshape(1, *self.out_shape)
    
    def forward(self, t):
        A = self.Lambda.expand(tuple([t.shape[0], *self.ones]))
        return A
