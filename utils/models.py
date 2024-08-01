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
    def __init__(self, out_shape):
        super(MatrixTimeEmbedding,self).__init__()
        self.out_shape = out_shape
        self.real_dim = np.prod(out_shape)
        self.ones = [-1] * len(self.out_shape)

        self.Lambda = nn.Parameter(torch.zeros(self.real_dim))
        

    def forward(self, t):
        A = self.Lambda.reshape(1, *self.out_shape).expand(tuple([t.shape[0], *self.ones]))
        return A
