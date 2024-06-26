import torch
import torch.nn as nn
import numpy as np
import math 

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
            nn.SiLU(),
            nn.Linear(128,256),
            nn.SiLU(),
            nn.Linear(256,128),
            nn.SiLU(),
            nn.Linear(128,self.dim)
        )
        
    def forward(self,x,t):
        h = torch.cat([x, t.reshape(-1, 1)], dim=1)
        
        return self.sequential(h)

class MatrixTimeEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sequential = nn.Sequential(
            nn.Linear(1,128),
            nn.SiLU(),
            nn.Linear(128,256),
            nn.SiLU(),
            nn.Linear(256,128),
            nn.SiLU(),
            nn.Linear(128,self.in_dim * out_dim)
        )
        self.register_buffer('id',torch.eye(out_dim).unsqueeze(0))
        
    def forward(self,t):
        t = t.flatten().unsqueeze(-1)
        return torch.zeros(t.shape[0],self.out_dim, self.in_dim, device=t.device)
        
        At = self.sequential(t).view(-1,self.out_dim,self.in_dim)
        At[:,:, :self.out_dim] *= self.id
        At[:,:, -self.out_dim:] *= self.id
        
        return torch.zeros_like(At)

class LinearMLP(nn.Module):
    def __init__(self, dim, augmented_sde) -> None:
        super().__init__()
        self.dim = dim
        self.true_dim = self.dim + 1
        if augmented_sde:
            self.true_dim += self.dim
        self.sequential = nn.Sequential(
            nn.Linear(1,128),
            nn.SiLU(),
            nn.Linear(128,256),
            nn.SiLU(),
            nn.Linear(256,128),
            nn.SiLU(),
            nn.Linear(128,dim * dim)
        )
        
    def forward(self,x,t):
        t = t.flatten().unsqueeze(-1)
        A = self.sequential(t).view(-1,self.dim,self.dim)
        
        return batch_matrix_product(A,x)
    
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

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class FCs(torch.nn.Module):
    def __init__(self, dim_in,dim_hid,dim_out):
        super(FCs,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_hid),
            nn.ReLU(),
            nn.Linear(dim_hid,dim_out),
            nn.Identity()
            )
    def forward(self,x):
        return self.model(x)

class ResNet_FC(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map=nn.Linear(data_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)])

    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        return linear

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths =[hid]*4
        for i in range(len(widths) - 1):
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h=self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / np.sqrt(2)
        return h
    
class ToyPolicy(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, time_embed_dim=128):
        super(ToyPolicy,self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module = ResNet_FC(data_dim, hidden_dim, num_res_blocks=3)

        self.out_module = nn.Sequential(
            nn.Linear(hid,hid),
            nn.SiLU(),
            nn.Linear(hid, data_dim),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self,x, t):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]
        elif len(t.shape) == 2:
            t=t.flatten()
            
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out   = self.out_module(x_out+t_out)

        return out