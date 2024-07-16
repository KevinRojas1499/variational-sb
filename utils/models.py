import torch
import torch.nn as nn
import numpy as np
import math 

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
            nn.Linear(1,128),
            nn.SiLU(),
            nn.Linear(128,256),
            nn.SiLU(),
            nn.Linear(256,128),
            nn.SiLU()
        )
        self.out = nn.Linear(128, self.real_dim)
        
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)
        
    def forward(self,t):
        t = t.flatten().unsqueeze(-1)
        
        At = self.sequential(t)
        At = self.out(At).view(-1,*self.out_shape)
        return At

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
        
    def forward(self,x,t,cond=None):
        t = t.flatten().unsqueeze(-1)
        A = self.sequential(t).view(-1,self.dim,self.dim)
        
        return batch_matrix_product(A,x)

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

class SimpleNN(nn.Module):
    def __init__(self, input_dim, cond_input_dim, hidden_dim, t_embedding_dim):
        super(SimpleNN, self).__init__()
        self.cond_encoder = nn.LSTM(input_size=cond_input_dim, hidden_size=hidden_dim, num_layers=3,batch_first=True)
        self.fc1 = nn.Linear(hidden_dim + t_embedding_dim, input_dim)
        self.fc2 =  nn.Sequential(
            nn.Linear(2 * input_dim,128),
            nn.SiLU(),
            nn.Linear(128,256),
            nn.SiLU(),
            nn.Linear(256,256),
            nn.SiLU(),
            nn.Linear(256,256),
            nn.SiLU(),
            nn.Linear(256,128),
            nn.SiLU(),
            nn.Linear(128,input_dim)
        )
        self.t_embedding =  nn.Sequential(
            nn.Linear(1,128),
            nn.SiLU(),
            nn.Linear(128,256),
            nn.SiLU(),
            nn.Linear(256,128),
            nn.SiLU(),
            nn.Linear(128,t_embedding_dim)
        )

    def forward(self, x, t, cond):
        # x    : [B, D]
        # t    : [B, 1]
        # cond : [B, cond_length, D]
        # out  : [B, D]
        out, (hn, cn) = self.cond_encoder(cond)
        encoded_cond = hn[-1]  # shape: [B, hidden_dim]
        t = t.view(-1,1)
        t_embedded = self.t_embedding(t)  # shape: [B, t_embedding_dim]
        combined = torch.cat((encoded_cond, t_embedded), dim=1)  # shape: [B, hidden_dim + t_embedding_dim]
        
        combined = self.fc1(combined)  # shape: [B, input_dim]
        
        out = torch.cat((combined,x),dim=-1)  # shape: [B, input_dim]
        out = self.fc2(out)  # shape: [B, input_dim]
        return out
