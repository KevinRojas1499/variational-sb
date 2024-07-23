import torch
from torch import nn, Tensor

import sys
import os

# Add the parent directory to the sys.path to ensure it can find hello.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sde_lib import VP

def build_linear(zero_out_last_layer):
    return LinearPolicy(zero_out_last_layer=zero_out_last_layer)


class LinearPolicy(nn.Module):
    def __init__(
        self,
        data_dim,
        beta_min: float,
        beta_max: float,
        beta_r: float,
        interval: int,
    ):
        super(LinearPolicy,self).__init__()

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_r = beta_r
        self.interval = interval
        
        self.sde = VP()

        self.A = nn.Parameter(torch.zeros(data_dim, data_dim))

        self.Sigma = nn.Parameter(torch.zeros(data_dim))
        self.U = nn.utils.parametrizations.orthogonal(nn.Linear(data_dim, data_dim, bias=False))
        self.V = nn.utils.parametrizations.orthogonal(nn.Linear(data_dim, data_dim, bias=False))
        self.U.weight = torch.eye(data_dim)
        self.V.weight = torch.eye(data_dim)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    @property
    def A(self):
        Sigma_mat = torch.diag(self.Sigma)
        return self.U.weight @ Sigma_mat @ self.V.weight.T

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """
        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]

        out = x @ self.A.T

        diffusion = self.sde.diffusion(None,t).view(-1, *[1]*(out.ndim - 1))

        out = out * diffusion
        return out
