import torch
import torch.nn as nn
from copy import deepcopy



class EMA(nn.Module):
    def __init__(self, model, beta=0.9999):
        super().__init__()
        self.beta = beta
        self.ema = deepcopy(model).eval()
        self.requires_grad(False)


    @torch.no_grad()
    def update(self, model, beta=None):
        beta = self.beta if beta is None else beta
        for p_net, p_ema in zip(model.parameters(), self.ema.parameters()):
            p_ema.lerp_(p_net, 1 - beta)
    
    def requires_grad(self, flag=True):
        """
        Set requires_grad flag for all parameters in a model.
        """
        for p in self.ema.parameters():
            p.requires_grad = flag
    
    def forward(self, *args, **kwargs):
        return self.ema(*args, **kwargs)