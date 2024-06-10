import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim, augmented_sde) -> None:
        super().__init__()
        self.dim = dim
        self.true_dim = self.dim + 1
        if augmented_sde:
            self.true_dim += self.dim
        self.sequential = nn.Sequential(
            nn.Linear(self.true_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,self.true_dim - 1)
        )
        
    def forward(self,x,t):
        h = torch.cat([x, t.reshape(-1, 1)], dim=1)
        
        return self.sequential(h)