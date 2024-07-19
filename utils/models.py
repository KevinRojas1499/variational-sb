import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 


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
        
        
        self.apply(self.zero_init)
        
    def zero_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            torch.nn.init.zeros_(m.bias)
        
    def forward(self,t):
        t = t.flatten().unsqueeze(-1)
        
        At = self.sequential(t)
        At = self.out(At).view(-1,*self.out_shape)
        return At

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
        
        combined = self.fc1(combined).unsqueeze(1)  # shape: [B, input_dim]
        out = torch.cat((combined,x),dim=-1)  # shape: [B, input_dim]
        out = self.fc2(out)  # shape: [B, input_dim]
        return out#.view(B,L,D)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, in_channels, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Sequential(nn.Conv1d(
            in_channels, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        ), nn.Linear(hidden_size+4,6))
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    #     nn.init.kaiming_normal_(self.conditioner_projection.weight)
    #     nn.init.kaiming_normal_(self.output_projection.weight)

    # def kaiming_normal_init(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.kaiming_normal_(module.weight)
        

    def forward(self, x, cond, t_embed):
        t_embed = self.diffusion_projection(t_embed).unsqueeze(-1)
        cond = self.conditioner_projection(cond)
        y = x + t_embed
        y = self.dilated_conv(y) + cond

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip

class TimeSeriesNetwork(nn.Module):
    def __init__(self, input_dim, pred_length, cond_length, hidden_dim=64, 
                 residual_channels=32,dilation_cycle_length=2, num_residual_layers=8):
        super(TimeSeriesNetwork, self).__init__()
        
        self.cond_encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=6,batch_first=True)
        self.input_projection = nn.Sequential(
            nn.Conv1d(pred_length, residual_channels, 1, padding=2, padding_mode="circular"),
            nn.SiLU())                        
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(in_channels=cond_length,residual_channels=residual_channels,
                           dilation=2**(i%dilation_cycle_length),
                           hidden_size=hidden_dim)
                for i in range(num_residual_layers)
             ]
        )
        self.outtt = nn.Conv1d(residual_channels,pred_length,1)
        self.out = nn.Sequential(
            nn.Linear((input_dim + 4),128), # + 4 is coming from the padding in the convolutional layers
                                 nn.SiLU(),
                                 nn.Linear(128, input_dim))

        self.t_embedding =  nn.Sequential(
            nn.Linear(1,128),
            nn.SiLU(),
            nn.Linear(128,256),
            nn.SiLU(),
            nn.Linear(256,128),
            nn.SiLU(),
            nn.Linear(128,hidden_dim)
        )

    def forward(self, x, t, cond):
        # x    : [B, pred_length, D]
        # t    : [B]
        # cond : [B, cond_length, D]
        # out  : [B, pred_length, D]
        # Encode Input
        x = self.input_projection(x) # Encode input
        
        # Encode time TODO : Do better encoding
        t = t.view(-1,1)
        t_embedded = self.t_embedding(t)  # shape: [B, t_embedding_dim]
        
        #Encode condition
        out, (hn, cn) = self.cond_encoder(cond)
        encoded_cond = out
        skip = []
        for layer in self.residual_blocks:
            x, skip_connection = layer(x, encoded_cond, t_embedded)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(skip))
        x = self.outtt(x)
        x = self.out(x)
        return x