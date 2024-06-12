import torch
import matplotlib.pyplot as plt
from utils.sde_lib import LinearSchrodingerBridge



device = 'cpu'

# VP Case
# dim = 2
# sde = LinearSchrodingerBridge(dim, device=device, T=.5)
# t = torch.linspace(0.1,.8,5,device=device).unsqueeze(-1)
# sde.A.A.weight.data.copy_(torch.zeros((dim,dim)).view(-1,1))

# print(sde.A.A.weight.view(dim,dim))

# print(sde.beta_int(t))
# print(sde.int_beta_ds(t))
# print(torch.exp(-.5 * sde.beta_int(t)))
# print(sde.exp_int(t))

# print(1 - torch.exp(- sde.beta_int(t)))
# print(sde.compute_variance(t)[0])
# noise = sde.prior_sampling((1000,dim),device)

# plt.scatter(noise[:,0].detach(),noise[:,1].detach())
# plt.show()


# VE Case
print("VARIANCE EXPLODING")
dim = 2
sde = LinearSchrodingerBridge(dim, device=device, T=.5)
t = torch.linspace(0.1,.8,5,device=device).unsqueeze(-1)
sde.D.A.bias.data.copy_(torch.eye(dim).view(-1) * 1/2)


print(sde.D.A.weight.view(dim,dim))
print(sde.D.A.bias.data.view(dim,dim))

print(sde.D(t))

print(sde.beta_int(t))
print(sde.compute_variance(t)[0])
noise = sde.prior_sampling((1000,dim),device)

# plt.scatter(noise[:,0].detach(),noise[:,1].detach())
# plt.show()