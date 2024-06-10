import torch
import matplotlib.pyplot as plt
from utils.sde_lib import LinearSchrodingerBridge



device = 'cpu'

sde = LinearSchrodingerBridge(2, device=device, T=.5)



t = torch.linspace(0.1,.8,5,device=device).unsqueeze(-1)

print(sde.A.A.weight)

print(1 - torch.exp(- sde.beta_int(t)))
print(sde.compute_variance(t)[0])
noise = sde.prior_sampling((1000,2),device)

# plt.scatter(noise[:,0].detach(),noise[:,1].detach())
# plt.show()