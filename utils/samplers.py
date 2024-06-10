import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.models import MLP
from utils.sde_lib import SDE, CLD

def plot_trajectory(x_t, i, t):
    lim = 10
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    plt.scatter(x_t[:,0].cpu(), x_t[:,1].cpu(),s=2)
    plt.savefig(f'./trajectory/{i}_{t.item() : .3f}.png')
    plt.close()

def em_sampler(sde : SDE, score_model, 
                sampling_shape,
                batch_size=32, 
                num_steps=500, 
                device='cuda'):
    ones = torch.ones(batch_size, device=device)
    time_pts = torch.linspace(0, sde.T() - sde.delta, num_steps, device=device)
    time_pts = torch.cat((time_pts, torch.tensor([sde.T()],device=device)))
    x_t = sde.prior_sampling(sampling_shape)
    T = sde.T()
    with torch.no_grad():
        for i in tqdm(range(num_steps), leave=False):
            # plot_32_mnist(x_t,f'trajectory/{i}_mnist.jpeg')
            dt = time_pts[i+1] - time_pts[i]
            
            t = ones * time_pts[i]
            score = score_model(x_t, T - t)
            # e_h = torch.exp(sde.beta_int(time_pts[i+1]) - sde.beta_int(time_pts[i]))
            beta = sde.beta(T - time_pts[i]) 
            # exponential integrator step
            # x_t = e_h * x_t + 2 * (e_h - 1) * score + (e_h**2 - 1)**.5 * torch.randn_like(x_t)
            x_mean = x_t + (.5 * beta * x_t + beta * score) * dt
            x_t = x_mean + (beta * dt)**.5 * torch.randn_like(x_t)
            
    return x_mean

def get_euler_maruyama(num_samples, sde, model, dim, device):
    with torch.no_grad():
        x_t = sde.prior_sampling((num_samples,dim),device=device)

        time_pts = sde.time_steps(100, device)
        pbar = tqdm(range(len(time_pts) - 1),leave=False)
        T = sde.T()
        for i in pbar:
            t = time_pts[i].expand(num_samples,1)
            dt = time_pts[i + 1] - t
            score = model(x_t, T- t)
            diffusion = sde.diffusion(x_t,T - t)
            tot_drift = - sde.drift(x_t,T - t) + diffusion**2 * score
            # euler-maruyama step    print(samples.shape)

            x_t += tot_drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
            
            # plot_trajectory(x_t, i, time_pts[i])
        pbar.close()
        return x_t

def get_exponential_integrator(num_samples, sde, model : MLP, device):

    x_t = sde.prior_sampling((num_samples,2),device=device)

    time_pts = sde.time_steps(25, device)
    T = sde.T()
    pbar = tqdm(range(len(time_pts) - 1),leave=False)
    for i in pbar:
        t = time_pts[i].unsqueeze(-1).expand(num_samples,-1)
        dt = time_pts[i+1] - time_pts[i]
        score = model(x_t, T - t)
        e_h = torch.exp(dt)
        # exponential integrator step
        x_t = e_h * x_t + 2 * (e_h - 1) * score + ((e_h**2 - 1))**.5 * torch.randn_like(x_t)
        # plot_trajectory(x_t, i, t)
    pbar.close()
    return x_t


def get_cld_euler(sde : CLD, score_model, 
                sampling_shape,
                batch_size=32, 
                num_steps=500, 
                device='cuda'):
    z_t = sde.prior_sampling(shape=sampling_shape,device=device)
    z_x, z_v = z_t.chunk(2,dim=1)
    time_pts = sde.time_steps(num_steps, device)
    time_pts = torch.linspace(sde.delta, sde.T(), num_steps, device=device)
    T = sde.T()
    pbar = tqdm(range(len(time_pts) - 1),leave=False)
    for i in pbar:
        t = time_pts[i].unsqueeze(-1)
        b = sde.beta_int(t)
        dt = time_pts[i+1] - time_pts[i]
        score = score_model(z_t, (T - t))
        score_x, score_v = score.chunk(2,dim=1)

        n_z_x = - z_v * dt
        z_v = b * (z_x + sde.gamma * z_v + 2 * sde.gamma * score_v) * dt + b**.5 * (2 * dt * sde.gamma)**.5 * torch.randn_like(z_v)
        z_x = n_z_x
    pbar.close()
    return z_x