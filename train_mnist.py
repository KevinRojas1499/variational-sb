import torch
import wandb
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from utils.model import ScoreNet
from utils.networks_edm2 import Precond
from utils.sde_lib import VP
from utils.losses import dsm_loss, fp_loss


def em_sampler(sde : VP, score_model,  
                           batch_size=32, 
                           num_steps=500, 
                           device='cuda',file_name='mnist_samples.jpeg'):
    ones = torch.ones(batch_size, device=device)
    time_pts = torch.linspace(0, sde.T() - sde.delta, num_steps, device=device)
    time_pts = torch.cat((time_pts, torch.tensor([sde.T()],device=device)))
    x_t = torch.randn(batch_size, 1, 32, 32, device=device)
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
            
    x_mean = x_mean.clip(0,1)
    plot_32_mnist(x_mean,file_name)    

    return x_mean

def plot_32_mnist(x_t,file_name='mnist_samples.jpeg'):
    n_rows, n_cols = 4,8
    fig, axs = plt.subplots(n_rows,n_cols)
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            im = x_t[idx].permute(1,2,0)
            axs[i][j].axis('off')
            axs[i][j].imshow(im.cpu().numpy(),vmin=0,vmax=1,cmap='grey')
            idx+=1
    fig.savefig(file_name)
    plt.close(fig)


def init_wandb(num_samples):
    wandb.init(
    # set the wandb project where this run will be logged
    name=f'fp-mine-{num_samples}',
    project='kinetic-fp',
    # name= get_run_name(config),
    tags= ['mnist','fp-quality'],
    # # track hyperparameters and run metadata
    # config=config
)

def train(n_epochs=50, batch_size=64, lr=1e-4):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    sde = VP()
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=sde.marginal_prob_std))
    # score_model = torch.nn.DataParallel(Precond(32,1))
    # score_model.load_state_dict(torch.load('checkpoints/mnist/ckpt49.pth'))
    score_model = score_model.to(device)

    num_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)

    print("Number of parameters in the network:", num_params)

    dataset = MNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Resize((32,32))]), download=True)
    init_wandb(num_samples=batch_size)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = Adam(score_model.parameters(), lr=lr)
    tqdm_epoch = tqdm(range(n_epochs))

    for i in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader: # Load data
            x = x.to(device)    
            loss = dsm_loss(sde, x, score_model)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            # Print the averaged training loss so far.
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            wandb.log({'loss' : loss.item()})
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), f'checkpoints/mnist/ckpt{i}.pth')
        em_sampler(sde, score_model,file_name=f'mnist/epoch {i}')
        
        
def eval_fp_loss(n_batches_fp, batch_size_fp):
    init_wandb(num_samples=n_batches_fp*batch_size_fp)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    sde = VP()
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=sde.marginal_prob_std))
    score_model.load_state_dict(torch.load('checkpoints/mnist/ckpt49.pth'))
    score_model = score_model.to(device)
    
    data_loader = DataLoader(dataset, batch_size=batch_size_fp, shuffle=True, num_workers=4)
    data_iterator = iter(data_loader)

    with torch.no_grad():
        for t in tqdm(torch.linspace(0.1,sde.T(),50, device=device)):
            fp_loss_ = 0
            n_items = 0 
            for _ in tqdm(range(n_batches_fp),leave=False):
                try:
                    data, labels = next(data_iterator) 
                except StopIteration:
                    data_iterator = iter(data_loader)
                    data, labels = next(data_iterator)
                    
                data = data.to(device)
                t_shape = t.expand(data.shape[0])
                mean, var  = sde.marginal_prob(data,t)
                perturbed_data = mean + var**.5 * torch.randn_like(mean)
                fp_loss_ += fp_loss(sde,perturbed_data,score_model,t_shape,approx_div=True).detach() * data.shape[0]
                n_items += data.shape[0]
            fp_loss_ /= n_items
            wandb.log({
                't': t,
                'fp_loss_t': fp_loss_
            })
        
train()
# sde = VP()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=sde.marginal_prob_std))
# score_model.load_state_dict(torch.load('checkpoints/mnist/ckpt49.pth'))
# score_model = score_model.to(device)

# em_sampler(sde, score_model)
# eval_fp_loss(50,256)

wandb.finish()