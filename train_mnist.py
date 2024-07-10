import torch
import click
import wandb
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from utils.misc import dotdict

import utils.losses as losses
from utils.unet import ScoreNet
from utils.sde_lib import SDE
from utils.losses import dsm_loss
from utils.model_utils import get_preconditioned_model
from utils.sde_lib import get_sde


def create_sample_logs(sde : SDE, 
                        batch_size=32, 
                        device='cuda',file_name='mnist_samples.jpeg'):
    x_mean, traj = sde.sample((batch_size,2 if sde.is_augmented else 1,28,28),device,return_traj=False)
    plot_32_mnist(x_mean[:,0:1],file_name)    

    return x_mean

def plot_32_mnist(x_t,file_name='mnist_samples.jpeg'):
    n_rows, n_cols = 4,8
    fig, axs = plt.subplots(n_rows,n_cols)
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            im = x_t[idx].permute(1,2,0)
            axs[i][j].axis('off')
            axs[i][j].imshow(im.cpu().numpy(),vmin=0,vmax=1,cmap='gray')
            idx+=1
    fig.savefig(file_name)
    plt.close(fig)


def init_wandb(num_samples):
    wandb.init(
    # set the wandb project where this run will be logged
    name=f'variational-{num_samples}',
    project='variational-sb',
    # name= get_run_name(config),
    tags= ['mnist'],
    # # track hyperparameters and run metadata
    # config=config
)


@click.command()
@click.option('--dataset',type=click.Choice(['mnist']),default='mnist')
@click.option('--model_forward',type=click.Choice(['unet']), default='unet')
@click.option('--model_backward',type=click.Choice(['unet']), default='unet')
@click.option('--precondition', is_flag=True, default=True)
@click.option('--sde',type=click.Choice(['vp','cld','sb','edm', 'linear-sb','momentum-sb','linear-momentum-sb']), default='vp')
@click.option('--loss_routine', type=click.Choice(['joint','alternate','variational']),default='alternate')
@click.option('--dsm_warm_up', type=int, default=2000, help='Perform first iterations using just DSM')
@click.option('--dsm_cool_down', is_flag=True, default=False, help='Perform last iterations using just DSM for Variational Scores')
@click.option('--forward_opt_steps', type=int, default=100, help='Number of forward opt steps in alternate training scheme')
@click.option('--backward_opt_steps', type=int, default=100, help='Number of backward opt steps in alternate training scheme')
# Training Options
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adam')
@click.option('--lr', type=float, default=3e-4)
@click.option('--ema_beta', type=float, default=.99)
@click.option('--clip_grads', is_flag=True, default=False)
@click.option('--batch_size', type=int, default=64)
@click.option('--num_epochs',type=int,default=50)
@click.option('--dir',type=str)
@click.option('--load_from_ckpt', type=str)
def train(**opts):
    opts = dotdict(opts)
    print(opts)
    batch_size = opts.batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sde = get_sde(opts.sde)
    score_model = torch.nn.DataParallel(ScoreNet(in_channels=2 if sde.is_augmented else 1))
    # score_model = torch.nn.DataParallel(Precond(32,1))
    # score_model.load_state_dict(torch.load('checkpoints/mnist/ckpt49.pth'))
    score_model = score_model.to(device)
    sde.backward_score = get_preconditioned_model(score_model,sde)
    num_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)

    print("Number of parameters in the network:", num_params)

    dataset = MNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Resize((28,28))]), download=True)
    init_wandb(num_samples=batch_size)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = Adam(score_model.parameters(), lr=opts.lr)
    
    loss_fn = losses.get_loss(opts.sde, False, False) 
    
    tqdm_epoch = tqdm(range(opts.num_epochs))

    for i in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader: # Load data
            x = x.to(device)    
            loss = loss_fn(sde, x)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            # Print the averaged training loss so far.
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            wandb.log({'loss' : loss.item()})
        # Update the checkpoint after each epoch of training.
        path = os.path.join(opts.dir, f'itr_{i+1}/')
        os.makedirs(path,exist_ok=True) # Still wondering it this is the best idea
            
        torch.save(score_model.state_dict(),os.path.join(path, 'backward.pt'))
        
        create_sample_logs(sde,file_name=os.path.join(path,f'epoch {i}'))
        
    wandb.finish()
    
if __name__ == '__main__':
    train()
