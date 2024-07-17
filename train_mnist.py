import torch
import click
import wandb
import itertools
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from utils.misc import dotdict

import utils.losses as losses
from utils.training_routines import VariationalDiffusionTrainingRoutineEpoch, VariationalDiffusionTrainingRoutine
from utils.sde_lib import SDE
from utils.model_utils import get_model
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
    fig.savefig(file_name,bbox_inches='tight')
    plt.close(fig)

def update_ema(model, model_ema, beta):
    for p_ema, p_net in zip(model_ema.parameters(), model.parameters()):
        p_ema.copy_(p_net.detach().lerp(p_ema, beta))

@torch.no_grad()
def copy_ema_to_model(model, model_ema):
    for p_ema, p_net in zip(model_ema.parameters(), model.parameters()):
        p_net.copy_(p_ema.detach())

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
@click.option('--model_forward',type=click.Choice(['linear','none']), default='none')
@click.option('--model_backward',type=click.Choice(['unet']), default='unet')
@click.option('--precondition', is_flag=True, default=True)
@click.option('--sde',type=click.Choice(['vp','cld','sb','edm', 'linear-sb','momentum-sb','linear-momentum-sb']), default='vp')
@click.option('--loss_routine', type=click.Choice(['none','variational']),default='none')
@click.option('--dsm_warm_up', type=int, default=2000, help='Number of epochs to perform warm up in')
@click.option('--dsm_cool_down', type=int, default=5000, help='Number of epochs for cool down')
@click.option('--forward_opt_steps', type=int, default=5, help='Number of forward opt steps in alternate training scheme')
@click.option('--backward_opt_steps', type=int, default=495, help='Number of backward opt steps in alternate training scheme')
# Training Options
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adam')
@click.option('--lr', type=float, default=3e-4)
@click.option('--ema_beta', type=float, default=.99)
@click.option('--clip_grads', is_flag=True, default=False)
@click.option('--batch_size', type=int, default=64)
@click.option('--num_epochs',type=int,default=30000)
@click.option('--dir',type=str)
@click.option('--load_from_ckpt', type=str)
def train(**opts):
    opts = dotdict(opts)
    print(opts)
    batch_size = opts.batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_sb = (opts.model_forward != 'none')
    sde = get_sde(opts.sde)
    sampling_sde = get_sde(opts.sde)
    model_backward, ema_backward = get_model(opts.model_backward, sde, device) 
    model_backward = model_backward.to(device)
    sde.backward_score = get_preconditioned_model(model_backward,sde)
    sampling_sde.backward_score = get_preconditioned_model(ema_backward, sde)
    
    if is_sb:
        network_opts = dotdict({
            'out_shape' : [1, 28, 28] 
        })
        model_forward, ema_forward = get_model(opts.model_forward, sde, device, network_opts)
        sde.forward_score = model_forward
        sampling_sde.forward_score = ema_forward
    
    # num_params_forward = 0
    num_params = sum(p.numel() for p in model_backward.parameters() if p.requires_grad)
    num_params_forward = sum(p.numel() for p in model_forward.parameters() if p.requires_grad)

    print("Number of parameters in the network:", num_params, num_params_forward)

    dataset = MNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Resize((28,28))]), download=True)
    init_wandb(num_samples=batch_size)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    data_loader = itertools.cycle(data_loader)
    optimizer = Adam(model_backward.parameters(), lr=opts.lr)
    
    loss_fn = losses.get_loss(opts.sde, False) 
    if opts.loss_routine == 'variational':
        routine = VariationalDiffusionTrainingRoutine(sde,sampling_sde,model_forward,model_backward,
                                                      opts.dsm_warm_up,opts.num_epochs-opts.dsm_warm_up - opts.dsm_cool_down, opts.dsm_cool_down,
                                                      opts.forward_opt_steps, opts.backward_opt_steps,100,device)
    tqdm_epoch = tqdm(range(opts.num_epochs))
    for i in tqdm_epoch:
        x, y = next(data_loader)
        x = x.to(device)  
        
        if opts.loss_routine == 'variational':
            loss = routine.training_iteration(i, x)
        else:
            loss = loss_fn(sde, x)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        wandb.log({'loss' : loss.item()})
        
        # Update EMA
        update_ema(model_backward, ema_backward, opts.ema_beta)
        if is_sb:
            update_ema(model_forward,  ema_forward, opts.ema_beta)
        if opts.loss_routine == 'variational':
            copy_ema_to_model(model_forward, ema_forward)
        
    # Update the checkpoint after each epoch of training.
    path = os.path.join(opts.dir, f'itr_{i+1}/')
    os.makedirs(path,exist_ok=True) # Still wondering it this is the best idea
        
    torch.save(model_backward.state_dict(),os.path.join(path, 'backward.pt'))
    
    create_sample_logs(sde,file_name=os.path.join(path,f'epoch {i}'))
    
    wandb.finish()
    
if __name__ == '__main__':
    train()
