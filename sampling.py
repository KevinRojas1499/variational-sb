import os

import click
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.distributed as dist
from tqdm import tqdm
import PIL.Image
import wandb
from datasets.dataset_utils import get_dataset
from utils.metrics import get_w2
from utils.misc import dotdict
from utils.model_utils import get_model, get_preconditioned_model
from utils.sde_lib import get_sde
from utils.autoencoder import Autoencoder
from ema_pytorch import EMA

def init_wandb(opts):
    wandb.init(
        # set the wandb project where this run will be logged
        project='variational-sb',
        name= f'{opts.dataset}-{opts.sde}',
        tags= ['training',opts.dataset],
        # # track hyperparameters and run metadata
        # config=config
    )

def unprocess(x):
    return (x+1)/2 

def plot_32_mnist(x_t,file_name='mnist_samples.jpeg'):
    n_rows, n_cols = 4,8
    fig, axs = plt.subplots(n_rows,n_cols)
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            im = x_t[idx].permute(1,2,0)
            axs[i][j].axis('off')
            axs[i][j].imshow(im.clamp(0,1).cpu().numpy(),vmin=0,vmax=1,cmap='gray')
            idx+=1
    plt.tight_layout()
    fig.savefig(file_name,bbox_inches='tight')
    # plt.close(fig) # TODO : Why is this not working?

def get_dataset_type(name):
    if name in ['spiral','checkerboard']:
        return 'toy'
    elif name in ['mnist', 'cifar']:
        return 'image'
    else:
        return 'time-series'

def is_sb_sde(name):
    return (name in ['vsdm','linear-momentum-sb'])
    
@click.command()
@click.option('--dataset',type=click.Choice(['mnist','spiral','checkerboard','cifar']), default='cifar')
@click.option('--model_forward',type=click.Choice(['linear']), default='linear')
@click.option('--model_backward',type=click.Choice(['DiT','unet','mlp', 'linear']), default='unet')
@click.option('--encoder',type=str, default=None)
@click.option('--sde',type=click.Choice(['vp','cld','sb', 'vsdm','momentum-sb','linear-momentum-sb']), default='linear-momentum-sb')
@click.option('--damp_coef',type=float, default=1.)
@click.option('--num_steps', type=int, default=100)
@click.option('--num_samples', type=int)
@click.option('--batch_size', type=int, default=100)
@click.option('--seed', type=int, default=42)
@click.option('--dir',type=str)
@click.option('--make_plots', is_flag=True, default=False)
@click.option('--load_from_ckpt', type=str)
@click.option('--not_ema', is_flag=True, default=False)
def training(**opts):
    opts = dotdict(opts)
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = opts.seed * world_size + rank 
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    batch_size = opts.batch_size
    assert batch_size % world_size == 0, 'Batch size must be divisible by world size'
    assert opts.num_samples % batch_size == 0, 'Num samples must be divisible by world size'
    print(opts)
    print(f'Initializing {rank} with {device}')
    dataset, out_shape, label_dim = get_dataset(opts)
    is_sb = is_sb_sde(opts.sde)
    sde = get_sde(opts.sde)
    encode = opts.encoder is not None
    if encode:
        out_shape = [4, 16, 16]
        autoencoder = Autoencoder(opts.encoder).to(device)

    # Set up backwards model
    if sde.is_augmented:
        out_shape[0] *= 2 
    network_opts = dotdict({'out_shape' : out_shape, 'damp_coef' : opts.damp_coef})
    
    
    use_ema = not opts.not_ema
    model_backward = get_model(opts.model_backward,sde, device,label_dim=label_dim, network_opts=network_opts)
    print(f"Backward Model parameters: {sum(p.numel() for p in model_backward.parameters() if p.requires_grad)//1e6} M")
    if use_ema:
        model_backward = EMA(model_backward)
        state = torch.load(os.path.join(opts.load_from_ckpt, 'backward_ema.pt'), weights_only=True)
    else:
        state = torch.load(os.path.join(opts.load_from_ckpt, 'backward.pt'), weights_only=True)
        # state = torch.load(os.path.join(opts.load_from_ckpt), weights_only=True)
    updated_dict = {}
    for key, value in state.items():
        # Remove 'module.' prefix if it exists
        new_key = key.replace('module.', '')
        updated_dict[new_key] = value
    model_backward.load_state_dict(updated_dict)

    if is_sb:
        # We need a forward model
        model_forward  = get_model(opts.model_forward,sde,device,network_opts=network_opts)
        if use_ema:
            model_forward = EMA(model_forward)
            state_f = torch.load(os.path.join(opts.load_from_ckpt,'forward_ema.pt'), weights_only=True)
        else:
            state_f = torch.load(os.path.join(opts.load_from_ckpt,'forward.pt'), weights_only=True)
            # state_f = torch.load(os.path.join(opts.load_from_ckpt), weights_only=True)
        updated_dict_f = {}
        for key, value in state_f.items():
            # Remove 'module.' prefix if it exists
            new_key = key.replace('module.', '')
            updated_dict_f[new_key] = value
        model_forward.load_state_dict(updated_dict_f)
        print(f"Forward Model parameters: {sum(p.numel() for p in model_forward.parameters() if p.requires_grad)//1e6} M")
        sde.forward_score = model_forward

    sde.backward_score = get_preconditioned_model(model_backward, sde)
    
    batches = opts.num_samples// batch_size 
    effective_batch = opts.batch_size//world_size 
    for batch in tqdm(range(batches)):
        sampling_shape = (effective_batch, *network_opts.out_shape)
        cond = torch.randint(0,10,(effective_batch,), device=device)
        new_data, _ = sde.sample(sampling_shape, device, cond=cond, prob_flow=True, n_time_pts=opts.num_steps)
        new_data = unprocess(new_data)
        if encode:
            new_data = autoencoder.decode(new_data)
        folder = os.path.join(opts.dir, f'{batch}/{rank}')
        os.makedirs(folder, exist_ok=True)
        images_np = (new_data * 255 ).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for i in range(effective_batch):
            PIL.Image.fromarray(images_np[i], 'RGB').save(os.path.join(folder, f'{i}.png'))
        
        dist.barrier()
        
    dist.destroy_process_group()
    
if __name__ == '__main__':
    training()