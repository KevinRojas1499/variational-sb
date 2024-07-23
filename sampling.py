import os
import torch
import click
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from math import ceil
from tqdm import tqdm

from utils.sde_lib import get_sde
from utils.model_utils import get_model, get_preconditioned_model
from datasets.dataset_utils import get_dataset
from utils.metrics import get_w2
from utils.misc import dotdict

def init_wandb(opts):
    wandb.init(
        # set the wandb project where this run will be logged
        project='variational-sb',
        name= f'{opts.dataset}-{opts.sde}',
        tags= ['training',opts.dataset],
        # # track hyperparameters and run metadata
        # config=config
    )

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
    plt.tight_layout()
    fig.savefig(file_name,bbox_inches='tight')
    # plt.close(fig) # TODO : Why is this not working?

def get_dataset_type(name):
    if name in ['spiral','checkerboard']:
        return 'toy'
    elif name in ['mnist']:
        return 'image'
    else:
        return 'time-series'

def is_sb_sde(name):
    return (name in ['sb','linear-sb','momentum-sb','linear-momentum-sb'])
    
def update_ema(model, model_ema, beta):
    for p_ema, p_net in zip(model_ema.parameters(), model.parameters()):
        p_ema.copy_(p_net.detach().lerp(p_ema, beta))

@torch.no_grad()
def copy_ema_to_model(model, model_ema):
    for p_ema, p_net in zip(model_ema.parameters(), model.parameters()):
        p_net.copy_(p_ema.detach())

def default_num_iters(ctx, param, value):
    sde = ctx.params.get('sde')
    if value is not None: 
        return value
    return 2000 if is_sb_sde(sde) else 10000
def default_log_rate(ctx, param, value):
    sde = ctx.params.get('sde')
    if value is not None: 
        return value
    return 2000 if is_sb_sde(sde) else 2000

@click.command()
@click.option('--dataset',type=click.Choice(['mnist','spiral','checkerboard','exchange_rate','electricity_nips']))
@click.option('--model_forward',type=click.Choice(['mlp','linear']), default='mlp')
@click.option('--model_backward',type=click.Choice(['mlp','unet', 'linear','time-series']), default='mlp')
@click.option('--sde',type=click.Choice(['vp','cld','sb','edm', 'linear-sb','momentum-sb','linear-momentum-sb']), default='vp')
# Training Options
@click.option('--batch_size', type=int, default=128)
@click.option('--dir',type=str)
@click.option('--load_from_ckpt', type=str)
def training(**opts):
    opts = dotdict(opts)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)
    print(device)
    dataset = get_dataset(opts)
    dataset_type = get_dataset_type(opts.dataset)
    is_sb = is_sb_sde(opts.sde)
    sde = get_sde(opts.sde)
    # Set up backwards model
    network_opts = dotdict({
        'out_shape' : dataset.out_shape,
        'cond_length' : dataset.metadata['cond_length']
    })
    model_backward, _ = get_model(opts.model_backward,sde, device,network_opts=network_opts)
    model_backward.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'backward.pt')))
    print(f"Backward Model parameters: {sum(p.numel() for p in model_backward.parameters() if p.requires_grad)//1e6} M")

    if is_sb:
        # We need a forward model
        model_forward , _  = get_model(opts.model_forward,sde,device,network_opts=network_opts)
        model_forward.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'forward.pt')))
        print(f"Forward Model parameters: {sum(p.numel() for p in model_backward.parameters() if p.requires_grad)//1e6} M")
    
    sde.backward_score = get_preconditioned_model(model_backward, sde)
    
    init_wandb(opts)
    
    if dataset_type == 'time-series':
        future, past = dataset.__next__(train=False)
        future = future.to(device)
        past = past.to(device)
        pred = create_diffusion_time_series_prediction(24,network_opts.out_shape[0],past,sde)
        plot_time_series(past,future,[pred],['pred'])
    else:
        sampling_shape = (opts.batch_size, *network_opts.out_shape)
        new_data, _ = sde.sample(sampling_shape, device)
        if dataset_type == 'toy':
            relevant_log_info = toy_data_figs([new_data], ['normal'])
            wandb.log(relevant_log_info)
        elif dataset_type == 'image':
            plot_32_mnist(new_data,os.path.join(opts.dir,'samples.png'))

            
    wandb.finish()

@torch.no_grad()
def create_diffusion_time_series_prediction(tot_pred_length, pred_size, past, sde):
    k = ceil(tot_pred_length//pred_size)
    prediction = torch.zeros(past.shape[0],pred_size * k,past.shape[-1], device=past.device)
    cur_past = past.detach().clone()
    for i in tqdm(range(k)):
        prediction[:,pred_size * i:pred_size * (i+1)], traj = sde.sample((past.shape[0],pred_size, past.shape[-1]),past.device,cond=cur_past,n_time_pts=25) 
        cur_past = torch.cat((cur_past[:,pred_size:],prediction[:,pred_size * i:pred_size * (i+1)]),dim=1)
    return prediction

def plot_time_series(past,future, prediction_array, names):
    figure = go.Figure()
    idx = torch.arange(past.shape[1] + 30)
    figure.add_vline(past.shape[1]-1)
    for j in range(min(past.shape[0],5)):
        figure.add_trace(go.Scatter(x=idx,y=torch.cat((past[j,:,-1],future[j,:,-1]),dim=-1).cpu().detach().numpy(),name=f'True {j}'))
        for name, prediction in zip(names, prediction_array):
            figure.add_trace(go.Scatter(x=idx,y=torch.cat((past[j,:,-1],prediction[j,:,-1]),dim=-1).cpu().detach().numpy(),name=f'Prediction {j}-{name}'))
    figure.update_layout(yaxis_range=[-5,5])
    
    wandb.log({'figure': figure})

def toy_data_figs(data_array, names):
    # We assume that the ground truth is in the zeroth position
    fig = go.Figure()
    stats_and_figs = {}
    for data, name in zip(data_array,names):
        if data.shape[-1] == 1:
            fig.add_trace(go.Histogram(x=data[:,0].cpu().detach(),
                                            histnorm='probability',name=name))                   
        else:
            fig.add_trace(go.Scatter(x=data[:,0].cpu().detach().numpy(), 
                                            y=data[:,1].cpu().detach().numpy(),
                                            mode='markers',name=name))
            # fig.update_layout(yaxis_range=[-16,16], xaxis_range=[-16,16])
        stats_and_figs[f'w2-{name}'] = get_w2(data_array[0], data)   
    stats_and_figs['samples'] = fig  
    return stats_and_figs

    
if __name__ == '__main__':
    training()