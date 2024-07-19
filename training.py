import os
import torch
import click
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from itertools import chain
from torch.optim import Adam
from math import ceil

from utils.training_routines import get_routine
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
@click.option('--dataset',type=click.Choice(['mnist','spiral','checkerboard','exchange_rate']))
@click.option('--model_forward',type=click.Choice(['mlp','linear']), default='mlp')
@click.option('--model_backward',type=click.Choice(['mlp','unet', 'linear','time-series']), default='mlp')
@click.option('--precondition', is_flag=True, default=False)
@click.option('--sde',type=click.Choice(['vp','cld','sb','edm', 'linear-sb','momentum-sb','linear-momentum-sb']), default='vp')
@click.option('--loss_routine', type=click.Choice(['joint','alternate','variational','none']),default='none')
@click.option('--dsm_warm_up', type=int, default=2000, help='Perform first iterations using just DSM')
@click.option('--dsm_cool_down', type=int, default=5000, help='Perform last iterations using just DSM')
@click.option('--forward_opt_steps', type=int, default=5, help='Number of forward opt steps in alternate training scheme')
@click.option('--backward_opt_steps', type=int, default=495, help='Number of backward opt steps in alternate training scheme')
# Training Options
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adam')
@click.option('--lr', type=float, default=3e-4)
@click.option('--ema_beta', type=float, default=.99)
@click.option('--clip_grads', is_flag=True, default=False)
@click.option('--batch_size', type=int, default=128)
@click.option('--log_rate',type=int,callback=default_log_rate)
@click.option('--num_iters',type=int,callback=default_num_iters)
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
    sampling_sde = get_sde(opts.sde)
    # Set up backwards model
    network_opts = dotdict({
        'out_shape' : dataset.out_shape,
        'cond_length' : 90
    })
    model_backward, ema_backward = get_model(opts.model_backward,sde, device,network_opts=network_opts)
    sde.backward_score, sampling_sde.backward_score = model_backward, ema_backward
    print(f"Backward Model parameters: {sum(p.numel() for p in model_backward.parameters() if p.requires_grad)//1e6} M")
    if is_sb:
        # We need a forward model
        model_forward , ema_forward  = get_model(opts.model_forward,sde,device,network_opts=network_opts)
        sde.forward_score, sampling_sde.forward_score = model_forward, ema_forward
        print(f"Forward Model parameters: {sum(p.numel() for p in model_backward.parameters() if p.requires_grad)//1e6} M")
    
    start_iter = 0
    if opts.load_from_ckpt is not None:
        start_iter = int(opts.load_from_ckpt.split('_')[-1])
        print(f'Loading checkpoint at {opts.load_from_ckpt}, now starting at {start_iter}')
        model_backward.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'backward.pt')))
        ema_backward.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'backward_ema.pt')))
        if is_sb:
            model_forward.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'forward.pt')))
            ema_forward.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'forward_ema.pt')))
            
    if opts.precondition:
        sde.backward_score = get_preconditioned_model(model_backward, sde)
        sampling_sde.backward_score = get_preconditioned_model(model_backward, sde)
    
    opt = Adam(chain(model_forward.parameters(), model_backward.parameters()) 
               if is_sb else model_backward.parameters(), lr=opts.lr )
    
    num_iters = opts.num_iters
    log_sample_quality=opts.log_rate
    routine = get_routine(sde,sampling_sde,opts)

    init_wandb(opts)
    
    pbar = tqdm(range(start_iter, start_iter+opts.num_iters))
    for i in pbar:
        if dataset_type == 'toy':
            data, cond = next(dataset), None
        else:
            data, cond = next(dataset)
            cond = cond.to(device)
            
        data = data.to(device)
        opt.zero_grad()

        loss = routine.training_iteration(i,data, cond)           
        loss.backward()
        
        if opts.clip_grads:
            torch.nn.utils.clip_grad_norm_(model_forward.parameters(),1)
            torch.nn.utils.clip_grad_norm_(model_backward.parameters(), 1)
        
        opt.step()
        
        # Update EMA
        update_ema(model_backward, ema_backward, opts.ema_beta)
        if is_sb:
            update_ema(model_forward,  ema_forward, opts.ema_beta)
        if opts.loss_routine == 'variational':
            copy_ema_to_model(model_forward, ema_forward)
        
        wandb.log({
            'loss': loss
        })
        # Evaluate sample accuracy
        if (i+1)%log_sample_quality == 0 or i+1 == num_iters:
            # Save Checkpoints
            path = os.path.join(opts.dir, f'itr_{i+1}/')
            os.makedirs(path,exist_ok=True) # Still wondering it this is the best idea
            torch.save(model_backward.state_dict(),os.path.join(path, 'backward.pt'))
            torch.save(ema_backward.state_dict(),os.path.join(path, 'backward_ema.pt'))
            
            if is_sb:
                torch.save(model_forward.state_dict(),os.path.join(path, 'forward.pt'))
                torch.save(ema_forward.state_dict(),os.path.join(path, 'forward_ema.pt'))
                
            if dataset_type == 'time-series':
                future, past = dataset.__next__(train=False)
                future = future.to(device)
                past = past.to(device)
                pred = create_diffusion_time_series_prediction(24,network_opts.out_shape[0],past,sde)
                pred_ema = create_diffusion_time_series_prediction(24,network_opts.out_shape[0],past,sampling_sde)
                plot_time_series(past,future,[pred,pred_ema],['pred','ema'])
            else:
                sampling_shape = (1000 if dataset_type else 32, *network_opts.out_shape)
                new_data, _ = sde.sample(sampling_shape, device)
                new_data_ema, _  = sampling_sde.sample(sampling_shape, device)
                if dataset_type == 'toy':
                    relevant_log_info = toy_data_figs([data, new_data, new_data_ema], ['true','normal', 'ema'])
                    wandb.log(relevant_log_info)
                elif dataset_type == 'image':
                    plot_32_mnist(new_data,os.path.join(opts.dir,f'itr_{i+1}.png'))

            
    wandb.finish()

@torch.no_grad()
def create_diffusion_time_series_prediction(tot_pred_length, pred_size, past, sde):
    k = ceil(tot_pred_length//pred_size)
    prediction = torch.zeros(past.shape[0],pred_size * k,past.shape[-1], device=past.device)
    cur_past = past.detach().clone()
    for i in range(k):
        prediction[:,pred_size * i:pred_size * (i+1)] = sde.sample((past.shape[0],pred_size, past.shape[-1]),past.device,cond=past)[0] 
        cur_past = torch.cat((cur_past[:,pred_size:],prediction[:,pred_size * i:pred_size * (i+1)]),dim=1)
    return prediction

def plot_time_series(past,future, prediction_array, names):
    figure = go.Figure()
    idx = torch.arange(past.shape[1] + 30)
    figure.add_vline(past.shape[1]-1)
    for j in range(1):
        figure.add_trace(go.Scatter(x=idx,y=torch.cat((past[j,:,-1],future[j,:,-1]),dim=-1).cpu().detach().numpy(),name=f'True {j}'))
        for name, prediction in zip(names, prediction_array):
            figure.add_trace(go.Scatter(x=idx,y=torch.cat((past[j,:,-1],prediction[j,:,-1]),dim=-1).cpu().detach().numpy(),name=f'Prediction {j}-{name}'))
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