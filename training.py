import os
import torch
import click
import wandb
import plotly.graph_objects as go
from tqdm import tqdm
from itertools import chain
from torch.optim import Adam

from utils.sde_lib import get_sde
import utils.losses as losses
from utils.model_utils import get_model, get_preconditioned_model
from utils.datasets import get_dataset
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
    
def update_ema(model, model_ema, beta):
    for p_ema, p_net in zip(model_ema.parameters(), model.parameters()):
        p_ema.copy_(p_net.detach().lerp(p_ema, beta))

def default_num_iters(ctx, param, value):
    sde = ctx.params.get('sde')
    if value is not None: 
        return value
    return 2000 if sde == 'sb' else 30000
def default_log_rate(ctx, param, value):
    sde = ctx.params.get('sde')
    if value is not None: 
        return value
    return 500 if sde == 'sb' else 5000

@click.command()
@click.option('--dataset',type=click.Choice(['gmm','spiral','checkerboard']))
@click.option('--model_forward',type=click.Choice(['mlp','toy','linear']), default='mlp')
@click.option('--model_backward',type=click.Choice(['mlp','toy','linear']), default='mlp')
@click.option('--precondition', is_flag=True, default=False)
@click.option('--sde',type=click.Choice(['vp','sb','edm']), default='vp')
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adam')
@click.option('--lr', type=float, default=3e-3)
@click.option('--ema_beta', type=float, default=.99)
@click.option('--batch_size', type=int, default=512)
@click.option('--log_rate',type=int,callback=default_log_rate)
@click.option('--num_iters',type=int,callback=default_num_iters)
@click.option('--dir',type=str)
def training(**opts):
    opts = dotdict(opts)
    print(opts)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(opts)
    dim = dataset.dim
    is_sb = (opts.sde == 'sb')
    
    sde = get_sde(opts.sde)
    sampling_sde = get_sde(opts.sde)
    # Set up backwards model
    model_backward, ema_backward = get_model(opts.model_backward,device)
    sde.backward_score = model_backward
    sampling_sde.backward_score = model_backward
    if is_sb:
        # We need a forward model
        model_forward , ema_forward  = get_model(opts.model_forward,device)
        sde.forward_score = model_forward
        sampling_sde.forward_score = model_forward
        
    if opts.precondition:
        sde.backward_score = get_preconditioned_model(model_backward, sde)
        sampling_sde.backward_score = get_preconditioned_model(model_backward, sde)
        
    opt = Adam(chain(model_forward.parameters(), model_backward.parameters()) 
               if is_sb else model_backward.parameters(), lr=opts.lr )
    num_iters = opts.num_iters
    batch_size = opts.batch_size
    log_sample_quality=opts.log_rate

    loss_fn = losses.standard_sb_loss if is_sb else losses.dsm_loss 
    init_wandb(opts)
    for i in tqdm(range(num_iters)):
        data = dataset.sample(batch_size).to(device=device)
        opt.zero_grad()
        
        loss = loss_fn(sde,data,model_backward)
        loss.backward()
        
        opt.step()
        
        # Update EMA
        update_ema(model_backward, ema_backward, opts.ema_beta)
        if is_sb:
            update_ema(model_forward,  ema_forward, opts.ema_beta)
        
        
        wandb.log({
            'loss': loss
        })
        # Evaluate sample accuracy
        if (i+1)%log_sample_quality == 0 or i+1 == num_iters:
            new_data, _ = sde.sample((1000,2), device)
            new_data_ema, _  = sampling_sde.sample((1000,2), device)
            fig = create_figs(dim, [data, new_data, new_data_ema], ['true','normal', 'ema'])
            wandb.log({'w2': get_w2(new_data,data), 'w2-ema': get_w2(new_data_ema, data),'samples': fig })
            
            path = os.path.join(opts.dir, f'itr_{i+1}/')
            os.makedirs(path,exist_ok=True) # Still wondering it this is the best idea
            
            torch.save(model_backward,os.path.join(path, f'backward_{i+1}.pt'))
            torch.save(ema_backward,os.path.join(path, f'backward_ema_{i+1}.pt'))
            if is_sb:
                torch.save(model_forward,os.path.join(path, f'forward_{i+1}.pt'))
                torch.save(ema_forward,os.path.join(path, f'forward_ema_{i+1}.pt'))
                

def create_figs(dim, data_array, names):
    fig = go.Figure()
    for data, name in zip(data_array,names):
        if dim == 1:
            fig.add_trace(go.Histogram(x=data[:,0].cpu().detach(),
                                            histnorm='probability',name=name))                   
        else:
            fig.add_trace(go.Scatter(x=data[:,0].cpu().detach().numpy(), 
                                            y=data[:,1].cpu().detach().numpy(),
                                            mode='markers',name=name))
            # fig.update_layout(yaxis_range=[-16,16], xaxis_range=[-16,16])     
    return fig

if __name__ == '__main__':
    training()