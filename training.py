import os
import torch
import click
import wandb
import plotly.graph_objects as go
from tqdm import tqdm

import utils.sde_lib
import utils.models
import utils.losses as losses
from utils.model_utils import get_model
from utils.datasets import get_dataset
import utils.samplers
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

@click.command()
@click.option('--dataset',type=click.Choice(['gmm','spiral','checkerboard']))
@click.option('--model_type',type=click.Choice(['mlp','toy']), default='mlp')
@click.option('--sde',type=click.Choice(['vp','sb']), default='vp')
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adam')
@click.option('--lr', type=float, default=3e-3)
@click.option('--ema_beta', type=float, default=.99)
@click.option('--batch_size', type=int, default=512)
@click.option('--log_rate',type=int,default=5000)
@click.option('--num_iters',type=int,default=30000)
@click.option('--dir',type=str)
def training(**opts):
    opts = dotdict(opts)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(opts)
    dim = dataset.dim
    model_backward, ema_backward = get_model(opts.model_type,device)
    model_forward , ema_forward  = get_model(opts.model_type,device)
    
    
    sde = utils.sde_lib.VP() if opts.sde == 'vp' else utils.sde_lib.SchrodingerBridge(model_forward,model_backward)
    opt = torch.optim.Adam(model_backward.parameters(),lr=opts.lr)
    opt_sde = torch.optim.Adam(model_forward.parameters(),lr=opts.lr)
    num_iters = opts.num_iters
    batch_size = opts.batch_size
    log_sample_quality=opts.log_rate
    data = dataset.sample(1000)

    loss_fn = losses.dsm_loss if opts.sde == 'vp' else losses.standard_sb_loss
    init_wandb(opts)
    for i in tqdm(range(num_iters)):
        data = dataset.sample(batch_size).to(device=device)
        opt.zero_grad()
        opt_sde.zero_grad()
        loss = loss_fn(sde,data,model_backward)
        loss.backward()
        opt.step()
        opt_sde.step()
        
        # Update EMA
        # update_ema(model_forward,  ema_forward, opts.ema_beta)
        # update_ema(model_backward, ema_backward, opts.ema_beta)
        
        
        wandb.log({
            'loss': loss
        })
        # Evaluate sample accuracy
        if (i+1)%log_sample_quality == 0:
            # new_data = utils.samplers.get_euler_maruyama(1000,sde,model_backward,dim,device)
            sampling_sde = utils.sde_lib.SchrodingerBridge(ema_forward,ema_backward)
            new_data = sde.sample((1000,2), device)
            new_data_ema = sampling_sde.sample((1000,2), device)
            fig = create_figs(dim, [data, new_data, new_data_ema], ['true','normal', 'ema'])
            wandb.log({'w2': get_w2(new_data,data), 'w2-ema': get_w2(new_data_ema, data),'samples': fig })
            
            path = os.path.join(opts.dir, f'itr_{i+1}/')
            os.makedirs(path,exist_ok=True) # Still wondering it this is the best idea
            
            torch.save(model_backward,os.path.join(path, f'backward_{i+1}.pt'))
            torch.save(ema_backward,os.path.join(path, f'backward_ema_{i+1}.pt'))
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
                                
    return fig

if __name__ == '__main__':
    training()