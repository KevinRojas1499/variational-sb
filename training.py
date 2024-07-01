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

def is_sb_sde(name):
    return (name in ['sb','linear-sb','momentum-sb','linear-momentum-sb'])
    
def update_ema(model, model_ema, beta):
    for p_ema, p_net in zip(model_ema.parameters(), model.parameters()):
        p_ema.copy_(p_net.detach().lerp(p_ema, beta))

def default_num_iters(ctx, param, value):
    sde = ctx.params.get('sde')
    if value is not None: 
        return value
    return 2000 if is_sb_sde(sde) else 10000
def default_log_rate(ctx, param, value):
    sde = ctx.params.get('sde')
    if value is not None: 
        return value
    return 500 if is_sb_sde(sde) else 2000

@click.command()
@click.option('--dataset',type=click.Choice(['gmm','spiral','checkerboard']))
@click.option('--model_forward',type=click.Choice(['mlp','toy','linear']), default='mlp')
@click.option('--model_backward',type=click.Choice(['mlp','toy','linear']), default='mlp')
@click.option('--precondition', is_flag=True, default=False)
@click.option('--sde',type=click.Choice(['vp','cld','sb','edm', 'linear-sb','momentum-sb','linear-momentum-sb']), default='vp')
@click.option('--sb_loss_type', type=click.Choice(['joint','alternate']),default='alternate')
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adam')
@click.option('--lr', type=float, default=3e-4)
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
    is_sb = is_sb_sde(opts.sde)
    is_alternate_training = (opts.sb_loss_type == 'alternate')
    sde = get_sde(opts.sde)
    sampling_sde = get_sde(opts.sde)
    # Set up backwards model
    model_backward, ema_backward = get_model(opts.model_backward,sde, device)
    sde.backward_score = model_backward
    sampling_sde.backward_score = ema_backward
    if is_sb:
        # We need a forward model
        model_forward , ema_forward  = get_model(opts.model_forward,sde,device)
        sde.forward_score = model_forward
        sampling_sde.forward_score = ema_forward
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.zeros_(m.weight)

    # model_forward.apply(init_weights)
    # model_backward.apply(init_weights)  
    if opts.precondition:
        sde.backward_score = get_preconditioned_model(model_backward, sde)
        sampling_sde.backward_score = get_preconditioned_model(model_backward, sde)
        
    opt = Adam(chain(model_forward.parameters(), model_backward.parameters()) 
               if is_sb else model_backward.parameters(), lr=opts.lr )
    num_iters = opts.num_iters
    batch_size = opts.batch_size
    log_sample_quality=opts.log_rate

    loss_fn = losses.get_loss(opts.sde, is_alternate_training) 
    init_wandb(opts)
    for i in tqdm(range(num_iters)):
        data = dataset.sample(batch_size).to(device=device)
        opt.zero_grad()
        optimizing_forward = (i//250)%2 == 1
        
        if optimizing_forward:
            model_backward.requires_grad_(False)
            model_forward.requires_grad_(True)
        else:
            model_forward.requires_grad_(False)
            model_backward.requires_grad_(True)
        if is_alternate_training:
            loss = loss_fn(sde,data,optimize_forward=optimizing_forward, sampling_sde=sampling_sde)
        else:
            loss = loss_fn(sde,data)            
        loss.backward()
        
        # print('forward', torch.norm(torch.cat([p.grad.view(-1) for p in chain(model_forward.parameters())])))
        # print('backward',torch.norm(torch.cat([p.grad.view(-1) for p in chain(model_backward.parameters())])))
        
        torch.nn.utils.clip_grad_norm_(model_forward.parameters(),1)
        torch.nn.utils.clip_grad_norm_(model_backward.parameters(), 1)
        
        # print('forward', torch.norm(torch.cat([p.grad.view(-1) for p in chain(model_forward.parameters())])))
        # print('backward',torch.norm(torch.cat([p.grad.view(-1) for p in chain(model_backward.parameters())])))
        
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
            sampling_shape = (1000,4) if sde.is_augmented else (1000,2) 
            new_data, _ = sde.sample(sampling_shape, device)
            new_data_ema, _  = sampling_sde.sample(sampling_shape, device)
            fig = create_figs(dim, [data, new_data, new_data_ema], ['true','normal', 'ema'])
            wandb.log({'samples': fig })
            
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