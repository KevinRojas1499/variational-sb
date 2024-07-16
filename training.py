import os
import torch
import click
import wandb
import plotly.graph_objects as go
from tqdm import tqdm
from itertools import chain
from torch.optim import Adam

from utils.training_routines import AlternateTrainingRoutine, VariationalDiffusionTrainingRoutine
from utils.sde_lib import get_sde
import utils.losses as losses
from utils.model_utils import get_model, get_preconditioned_model
from datasets.toy_datasets import get_dataset
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
    return 500 if is_sb_sde(sde) else 2000

@click.command()
@click.option('--dataset',type=click.Choice(['gmm','spiral','checkerboard']))
@click.option('--model_forward',type=click.Choice(['mlp','linear']), default='mlp')
@click.option('--model_backward',type=click.Choice(['mlp','linear']), default='mlp')
@click.option('--precondition', is_flag=True, default=False)
@click.option('--sde',type=click.Choice(['vp','cld','sb','edm', 'linear-sb','momentum-sb','linear-momentum-sb']), default='vp')
@click.option('--loss_routine', type=click.Choice(['joint','alternate','variational']),default='alternate')
@click.option('--dsm_warm_up', type=int, default=2000, help='Perform first iterations using just DSM')
@click.option('--dsm_cool_down', type=int, default=5000, help='Perform last iterations using just DSM')
@click.option('--forward_opt_steps', type=int, default=100, help='Number of forward opt steps in alternate training scheme')
@click.option('--backward_opt_steps', type=int, default=100, help='Number of backward opt steps in alternate training scheme')
# Training Options
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adam')
@click.option('--lr', type=float, default=3e-4)
@click.option('--ema_beta', type=float, default=.99)
@click.option('--clip_grads', is_flag=True, default=False)
@click.option('--batch_size', type=int, default=512)
@click.option('--log_rate',type=int,callback=default_log_rate)
@click.option('--num_iters',type=int,callback=default_num_iters)
@click.option('--dir',type=str)
@click.option('--load_from_ckpt', type=str)
def training(**opts):
    opts = dotdict(opts)
    print(opts)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(opts)
    dim = dataset.dim
    is_sb = is_sb_sde(opts.sde)
    is_alternate_training = (is_sb and opts.loss_routine == 'alternate')
    sde = get_sde(opts.sde)
    sampling_sde = get_sde(opts.sde)
    # Set up backwards model
    network_opts = dotdict({
        'out_shape' : [4 if sde.is_augmented else 2] 
    })
    model_backward, ema_backward = get_model(opts.model_backward,sde, device,network_opts=network_opts)
    sde.backward_score = model_backward
    sampling_sde.backward_score = ema_backward
    if is_sb:
        # We need a forward model
        model_forward , ema_forward  = get_model(opts.model_forward,sde,device,network_opts=network_opts)
        sde.forward_score = model_forward
        sampling_sde.forward_score = ema_forward
        
    if opts.load_from_ckpt is not None:
        model_backward = torch.load(os.path.join(opts.load_from_ckpt,'backward.pt'))
        ema_backward = torch.load(os.path.join(opts.load_from_ckpt,'backward_ema.pt'))
        sde.backward_score = model_backward
        sampling_sde.backward_score = ema_backward
        if is_sb:
            model_forward = torch.load(os.path.join(opts.load_from_ckpt,'forward.pt'))
            ema_forward = torch.load(os.path.join(opts.load_from_ckpt,'forward_ema.pt'))
            sde.forward_score = model_forward
            sampling_sde.forward_score = ema_forward
            
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.zeros_(m.weight)

            torch.nn.init.zeros_(model_forward.out.bias)
            torch.nn.init.zeros_(model_forward.out.weight)
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
    if is_alternate_training:
        routine = AlternateTrainingRoutine(sde,sampling_sde,model_forward,model_backward,opts.refresh_rate,100,device)
    elif opts.loss_routine == 'variational':
        routine = VariationalDiffusionTrainingRoutine(sde,sampling_sde,model_forward,model_backward,
                                                      opts.dsm_warm_up,opts.num_iters-opts.dsm_warm_up - opts.dsm_cool_down, opts.dsm_cool_down,
                                                      opts.forward_opt_steps, opts.backward_opt_steps,100,device)
    loss_fn = losses.get_loss(opts.sde, is_alternate_training) 

    init_wandb(opts)
    
    torch.autograd.set_detect_anomaly(True)
    
    for i in tqdm(range(num_iters)):
        data = dataset.sample(batch_size).to(device=device)
        opt.zero_grad()

        if is_alternate_training:
            loss = routine.training_iteration(i,data)
        elif opts.loss_routine == 'variational':
            loss = routine.training_iteration(i,data)
        else:
            loss = loss_fn(sde,data)            
        loss.backward()
        
        # print('forward', torch.norm(torch.cat([p.grad.view(-1) for p in chain(model_forward.parameters())])))
        # print('backward',torch.norm(torch.cat([p.grad.view(-1) for p in chain(model_backward.parameters())])))
        
        if opts.clip_grads:
            torch.nn.utils.clip_grad_norm_(model_forward.parameters(),1)
            torch.nn.utils.clip_grad_norm_(model_backward.parameters(), 1)
            
        # print('forward', torch.norm(torch.cat([p.grad.view(-1) for p in chain(model_forward.parameters())])))
        # print('backward',torch.norm(torch.cat([p.grad.view(-1) for p in chain(model_backward.parameters())])))
        
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
            sampling_shape = (1000,4) if sde.is_augmented else (1000,2) 
            new_data, _ = sde.sample(sampling_shape, device,prob_flow=opts.forward_model != 'linear')
            new_data_ema, _  = sampling_sde.sample(sampling_shape, device)
            fig = create_figs(dim, [data, new_data, new_data_ema], ['true','normal', 'ema'])
            wandb.log({'samples': fig , 'w2' : get_w2(data,new_data[:,:2]), 
                       'w2-ema' : get_w2(data,new_data_ema[:,:2])})
            
            path = os.path.join(opts.dir, f'itr_{i+1}/')
            os.makedirs(path,exist_ok=True) # Still wondering it this is the best idea
            
            torch.save(model_backward,os.path.join(path, 'backward.pt'))
            torch.save(ema_backward,os.path.join(path, 'backward_ema.pt'))
            if is_sb:
                torch.save(model_forward,os.path.join(path, 'forward.pt'))
                torch.save(ema_forward,os.path.join(path, 'forward_ema.pt'))
                

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