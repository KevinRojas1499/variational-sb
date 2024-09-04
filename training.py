import os
import torch
import click
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm

from utils.training_routines import get_routine
from utils.sde_lib import get_sde
from utils.model_utils import get_model, get_preconditioned_model
from datasets.dataset_utils import get_dataset
from utils.metrics import get_w2
from utils.misc import dotdict
from utils.optim_utils import build_optimizer_ema_sched

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
    elif name in ['mnist','fashion']:
        return 'image'

def is_sb_sde(name):
    return (name in ['vsdm','linear-momentum-sb'])

@click.command()
@click.option('--dataset',type=click.Choice(['mnist','fashion','spiral','checkerboard']))
@click.option('--model_forward',type=click.Choice(['linear']), default='linear')
@click.option('--model_backward',type=click.Choice(['mlp','unet', 'linear']), default='mlp')
@click.option('--precondition', is_flag=True, default=True)
@click.option('--sde',type=click.Choice(['vp','cld','vsdm','linear-momentum-sb']), default='vp')
@click.option('--dsm_warm_up', type=int, default=0, help='Perform first iterations using just DSM')
@click.option('--dsm_cool_down', type=int, default=0, help='Stop optimizing the forward model for these last iterations')
@click.option('--forward_opt_steps', type=int, default=100, help='Number of forward opt steps in alternate training scheme')
@click.option('--backward_opt_steps', type=int, default=500, help='Number of backward opt steps in alternate training scheme')
# Training Options
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adamw')
@click.option('--lr', type=float, default=3e-4)
@click.option('--ema_beta', type=float, default=.99)
@click.option('--clip_grads', is_flag=True, default=True)
@click.option('--batch_size', type=int, default=128)
@click.option('--log_rate',type=int,default=5000)
@click.option('--num_iters',type=int,default=100000)
@click.option('--dir',type=str)
@click.option('--load_from_ckpt', type=str)
@click.option('--disable_wandb',is_flag=True,default=False)
def training(**opts):
    opts = dotdict(opts)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)
    print(device)
    enable_wandb = not opts.disable_wandb
    dataset = get_dataset(opts)
    dataset_type = get_dataset_type(opts.dataset)
    is_sb = is_sb_sde(opts.sde)
    sde = get_sde(opts.sde)
    sampling_sde = get_sde(opts.sde)
    # Set up backwards model
    network_opts = dotdict({
        # 'out_shape' : dataset.out_shape
        'out_shape' : [1,28,28]
        
    })
    model_backward = get_model(opts.model_backward,sde, device,network_opts=network_opts)
    opt_b, ema_backward, sched_b = build_optimizer_ema_sched(model_backward,opts.optimizer,opts.lr)
    sde.backward_score, sampling_sde.backward_score = model_backward, ema_backward
    print(f"Backward Model parameters: {sum(p.numel() for p in model_backward.parameters() if p.requires_grad)//1e6} M")
    opt_f, ema_forward, sched_f = None, None, None
    if is_sb:
        # We need a forward model
        model_forward  = get_model(opts.model_forward,sde,device,network_opts=network_opts)
        opt_f, ema_forward, sched_f = build_optimizer_ema_sched(model_forward,opts.optimizer,opts.lr)
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
    
    num_iters = opts.num_iters
    log_sample_quality=opts.log_rate
    routine = get_routine(opts, sde,sampling_sde,opt_b, sched_b, ema_backward, opt_f, sched_f, ema_forward)

    if enable_wandb:
        init_wandb(opts)
    
    pbar = tqdm(range(start_iter, start_iter+opts.num_iters))
    for i in pbar:
        if dataset_type == 'toy':
            data, cond = next(dataset), None
        else:
            data, cond = next(dataset)
            cond = cond.to(device)
            
        data = data.to(device)
        
        loss = routine.training_iteration(i,data, cond)           
        
        if enable_wandb:
            wandb.log({
                'loss': loss
            })
        
        if (i+1)%100 == 0:
            lr_b = opt_b.param_groups[0]['lr'] 
            # lr_f = opt_f.param_groups[0]['lr']
            # pbar.set_description(f'loss {loss : .3f} lr_b {lr_b : .5f} lr_f {lr_f : .5f}')
            pbar.set_description(f'loss {loss : .3f} lr_b {lr_b : .5f}')
            
        # Evaluate sample accuracy
        if (i+1)%log_sample_quality == 0 or i+1 == num_iters:
            if is_sb:
                print('EMA\n' ,  [param for param in ema_forward.parameters()])
            # Save Checkpoints
            path = os.path.join(opts.dir, f'itr_{i+1}/')
            os.makedirs(path,exist_ok=True) # Still wondering it this is the best idea
            torch.save(model_backward.state_dict(),os.path.join(path, 'backward.pt'))
            torch.save(ema_backward.state_dict(),os.path.join(path, 'backward_ema.pt'))
            
            if is_sb:
                torch.save(model_forward.state_dict(),os.path.join(path, 'forward.pt'))
                torch.save(ema_forward.state_dict(),os.path.join(path, 'forward_ema.pt'))
                
            n_samples = 2000 if dataset_type == 'toy' else opts.batch_size
            sampling_shape = (n_samples, 4 if sde.is_augmented else 2)
            # labels = torch.randint(0,10,(n_samples,),device=device) if cond is not None else None
            labels = cond
            
            new_data, _ = sde.sample(sampling_shape, device,cond=cond)
            new_data_ema, _  = sampling_sde.sample(sampling_shape, device, cond=cond)
            if dataset_type == 'toy':
                relevant_log_info = toy_data_figs([data, new_data, new_data_ema], ['true','normal', 'ema'])
                wandb.log(relevant_log_info)
            elif dataset_type == 'image':
                plot_32_mnist(new_data,os.path.join(opts.dir,f'itr_{i+1}.png'))
                plot_32_mnist(new_data,os.path.join(opts.dir,f'itr_ema_{i+1}.png'))
                

            
    wandb.finish()

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
            # fig.update_layout(yaxis_range=[-10,10], xaxis_range=[-30,50])
        stats_and_figs[f'w2-{name}'] = get_w2(data_array[0], data)   
    stats_and_figs['samples'] = fig  
    return stats_and_figs

    
if __name__ == '__main__':
    training()