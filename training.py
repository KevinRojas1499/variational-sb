import os
import torch
import click
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import DataLoader
from utils.training_routines import get_routine
from utils.sde_lib import get_sde
from utils.model_utils import get_model, get_preconditioned_model
from datasets.dataset_utils import get_dataset
from utils.metrics import get_w2
from utils.misc import dotdict
from utils.optim_utils import build_optimizer_ema_sched
from utils.autoencoder import Autoencoder

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
    n_rows, n_cols = 4,4
    fig, axs = plt.subplots(n_rows,n_cols)
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            im = x_t[idx].permute(1,2,0)
            axs[i][j].axis('off')
            axs[i][j].imshow(im.clamp(0,1).cpu().numpy())
            idx+=1
    plt.tight_layout()
    fig.savefig(file_name,bbox_inches='tight')
    # plt.close(fig) # TODO : Why is this not working?

def process(x):
    return 2 * x - 1

def unprocess(x):
    return (x+1)/2

def get_dataset_type(name):
    if name in ['spiral','checkerboard']:
        return 'toy'
    elif name in ['mnist','fashion', 'cifar']:
        return 'image'

def is_sb_sde(name):
    return (name in ['vsdm','linear-momentum-sb'])

@click.command()
@click.option('--dataset',type=click.Choice(['mnist','cifar','fashion','spiral','checkerboard']), default='cifar')
@click.option('--model_forward',type=click.Choice(['linear']), default='linear')
@click.option('--model_backward',type=click.Choice(['mlp','unet','DiT', 'linear']), default='unet')
@click.option('--encoder',type=str, default=None)
@click.option('--precondition', is_flag=True, default=True)
@click.option('--sde',type=click.Choice(['vp','cld','vsdm','linear-momentum-sb']), default='linear-momentum-sb')
@click.option('--damp_coef',type=float, default=1.)
@click.option('--dsm_warm_up', type=int, default=0, help='Perform first iterations using just DSM')
@click.option('--dsm_cool_down', type=int, default=0, help='Stop optimizing the forward model for these last iterations')
@click.option('--forward_opt_steps', type=int, default=100, help='Number of forward opt steps in alternate training scheme')
@click.option('--backward_opt_steps', type=int, default=9900, help='Number of backward opt steps in alternate training scheme')
# Training Options
@click.option('--seed', type=int, default=42)
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adamw')
@click.option('--lr', type=float, default=3e-4)
@click.option('--ema_beta', type=float, default=.9999)
@click.option('--clip_grads', is_flag=True, default=True)
@click.option('--batch_size', type=int, default=128)
@click.option('--log_rate',type=int,default=10000)
@click.option('--num_epochs',type=int,default=10000)
@click.option('--dir',type=str)
@click.option('--load_from_ckpt', type=str)
@click.option('--disable_wandb',is_flag=True,default=False)
def training(**opts):
    opts = dotdict(opts)
    batch_size = opts.batch_size
    
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    assert batch_size % world_size == 0, f'Batch size must be divisible by world size, got {batch_size} and {world_size}'
    batch_size//=world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = opts.seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    print(opts)
    print(device)
    enable_wandb = not opts.disable_wandb and rank == 0
    
    dataset, out_shape, label_dim = get_dataset(opts)
    dataset_type = get_dataset_type(opts.dataset)
    if dataset_type == 'toy':
        data_loader = dataset
    else:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=opts.seed)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, 
                                sampler= sampler, drop_last=True, pin_memory=True)
    epochs = opts.num_epochs
    num_iters = epochs * (len(dataset)//(world_size * batch_size) )
    
    encode = False
    if opts.encoder is not None:
        encode = True
        autoencoder = Autoencoder(opts.encoder).to(device) 
        out_shape = [4, 16, 16]

    is_sb = is_sb_sde(opts.sde)
    sde = get_sde(opts.sde)
    sampling_sde = get_sde(opts.sde)
    # Set up backwards model
    if sde.is_augmented:
        out_shape[0] *= 2 
    network_opts = dotdict({'out_shape' : out_shape, 'damp_coef' : opts.damp_coef})
    model_backward = DDP(get_model(opts.model_backward,sde, device,label_dim=label_dim, network_opts=network_opts))
    opt_b, ema_backward, sched_b = build_optimizer_ema_sched(model_backward,opts.optimizer,opts.lr, step_size=10000, ema_beta=opts.ema_beta)
    sde.backward_score, sampling_sde.backward_score = model_backward, ema_backward
    print(f"Backward Model parameters: {sum(p.numel() for p in model_backward.parameters() if p.requires_grad)//1e6} M")
    opt_f, ema_forward, sched_f = None, None, None
    if is_sb:
        # We need a forward model
        model_forward  = DDP(get_model(opts.model_forward,sde,device,network_opts=network_opts))
        opt_f, ema_forward, sched_f = build_optimizer_ema_sched(model_forward,opts.optimizer,opts.lr/100, step_size=100, ema_beta=opts.ema_beta)
        sde.forward_score, sampling_sde.forward_score = model_forward, ema_forward
        print(f"Forward Model parameters: {sum(p.numel() for p in model_forward.parameters() if p.requires_grad)//1e6} M")
    
    start_iter = 0
    if opts.load_from_ckpt is not None:
        start_iter = int(opts.load_from_ckpt.split('_')[-1])
        print(f'Loading checkpoint at {opts.load_from_ckpt}, now starting at {start_iter}')
        model_backward.module.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'backward.pt'), weights_only=True))
        ema_backward.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'backward_ema.pt'), weights_only=True))
        opt_b.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'opt_b.pt'), weights_only=True))
        sched_b.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'sched_b.pt'), weights_only=True))
        if is_sb:
            model_forward.module.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'forward.pt'), weights_only=True))
            ema_forward.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'forward_ema.pt'), weights_only=True))
            opt_f.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'opt_f.pt'), weights_only=True))
            sched_f.load_state_dict(torch.load(os.path.join(opts.load_from_ckpt,'sched_f.pt'), weights_only=True))
    if opts.precondition:
        sde.backward_score = get_preconditioned_model(model_backward, sde)
        sampling_sde.backward_score = get_preconditioned_model(model_backward, sde)
    
    log_sample_quality=opts.log_rate
    routine = get_routine(opts, num_iters, sde,sampling_sde,opt_b, sched_b, ema_backward, opt_f, sched_f, ema_forward)

    if enable_wandb:
        init_wandb(opts)

    print(f'Running for {num_iters} iterations')
    cur_itr = start_iter
    for epoch in range(epochs):
        pbar = tqdm(data_loader, leave=False) if rank == 0 else data_loader
        for _data in pbar:
            if dataset_type == 'toy':
                data, cond = _data, None
            else:
                data, cond = _data
                cond = cond.to(device)
                
            data = data.to(device)

            if encode:
                data = autoencoder.encode(data)
            
            data = process(data)
            loss = routine.training_iteration(cur_itr,data, cond)           
            dist.all_reduce(loss)
            loss = loss/world_size 
            if enable_wandb:
                wandb.log({
                    'loss': loss,
                    'itr' : cur_itr
                })
            if (cur_itr+1)%5 == 0 and rank == 0:
                # lr_b = opt_b.param_groups[0]['lr'] 
                # lr_f = opt_f.param_groups[0]['lr']
                # pbar.set_description(f'loss {loss : .3f} lr_b {lr_b : .5f} lr_f {lr_f : .5f}')
                pbar.set_description(f'Epoch {epoch}/{epochs} - Iter {cur_itr} loss {loss : .3f}')

            dist.barrier() 
            # Evaluate sample accuracy
            if (cur_itr+1)%log_sample_quality == 0 or cur_itr+1 == num_iters:
                # if is_sb:
                #     print('EMA\n' ,  [param for param in ema_forward.parameters()])
                # Save Checkpoints
                path = os.path.join(opts.dir, f'itr_{cur_itr+1}/')
                os.makedirs(path,exist_ok=True) 
                if rank == 0:
                    snapshot = {
                        'backward' : model_backward.module.state_dict(),
                        'backward_ema' : ema_backward.module.ema.state_dict(),
                        'opt_b' : routine.opt_b.state_dict(),
                        'sched_b' : routine.sched_b.state_dict()
                    }
                    if is_sb:
                        snapshot.update({
                            'forward': model_forward.module.state_dict(),
                            'forward_ema': ema_forward.module.ema.state_dict(),
                            'opt_f': routine.opt_f.state_dict(),
                            'sched_f': routine.sched_f.state_dict()
                        })
                    
                    torch.save(snapshot, os.path.join(path, 'snapshot.pt'))
                
                n_samples = 2000 if dataset_type == 'toy' else 32 
                sampling_shape = (n_samples, *out_shape)
                labels = cond[:n_samples]
                
                new_data, _ = sde.sample(sampling_shape, device,cond=labels)
                new_data_ema, _  = sampling_sde.sample(sampling_shape, device, cond=labels)
                new_data = unprocess(new_data)
                new_data_ema = unprocess(new_data_ema)
                if encode:
                    new_data = autoencoder.decode(new_data)
                    new_data_ema = autoencoder.decode(new_data_ema)
                dist.barrier()
                if dataset_type == 'toy':
                    relevant_log_info = toy_data_figs([data, new_data, new_data_ema], ['true','normal', 'ema'])
                    wandb.log(relevant_log_info)
                elif dataset_type == 'image':
                    path_samples = os.path.join(opts.dir,f'itr_{cur_itr+1}_{rank}.png')
                    path_samples_ema = os.path.join(opts.dir,f'itr_ema_{cur_itr+1}_{rank}.png')
                    plot_32_mnist(new_data, path_samples)
                    plot_32_mnist(new_data_ema, path_samples_ema)
                    if enable_wandb:
                        wandb.log({
                                'samples' : wandb.Image(path_samples), 
                                'samples-ema' : wandb.Image(path_samples_ema),
                                'itr-samples' : cur_itr + 1
                            })
                
                dist.barrier() 
            cur_itr += 1

    if enable_wandb:
        wandb.finish()
    dist.destroy_process_group()
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