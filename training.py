import torch
import click
import wandb
import plotly.graph_objects as go
from tqdm import tqdm

import utils.sde_lib
import utils.models
import utils.losses as losses
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


@click.command()
@click.option('--dataset',type=click.Choice(['gmm','spiral','checkerboard']))
@click.option('--sde',type=click.Choice(['vp','sb']), default='vp')
@click.option('--lr', type=float, default=3e-4)
@click.option('--batch_size', type=int, default=512)
@click.option('--log_rate',type=int,default=5000)
def training(**opts):
    opts = dotdict(opts)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(opts)
    dim = dataset.dim
    sde = utils.sde_lib.VP() if opts.sde == 'vp' else utils.sde_lib.LinearSchrodingerBridge(dim, device)
    model = utils.models.MLP(dim=dim,augmented_sde=False).to(device=device)
    if opts.sde == 'sb':
        score_model = utils.models.SB_Preconditioning(model,sde)
    opt = torch.optim.Adam(model.parameters(),lr=opts.lr)

    num_iters = 30000
    batch_size = opts.batch_size
    log_sample_quality=opts.log_rate
    data = dataset.sample(1000)

    loss_fn = losses.dsm_loss if opts.sde == 'vp' else losses.sb_loss
    init_wandb(opts)
    for i in tqdm(range(num_iters)):
        data = dataset.sample(batch_size).to(device=device)
        opt.zero_grad()
        loss = loss_fn(sde,data,model)
        loss.backward()
        opt.step()
        wandb.log({
            'loss': loss
        })
        # Evaluate sample accuracy
        if i%log_sample_quality == 0:
            new_data = utils.samplers.get_euler_maruyama(1000,sde,score_model,dim,device)
            fig = go.Figure()
            if dim == 1:
                fig.add_trace(go.Histogram(x=new_data[:,0].cpu().detach(),
                                           histnorm='probability',name='Generated'))
                fig.add_trace(go.Histogram(x=data[:,0].cpu().detach(),
                                           histnorm='probability',name='True'))
                
            else:
                fig.add_trace(go.Scatter(x=data[:,0].cpu().detach().numpy(), 
                                        y=data[:,1].cpu().detach().numpy(),
                                        mode='markers',name='True'))
                fig.add_trace(go.Scatter(x=new_data[:,0].cpu().detach().numpy(), 
                                        y=new_data[:,1].cpu().detach().numpy(),
                                        mode='markers',name='generated'))
            wandb.log({'w2': get_w2(new_data,data),
                    'samples': fig })
    torch.save(model,f'checkpoints/{opts.dataset}/{dim}d_ou_{opts.dataset}.pt')

if __name__ == '__main__':
    training()