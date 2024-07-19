import torch
import click
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from torch.optim.adam import Adam
from tqdm import tqdm

from utils.misc import dotdict
from datasets.time_series_datasets import get_transformed_dataset, TimeSeriesDataset
from utils.models import SimpleNN, TimeSeriesNetwork
from utils.model_utils import PrecondVP
from utils.losses import dsm_loss
from utils.sde_lib import VP

@torch.no_grad()
def create_diffusion_prediction(pred_size, past, sde : VP):
    prediction = torch.zeros(past.shape[0],pred_size * 2,past.shape[-1], device=past.device)
    cur_past = past.detach().clone()
    for i in range(2):
        prediction[:,pred_size * i:pred_size * (i+1)] = sde.sample((past.shape[0],pred_size, past.shape[-1]),past.device,cond=past)[0] 
        cur_past = torch.cat((cur_past[:,pred_size:],prediction[:,pred_size * i:pred_size * (i+1)]),dim=1)
    return prediction

def init_wandb(opts):
    wandb.init(
        # set the wandb project where this run will be logged
        project='variational-sb-time-series',
        name= opts.dataset,
        tags= ['training'],
        # # track hyperparameters and run metadata
        # config=config
    )

@click.command()
@click.option('--dataset',type=click.Choice(['exchange_rate','electricity_nips']), default='electricity_nips')
@click.option('--batch_size',type=int, default=128)
@click.option('--lr', type=float, default=3e-4)
@click.option('--ema', type=float, default=.9999)
@click.option('--num_epochs',type=int, default=200)
@click.option('--load_from_ckpt', type=str, default=None)
def train(**opts):
    print(opts)
    opts = dotdict(opts)
    batch_size = opts.batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_ds, test_ds, metadata = get_transformed_dataset(opts.dataset, batch_size,100)
    train_ds = TimeSeriesDataset(opts.dataset, batch_size,100)
    metadata = dotdict(metadata)
    print(metadata)
    data_dim = metadata.dim
    pred_length = metadata.pred_length
    cond_length = metadata.cond_length
    hidden_dim = 40

    # model = SimpleNN(input_dim=input_dim, cond_input_dim=data_dim,hidden_dim=hidden_dim, t_embedding_dim=t_embedding_dim).to(device)
    model = TimeSeriesNetwork(input_dim=data_dim, pred_length= pred_length,cond_length=cond_length,hidden_dim=hidden_dim).to(device)
    if opts.load_from_ckpt is not None:
        model.load_state_dict(torch.load(opts.load_from_ckpt))
    sde = VP()
    sde.backward_score = PrecondVP(model,sde)

    print(f'Num of parameters {sum(m.numel() for m in model.parameters())}')
    opt = Adam(model.parameters(),lr=opts.lr)
    init_wandb(opts)

    for i in tqdm(range(opts.num_epochs * 100)):
        future, past = next(train_ds)
        past = past.to(device)
        future = future.to(device)
        opt.zero_grad()
        
        loss = dsm_loss(sde,future, past)
        loss.backward()
        
        opt.step()
        
        wandb.log({'loss': loss})
    
        if (i+1)%5000 == 0:
            batch = next(iter(test_ds))
            past = batch['past_target'].to(device)
            future = batch['future_target'].to(device)
            torch.save(model.state_dict(),f'checkpoints/time_series/{(i+1)}.pt')
            prediction = create_diffusion_prediction(pred_length, past.to(device),sde)
            figure = go.Figure()
            idx = torch.arange(past.shape[1] + 30)
            figure.add_vline(past.shape[1])
            for j in range(2):
                figure.add_trace(go.Scatter(x=idx,y=torch.cat((past[j,:,-1],prediction[j,:,-1]),dim=-1).cpu().detach().numpy(),name=f'Prediction {j}'))
                figure.add_trace(go.Scatter(x=idx,y=torch.cat((past[j,:,-1],future[j,:,-1]),dim=-1).cpu().detach().numpy(),name=f'True {j}'))
                # figure.update_layout(
                #     yaxis=dict(range=[52, 60])  # Set y-axis limits
                # )
            wandb.log({'figure': figure})
        
    

if __name__ == '__main__':
    train()