import torch
import click
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from torch.optim.adam import Adam
from tqdm import tqdm

from utils.misc import dotdict
from datasets.time_series_datasets import get_transformed_dataset
from utils.models import SimpleNN, TimeSeriesNetwork
from utils.model_utils import PrecondVP
from utils.model_t import EpsilonTheta
from utils.losses import dsm_loss
from utils.sde_lib import VP

@torch.no_grad()
def create_diffusion_prediction(past, sde : VP):
    prediction = torch.zeros(past.shape[0],30,past.shape[-1], device=past.device)
    cur_past = past.detach().clone()
    for i in range(30):
        prediction[:,i:i+1] = sde.sample((past.shape[0],1, past.shape[-1]),past.device,cond=past)[0] 
        cur_past = torch.cat((cur_past[:,1:],prediction[:,i:i+1]),dim=1)
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
    metadata = dotdict(metadata)
    print(metadata)
    data_dim = metadata.dim
    cond_length = metadata.cond_length
    input_dim = data_dim * metadata.pred_length
    hidden_dim = 40
    t_embedding_dim = 40

    # model = SimpleNN(input_dim=input_dim, cond_input_dim=data_dim,hidden_dim=hidden_dim, t_embedding_dim=t_embedding_dim).to(device)
    model = TimeSeriesNetwork(input_dim=input_dim, cond_input_dim=data_dim, cond_length=cond_length,hidden_dim=hidden_dim, t_embedding_dim=t_embedding_dim).to(device)
    if opts.load_from_ckpt is not None:
        model.load_state_dict(torch.load(opts.load_from_ckpt))
    # model = EpsilonTheta(input_dim,metadata.cond_length)
    sde = VP()
    sde.backward_score = PrecondVP(model,sde)

    opt = Adam(model.parameters(),lr=opts.lr)
    init_wandb(opts)
    for i in tqdm(range(opts.num_epochs)):
        k = 0
        for batch in train_ds:
            past = batch['past_target'].to(device)# * 10 + 50
            future = batch['future_target'].to(device) #* 10 + 50
            opt.zero_grad()
            
            loss = dsm_loss(sde,future, past)
            loss.backward()
            
            opt.step()
            
            wandb.log({'loss': loss})
            k+=1
        # print(f'Finished an epoch and used a total of {k} iterations')
        
        if (i+1)%50 == 0:
            batch = next(iter(test_ds))
            past = batch['past_target'].to(device)# * 10 + 50
            future = batch['future_target'].to(device) #* 10 + 50
            torch.save(model.state_dict(),f'checkpoints/time_series/{(i+1)}.pt')
            prediction = create_diffusion_prediction(past.to(device),sde)
            figure = go.Figure()
            idx = torch.arange(past.shape[1] + 30)
            figure.add_vline(past.shape[1])
            for j in range(4):
                figure.add_trace(go.Scatter(x=idx,y=torch.cat((past[j,:,-1],prediction[j,:,-1]),dim=-1).cpu().detach().numpy(),name=f'Prediction {j}'))
                figure.add_trace(go.Scatter(x=idx,y=torch.cat((past[j,:,-1],future[j,:,-1]),dim=-1).cpu().detach().numpy(),name=f'True {j}'))
                # figure.update_layout(
                #     yaxis=dict(range=[52, 60])  # Set y-axis limits
                # )
            wandb.log({'figure': figure})
        
    

if __name__ == '__main__':
    train()