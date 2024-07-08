import torch
import click
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from torch.optim.adam import Adam
from tqdm import tqdm

from utils.misc import dotdict
from datasets.time_series_datasets import get_transformed_dataset
from utils.models import SimpleNN
from utils.model_t import EpsilonTheta
from utils.losses import dsm_loss
from utils.sde_lib import VP

@torch.no_grad()
def create_diffusion_prediction(past, sde : VP):
    prediction = torch.zeros(past.shape[0],30,past.shape[-1], device=past.device)
    cur_past = past.detach().clone()
    for i in range(30):
        prediction[:,i], _ = sde.sample(prediction[:,i].shape,past.device,cond=cur_past)
        cur_past = torch.cat((cur_past[:,1:],prediction[:,i].unsqueeze(1)),dim=1)
    return prediction

def init_wandb():
    wandb.init(
        # set the wandb project where this run will be logged
        project='variational-sb-time-series',
        name= f'exchange-rate',
        tags= ['training'],
        # # track hyperparameters and run metadata
        # config=config
    )

@click.command()
@click.option('--dataset',type=click.Choice(['exchange_rate']), default='exchange_rate')
@click.option('--cond_length',type=int, default=90)
@click.option('--pred_length',type=int,default=30)
@click.option('--batch_size',type=int, default=32)
@click.option('--lr', type=float, default=3e-4)
@click.option('--num_epochs',type=int, default=200)
def train(**opts):
    print(opts)
    opts = dotdict(opts)
    batch_size = opts.batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_ds, test_ds = get_transformed_dataset('exchange_rate',1, batch_size,50)

    input_dim = 8
    hidden_dim = 40
    t_embedding_dim = 40


    # model = SimpleNN(input_dim=input_dim, hidden_dim=hidden_dim, t_embedding_dim=t_embedding_dim).to(device)
    model = EpsilonTheta(input_dim,opts.cond_length)
    sde = VP()
    sde.backward_score = model

    opt = Adam(model.parameters(),lr=opts.lr)
    init_wandb()
    for i in tqdm(range(opts.num_epochs)):
        for batch in train_ds:
            past = batch['past_target'].to(device)
            future = batch['future_target'].to(device).view(batch_size,-1)
            
            opt.zero_grad()
            
            loss = dsm_loss(sde,future, past)
            
            opt.step()
            
            wandb.log({'loss': loss})
        
        if i%10 == 0:
            past = batch['past_target'][:2]
            prediction = create_diffusion_prediction(past.to(device),sde)
            figure = go.Figure()
            idx = torch.arange(past.shape[1])
            pred_idx = torch.arange(past.shape[1], past.shape[1] + 30)
            figure.add_vline(past.shape[1])
            for i in range(2):
                figure.add_trace(go.Scatter(x=idx,y=past[i,:,-1].cpu().detach().numpy()))
                figure.add_trace(go.Scatter(x=pred_idx, y=prediction[i,:,-1].cpu().detach().numpy()))
            wandb.log({'figure': figure})
        
    

if __name__ == '__main__':
    train()