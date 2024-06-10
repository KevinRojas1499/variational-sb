import click
import torch
import wandb
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim import Adam
from tqdm import tqdm

from utils.model import ScoreNet
from utils.networks_edm2 import Precond
from utils.sde_lib import SDE, get_sde
from utils.losses import get_loss
from utils.samplers import em_sampler, get_cld_euler
from utils.misc import dotdict


def init_wandb(opt):
    wandb.init(
    # set the wandb project where this run will be logged
    name=f'fp-mine-{opt.sde}',
    project='kinetic-fp',
    # name= get_run_name(config),
    tags= [f'{opt.dataset}','fp-quality'],
    # # track hyperparameters and run metadata
    # config=config
)



@click.command()
@click.option('--dataset', type=click.Choice(['mnist']), default='mnist')
@click.option('--sde', type=click.Choice(['vp','cld']), default='cld')
@click.option('--network', type=click.Choice(['unet','edm2']), default='unet')
@click.option('--optimizer', type=click.Choice(['adam','sgd']), default='adam')
@click.option('--lr', type=float, default=2e-4)
@click.option('--num_epochs', type=int, default=50)

@click.option('--network',type=str, default='cld')
@click.option('--batch_size',type=int, default=256)
def train(**opts):
    opts = dotdict(opts)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sde : SDE = get_sde(opts.sde)
    loss_fn = get_loss(opts.sde)
    
    dataset = MNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Resize((32,32))]), download=True)
    
    img_resolution = dataset[0][0].shape[-1]  
    num_channels = dataset[0][0].shape[0]  
    num_channels = 2 * num_channels if opts.sde == 'cld' else num_channels
    data_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    score_model = torch.nn.DataParallel(ScoreNet(in_channels=num_channels, sde=sde))
    # def real_score(x,t):
    #     return sde.multiply_inv_std( score_model(x,t), t.reshape(-1,1,1,1)) 
    # score_model = torch.nn.DataParallel(Precond(img_resolution,num_channels))
    
    optimizer = Adam(score_model.parameters(), lr=opts.lr)
    
    tqdm_epoch = tqdm(range(opts.num_epochs))
    init_wandb(opts)
    for i in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader: # Load data
            x = x.to(device)    
            loss = loss_fn(sde, x, score_model)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            # Print the averaged training loss so far.
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            wandb.log({'loss' : loss.item()})
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), f'checkpoints/{opts.dataset}_{opts.sde}/ckpt{i}.pth')
        samples = get_cld_euler(sde, score_model, (1,num_channels, img_resolution,img_resolution))
        wandb.log({'sample images' : wandb.Image(samples.cpu().detach().numpy())})
        
    
    
    
if __name__ == '__main__':
    train()