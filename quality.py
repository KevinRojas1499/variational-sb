import torch
import click
import wandb
import plotly.graph_objects as go
from tqdm import tqdm

import utils.sde_lib
import utils.models
import utils.losses as losses
import utils.samplers
from utils.metrics import get_w2
from datasets.toy_datasets import get_gmm
from utils.misc import dotdict


wandb.init(
    # set the wandb project where this run will be logged
    project='kinetic-fp',
    # name= get_run_name(config),
    tags= ['fp-quality'],
    # # track hyperparameters and run metadata
    # config=config
)


@click.command()
@click.option('--dim', type=int)
@click.option('--eval_fp', is_flag=True)
def fp_quality_evaluation(**opts):
    opts = dotdict(opts)
    dim = opts.dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sde = utils.sde_lib.VP()
    dataset = get_gmm(dim, sde, device)
    alphas = [0.0,0.01,0.5,1.0]
    model_names = [f'fp-{alpha}' for alpha in alphas] 
    models = [torch.load(f'checkpoints/gmm/{dim}d_ou_gmm_{alpha}.pt').to(device=device) for alpha in alphas]
    # Append the ground truth
    models.append(dataset.score)
    model_names.append('truth')
    alphas.append(-1.0)

    num_samples = 1000
    data = dataset.sample(num_samples)
    data_np = data.detach().cpu().numpy()
    if opts.eval_fp:
        samples_fig = go.Figure()
        for i, model in enumerate(models):
            for t in torch.linspace(0.1,sde.T,50, device=device):
                t_shape = t.unsqueeze(-1).expand(num_samples,1)
                fp_loss = losses.fp_loss(sde,data,model,t_shape)
                wandb.log({
                    't': t,
                    f'fp_loss_t_{model_names[i]}': fp_loss
                })
            new_data = utils.samplers.get_euler_maruyama(1000,sde,model, dim, device)
            wandb.log({
                'alpha': alphas[i],
                f'w2-{model_names[i]}': get_w2(new_data,data)
            })
            new_data = new_data.detach().cpu().numpy()
            samples_fig.add_trace(go.Scatter(x=new_data[:,0],y=new_data[:,1], mode='markers',name=model_names[i]))

        samples_fig.add_trace(go.Scatter(x=data_np[:,0],y=data_np[:,1], mode='markers',name='Truth'))
        wandb.log({'samples' : samples_fig})


    # if dim == 1:
    #     model = models[0]
    #     opt = torch.optim.Adam(model.parameters(),lr=3e-4)
    #     alpha = 1
    #     for i in tqdm(range(1000)):
    #         if i%25 == 0:
    #             fig = plot_score_over_time(device, sde, model_names, models)
    #             fig.write_image(f'trajectory/score_{i}.jpeg')
    #         data = dataset.sample(1000)
    #         opt.zero_grad()
    #         loss = losses.dsm_loss(sde,data,model) + alpha * losses.fp_loss(sde,data,model)
    #         loss.backward()
    #         opt.step()
        

def plot_score_over_time(device, sde, model_names, models):
    num_x = 100
    x_pts = torch.linspace(-10,10,steps=num_x, device=device).unsqueeze(-1)
    for t in torch.linspace(0.1,sde.T,1, device=device):
        fig = go.Figure()
        t_shape = t.unsqueeze(-1).expand(num_x,1)
        for i, model in enumerate(models):
            fig.add_trace(go.Scatter(x=x_pts[:,0].detach().cpu(), 
                                         y=model(x_pts, t_shape)[:,0].detach().cpu(),name=model_names[i]))
        # wandb.log({'score': fig})
    return fig
            
if __name__ == '__main__':
    fp_quality_evaluation()