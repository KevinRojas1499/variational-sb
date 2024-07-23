import click
import json
import pickle
import random
import torch
import numpy as np
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

from estimator import DiffusionEstimator

import sys
import os

# Add the parent directory to the sys.path to ensure it can find hello.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.misc import dotdict

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent / 'results'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

@click.command()
@click.option('--data', type=click.Choice(list(set(dataset_recipes.keys()))))
@click.option('--seed', type=int, default=0)
@click.option('--cpu', is_flag=True)
@click.option('--device', type=int, default=0)
@click.option('--max_data_dim', type=int, default=2000)

# Training params
@click.option('--batch_size', type=int)
@click.option('--hidden_dim', type=int)
@click.option('--epochs', type=int)
@click.option('--sde',type=click.Choice(['vp','cld', 'linear-sb','linear-momentum-sb']), default='vp')
@click.option('--dsm_warm_up', type=int, default=500, help='Perform first iterations using just DSM')
@click.option('--dsm_cool_down', type=int, default=500, help='Stop optimizing the forward model for these last iterations')
@click.option('--forward_opt_steps', type=int, default=5, help='Number of forward opt steps in alternate training scheme')
@click.option('--backward_opt_steps', type=int, default=495, help='Number of backward opt steps in alternate training scheme')
def main(**opt):
    opt = dotdict(opt)
    print(opt)
    set_seed(opt.seed)

    if opt.ddpm:
        prefix = 'ddpm'
    elif opt.forward_opt_steps == 0:
        prefix = 'dsm'
    else:
        prefix = 'sb'

    name = f'{prefix}__{opt.data}__{opt.seed}__{opt.forward_opt_steps}__{opt.backward_opt_steps}'
    dir = ROOT_DIR / name
    dir.mkdir(exist_ok=True, parents=True)

    if (dir / 'forecasts.pickle').exists():
        print(f'Experiment {name} already exists. Skipping...')
        return

    dataset = get_dataset(opt.data, regenerate=False)

    opt.data_dim = min(int(dataset.metadata.feat_static_cat[0].cardinality), opt.max_data_dim)

    train_grouper = MultivariateGrouper(
        max_target_dim=opt.data_dim,
    )
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)),
        max_target_dim=opt.data_dim,
    )

    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)
    # print('Num test dates', int(len(dataset.test) / len(dataset.train)))
    
    # for x in dataset_test:
    #     print(x.keys())
    #     print(x['target'].shape)
    # i = 0
    # for batch in dataset_train:
    #     # print(i)
    #     for key, value in batch.items():
    #         print(f'KEY ------- {key}')
    #         print(f'VALUE ----------- {value}')
        
    #     if i == 0:
    #         break

    estimator = DiffusionEstimator(
        freq=dataset.metadata.freq,
        input_size=opt.data_dim,
        prediction_length=dataset.metadata.prediction_length,
        context_length=dataset.metadata.prediction_length * 3,
        batch_size=opt.batch_size,
        sde=opt.sde,
        dsm_warm_up=opt.dsm_warm_up,
        dsm_cool_down=opt.dsm_cool_down,
        forward_opt_steps=opt.forward_opt_steps,
        backward_opt_steps=opt.backward_opt_steps,
        num_layers=2,
        hidden_size=opt.hidden_dim,
        lags_seq=None,
        scaling='std',
        trainer_kwargs=dict(
            max_epochs=opt.epochs,
            accelerator='cpu' if opt.cpu else 'gpu',
            devices=[opt.device],
            callbacks=[ModelCheckpoint(monitor=None)],
            logger=CSVLogger(dir, name='logs'),
        ),
    )

    predictor = estimator.train(dataset_train, cache_data=True, shuffle_buffer_length=1024)

    if opt.sde not in ['vp','cld']:
        A = predictor.prediction_net.model.forward_net.detach().cpu().numpy()
        np.save(dir / 'forward_matrix.npy', A)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test,
        predictor=predictor,
        num_samples=estimator.num_parallel_samples,
    )
    print('We created some forecasts')
    print(ts_it)
    forecasts = list(forecast_it)
    targets = list(ts_it)
    print(len(targets))
    import matplotlib.pyplot as plt
    


    for i in range(4):
        ts_entry = targets[i].to_numpy()
        forecast_entry = forecasts[i].samples
        
        print(ts_entry.shape)
        print('Shape')
        print(forecast_entry.shape)
        plt.clf()
        k = forecast_entry.shape[1]
        plt.plot(np.arange(k),forecast_entry[0,:,0])
        plt.plot(np.arange(k),ts_entry[-k:,0])
        plt.legend(['forecast','true'])
        plt.savefig(f'./results/{i}.png')
    
    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum}
    )
    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))


    summary_metrics = {
        'CRPS': agg_metric['mean_wQuantileLoss'],
        'ND': agg_metric['ND'],
        'NRMSE': agg_metric['NRMSE'],
        'MSE': agg_metric['MSE'],
        'CRPS-Sum': agg_metric['m_sum_mean_wQuantileLoss'],
        'ND-Sum': agg_metric['m_sum_ND'],
        'NRMSE-Sum': agg_metric['m_sum_NRMSE'],
        'MSE-Sum': agg_metric['m_sum_MSE'],
    }

    print(summary_metrics)

    with open(dir / 'metrics.json', 'w') as f:
        json.dump(agg_metric, f)

    with open(dir / 'forecasts.pickle', 'wb') as f:
        pickle.dump([forecasts, targets], f)


if __name__ == '__main__':
    main()