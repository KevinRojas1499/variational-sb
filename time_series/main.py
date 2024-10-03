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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

@click.command()
@click.option('--data', type=click.Choice(list(set(dataset_recipes.keys()))))
@click.option('--seed', type=int, default=1)
@click.option('--cpu', is_flag=True)
@click.option('--device', type=int, default=0)
@click.option('--max_data_dim', type=int, default=2000)

# Training params
@click.option('--batch_size', type=int, default=32)
@click.option('--hidden_dim', type=int, default=64)
@click.option('--epochs', type=int, default=50)
@click.option('--sde',type=click.Choice(['vp','cld', 'vsdm','linear-momentum-sb']), default='vp')
@click.option('--damp_coef',type=float, default=1.)
@click.option('--beta_max', type=float, default=10)
@click.option('--dsm_warm_up', type=int, default=500, help='Perform first iterations using just DSM')
@click.option('--dsm_cool_down', type=int, default=1000, help='Stop optimizing the forward model for these last iterations')
@click.option('--forward_opt_steps', type=int, default=50, help='Number of forward opt steps in alternate training scheme')
@click.option('--backward_opt_steps', type=int, default=250, help='Number of backward opt steps in alternate training scheme')
@click.option('--dir', type=str)
def main(**opts):
    opts = dotdict(opts)
    print(opts)
    set_seed(opts.seed)

    dataset = get_dataset(opts.data, regenerate=False)

    opts.data_dim = min(int(dataset.metadata.feat_static_cat[0].cardinality), opts.max_data_dim)

    train_grouper = MultivariateGrouper(
        max_target_dim=opts.data_dim,
    )
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)),
        max_target_dim=opts.data_dim,
    )

    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)

    estimator = DiffusionEstimator(
        freq=dataset.metadata.freq,
        input_size=opts.data_dim,
        prediction_length=dataset.metadata.prediction_length,
        context_length=dataset.metadata.prediction_length * 3,
        batch_size=opts.batch_size,
        sde=opts.sde,
        damp_coef=opts.damp_coef,
        beta_max=opts.beta_max,
        dsm_warm_up=opts.dsm_warm_up,
        dsm_cool_down=opts.dsm_cool_down,
        forward_opt_steps=opts.forward_opt_steps,
        backward_opt_steps=opts.backward_opt_steps,
        num_layers=2,
        hidden_size=opts.hidden_dim,
        lags_seq=None,
        scaling='std',
        trainer_kwargs=dict(
            max_epochs=opts.epochs,
            accelerator='cpu' if opts.cpu else 'gpu',
            devices=[opts.device],
            callbacks=[ModelCheckpoint(monitor=None)],
            logger=CSVLogger(opts.dir, name='logs'),
        ),
    )

    predictor = estimator.train(dataset_train, cache_data=True, shuffle_buffer_length=1024)

    torch.save(predictor.prediction_net.model.backward_net.state_dict(), os.path.join(opts.dir,'backward_model.pt'))
    if opts.sde not in ['vp','cld']:
        torch.save(predictor.prediction_net.model.forward_net.state_dict(), os.path.join(opts.dir,'forward_model.pt'))

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test,
        predictor=predictor,
        num_samples=estimator.num_parallel_samples,
    )

    forecasts = list(forecast_it)
    targets = list(ts_it)

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

    with open(os.path.join(opts.dir,'metrics.json'), 'w') as f:
        json.dump(agg_metric, f)

    with open(os.path.join(opts.dir,'summary_metrics.json'), 'w') as f:
        json.dump(summary_metrics, f)
        
    with open(os.path.join(opts.dir, 'forecasts.pickle'), 'wb') as f:
        pickle.dump([forecasts, targets], f)


if __name__ == '__main__':
    main()