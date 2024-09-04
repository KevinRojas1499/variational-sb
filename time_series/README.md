# Time Series Experiments

This repo is based off of ...

# Options

A sample run can be:

```
python3 time_series/main.py --data exchange_rate --sde linear-momentum-sb --beta_max 6 --device 0 --dir results/momentum_exchange_rate
```

The full options are:



| Option               | Type                              | Default | Explanation                          |
|----------------------|-----------------------------------|---------|--------------------------------------|
| `--data`             |                                   | N/A     | Dataset to be run                    |
| `--seed`             | `int`                             | 1       | Random seed                          |
| `--cpu`              | `is_flag=True`                    | False   | If you want to use CPU               |
| `--device`           | `int`                             | 0       | GPU ID                               |
| `--max_data_dim`     | `int`                             | 2000    | Limit the number of dims of the multivariate time series |
| `--batch_size`       | `int`                             | 32      | Batch size                           |
| `--hidden_dim`       | `int`                             | 64      | Hidden dim for architecture          |
| `--epochs`           | `int`                             | 50      | Number of epochs                     |
| `--sde`              | ['vp','cld','vsdm','linear-momentum-sb'] | 'vp' | SDE                              |
| `--beta_max`         | `float`                           | 10      | Beta_max arg for sde schedule        |
| `--dsm_warm_up`      | `int`                             | 500     | Perform first iterations using just DSM |
| `--dsm_cool_down`    | `int`                             | 1000    | Stop optimizing the forward model for these last iterations |
| `--forward_opt_steps`| `int`                             | 50      | Number of forward opt steps in alternate training scheme |
| `--backward_opt_steps`| `int`                            | 250     | Number of backward opt steps in alternate training scheme |
