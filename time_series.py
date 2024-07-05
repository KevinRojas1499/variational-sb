import torch
from datasets.time_series_datasets import get_transformed_dataset

train_ds, test_ds = get_transformed_dataset('exchange_rate',32,50)

i = 0
for batch in train_ds:
    print(i)
    print(batch.keys())
    print(batch['past_target'].shape)
    print(batch['future_target'].shape)
    if i == 0:
        break
    i+=1
    
    