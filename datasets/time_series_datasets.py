import warnings
from itertools  import cycle
from datasets.toy_datasets import MyDataset

from gluonts.dataset.repository import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.field_names import FieldName
from gluonts.transform import InstanceSplitter, ExpectedNumInstanceSampler
from gluonts.torch.batchify import batchify


def get_transformed_dataset(name, batch_size, num_batches_per_epoch):
    # TODO : Can I figure out this warning
    warnings.filterwarnings(action='ignore', category=FutureWarning, message=r".*Use a DatetimeIndex.*")
    dataset = get_dataset(name)
    data_dim = 2 # dataset.metadata.feat_static_cat[0].cardinality


    train_grouper = MultivariateGrouper(max_target_dim=data_dim)
    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)),
                                   max_target_dim=data_dim)


    train_ds = train_grouper(dataset.train)
    test_ds = test_grouper(dataset.test)

    # Define the transformation
    prediction_length = 12
    true_pred_length = dataset.metadata.prediction_length
    context_length = dataset.metadata.prediction_length * 3
    metadata = {'dim' :  int(data_dim),
                'pred_length' : int(prediction_length),
                'cond_length' : int(context_length)}

    splitter_train = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=prediction_length,
        ),
        past_length=context_length,
        future_length=prediction_length
    )
    splitter_test = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=true_pred_length,
        ),
        past_length=context_length,
        future_length=true_pred_length # Only difference is that we are predicting for longer
    )


    # Create the DataLoader
    train_dataloader = TrainDataLoader(
        dataset=train_ds,
        transform=splitter_train,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        stack_fn=batchify
    )
    test_dataloader = TrainDataLoader(
        dataset=test_ds,
        transform=splitter_test,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        stack_fn=batchify
    )
    return train_dataloader, test_dataloader, metadata


class TimeSeriesDataset(MyDataset):
    def __init__(self,name, batch_size, num_batches_per_epoch,train=True):
        super().__init__()
        train_loader, _, self.metadata = get_transformed_dataset(name, batch_size, num_batches_per_epoch)
        self.train_loader = cycle(iter(train_loader))
    @property
    def out_shape(self):
        return [self.metadata['pred_length'], self.metadata['dim']]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = next(self.train_loader) 
        return batch['future_target'], batch['past_target'] 