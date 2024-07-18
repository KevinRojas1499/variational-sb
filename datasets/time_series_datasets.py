import warnings

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
    prediction_length = 1
    context_length = dataset.metadata.prediction_length #* 3
    metadata = {'dim' :  int(data_dim),
                'pred_length' : int(prediction_length),
                'cond_length' : int(context_length)}

    splitter = InstanceSplitter(
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


    # Create the DataLoader
    train_dataloader = TrainDataLoader(
        dataset=train_ds,
        transform=splitter,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        stack_fn=batchify
    )
    test_dataloader = TrainDataLoader(
        dataset=test_ds,
        transform=splitter,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        stack_fn=batchify
    )
    return train_dataloader, test_dataloader, metadata
