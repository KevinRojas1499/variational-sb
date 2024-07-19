from itertools import cycle 
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from datasets.toy_datasets import MyDataset, Spiral, CheckerBoard
from datasets.time_series_datasets import TimeSeriesDataset

def get_dataset(opts) -> MyDataset:
    ds_name = opts.dataset
    batch_size = opts.batch_size
    if  ds_name == 'spiral':
        return Spiral(batch_size, x_scalar=1., y_scalar=1.)
    elif ds_name == 'checkerboard':
        return CheckerBoard(batch_size)
    elif ds_name == 'mnist':
        dataset = MNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((28,28))]), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return cycle(data_loader)
    elif ds_name == 'exchange_rate':
        return TimeSeriesDataset(ds_name,batch_size,100)
    else:
        print('Dataset is not implemented')