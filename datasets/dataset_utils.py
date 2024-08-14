from itertools import cycle 
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader

from datasets.toy_datasets import MyDataset, Spiral, CheckerBoard

def get_dataset(opts) -> MyDataset:
    ds_name = opts.dataset
    batch_size = opts.batch_size
    if  ds_name == 'spiral':
        return Spiral(batch_size, x_scalar=3, y_scalar=3)
    elif ds_name == 'checkerboard':
        return CheckerBoard(batch_size,x_scalar=1.,y_scalar=7.)
    elif ds_name == 'mnist':
        dataset = MNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((28,28))]), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return cycle(data_loader)
    elif ds_name == 'fashion':
        dataset = FashionMNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((28,28))]), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return cycle(data_loader)
    else:
        print('Dataset is not implemented')