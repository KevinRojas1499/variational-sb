import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from datasets.toy_datasets import MyDataset, Spiral, CheckerBoard

def get_dataset(opts) -> MyDataset:
    ds_name = opts.dataset
    batch_size = opts.batch_size
    if  ds_name == 'spiral':
        return Spiral(batch_size, x_scalar=.4, y_scalar=3.2)
    elif ds_name == 'checkerboard':
        return CheckerBoard(batch_size,x_scalar=1.,y_scalar=7.)
    elif ds_name == 'mnist':
        dataset = MNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((28,28))]), download=True)
        return dataset, [1,28,28]
    elif ds_name == 'fashion':
        dataset = FashionMNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((28,28))]), download=True)
        return dataset, [1,28,28]
    elif ds_name == 'cifar':
        dataset= CIFAR10('.', train=True,transform=transforms.Compose([transforms.ToTensor()]), download=True)
        return dataset, [3,32,32]
    else:
        print('Dataset is not implemented')