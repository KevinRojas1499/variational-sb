import torch
import itertools
import abc
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class MyDataset(abc.ABC):
    """Implementes a basic iterable for toy datasets"""
    def __init__(self):
        super(MyDataset, self).__init__()

    @property
    @abc.abstractmethod
    def out_shape(self):
        pass
    
    @abc.abstractmethod
    def __iter__(self, index):
        return 

    @abc.abstractmethod
    def __next__(self):
        return 
    

    

class CheckerBoard(MyDataset):
    def __init__(self, batch_size, x_scalar=1.0, y_scalar=1.0):
        super(CheckerBoard, self).__init__()
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar
        self.batch_size = batch_size
    
    @property
    def out_shape(self):
        return [2]
    
    def __iter__(self):
        return self

    def __next__(self, n_samples=None):
        n = self.batch_size if n_samples is None else n_samples
        n_points = 3*n
        n_classes = 2
        freq = 5
        x = np.random.uniform(-(freq//2)*np.pi, (freq//2)*np.pi, size=(n_points, n_classes))
        mask = np.logical_or(np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0), \
        np.logical_and(np.sin(x[:,0]) < 0.0, np.sin(x[:,1]) < 0.0))
        y = np.eye(n_classes)[1*mask]
        x0=x[:,0]*y[:,0]
        x1=x[:,1]*y[:,0]
        sample=np.concatenate([x0[...,None],x1[...,None]],axis=-1)
        sqr=np.sum(np.square(sample),axis=-1)
        idxs=np.where(sqr==0)
        sample=np.delete(sample,idxs,axis=0)
        # res=res+np.random.randn(*res.shape)*1
        sample=torch.Tensor(sample)
        sample=sample[0:n,:]

        sample[:, 0] = self.x_scalar * sample[:, 0]
        sample[:, 1] = self.y_scalar * sample[:, 1]
        return sample

class Spiral(MyDataset):
    def __init__(self, batch_size, x_scalar=1.0, y_scalar=1.0):
        super(Spiral, self).__init__()
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar
        self.batch_size = batch_size
    
    @property
    def out_shape(self):
        return [2]
    
    def __iter__(self):
        return self

    def __next__(self, n_samples=None):
        n = self.batch_size if n_samples is None else n_samples
        theta = np.sqrt(np.random.rand(n))*3*np.pi-0.5*np.pi # np.linspace(0,2*pi,100)

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + 0.25*np.random.randn(n,2)
        samples = np.append(x_a, np.zeros((n,1)), axis=1)
        samples = samples[:,0:2]
        samples[:, 0] = self.x_scalar * samples[:, 0]
        samples[:, 1] = self.y_scalar * samples[:, 1]
        return torch.Tensor(samples)     
        
def get_dataset(opts):
    if  opts.dataset == 'spiral':
        return Spiral(opts.batch_size, x_scalar=1., y_scalar=1.)
    elif opts.dataset == 'checkerboard':
        return CheckerBoard(opts.batch_size)
    elif opts.dataset == 'mnist':
        dataset = MNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((28,28))]), download=True)
        data_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4)
        return itertools.cycle(data_loader)
    else:
        print('Dataset is not implemented')