import torch
import abc
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical

from utils.sde_lib import SDE


class Dataset(abc.ABC):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    @abc.abstractmethod
    def sample(self,num_samples):
        pass

class GMM(Dataset):
    def __init__(self,weights, means, covariances, sde : SDE = None) -> None:
        super().__init__(means[0].shape[0])
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.n = len(weights)
        gaussians = MultivariateNormal(means,covariances)
        cat = Categorical(weights)
        self.dist = MixtureSameFamily(cat, gaussians)
        self.sde = sde
    
    def sample(self,num_samples):
        return self.dist.sample((num_samples,))
    
    
    
    def score(self, x, t):
        assert self.sde is not None, "Please specify SDE to dataset"
        with torch.enable_grad():
            x_aux = x.clone().requires_grad_(True)
            # We assume only one t
            means_t, var = self.sde.marginal_prob(self.means, t[0])
            var_t = (1-var) * self.covariances + torch.eye(self.dim,device=x_aux.device).unsqueeze(0) * var
            gaussians = MultivariateNormal(means_t,var_t)
            cat = Categorical(self.weights)
            dist_t = MixtureSameFamily(cat, gaussians)
            log_prob = dist_t.log_prob(x_aux)
            grad = torch.autograd.grad(log_prob.sum(),x_aux,create_graph=True)[0]
            return  grad
       
class CheckerBoard(Dataset):
    def __init__(self, x_scalar=1.0, y_scalar=1.0):
        super().__init__(2)
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar


    def sample(self, num_samples):
        n = num_samples
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


class Spiral(Dataset):
    def __init__(self, x_scalar=1.0, y_scalar=1.0):
        super().__init__(2)
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar
        

    def sample(self, num_samples):
        n = num_samples
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
    if opts.dataset == 'gmm':
         return get_gmm(2, None, device='cuda')
    elif opts.dataset == 'spiral':
        return Spiral(x_scalar=.4, y_scalar=3.2)
    elif opts.dataset == 'checkerboard':
        return CheckerBoard()
    else:
        print('Dataset is not implemented')
        
def get_gmm(dim, sde, device):
    if dim == 2:
        return GMM(
            torch.tensor([.33,.33, .33],device=device),
            torch.tensor([[-5.,-5.],[5.,5.],[-5,8.]],device=device),
            torch.tensor([[[1., -.3],[-.3,1.]], [[1., .5],[.5,1.]],[[1., 0],[0,1.]]],device=device),
            sde
        )
        # return GMM(
        #     torch.tensor([.5,.5],device=device),
        #     torch.tensor([[-5.,-5.],[5.,5.]],device=device),
        #     torch.tensor([[[1., 0],[0,1.]], [[1., 0],[0,1.]]],device=device),
        #     sde
        # )
    elif dim == 1:
       return GMM(
            torch.tensor([.2,.8],device=device),
            torch.tensor([[-5],[5.]],device=device),
            torch.tensor([[[1.]], [[1]]],device=device),
            sde
        ) 
