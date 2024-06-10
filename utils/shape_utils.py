import torch

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape((x.shape[0], *self.shape))
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.reshape((x.shape[0], -1))
    
class Merge(torch.nn.Module):
    def __init__(self, net, shape):
        super(Merge, self).__init__()
        self.net = net
        self.reshape = Reshape(shape)
    def forward(self, x, t):
        x = self.reshape(x)
        out = self.net(x, t)
        out = Flatten()(out)
        return out