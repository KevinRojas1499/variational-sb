import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def batch_matrix_product(A, x):
    # Given matrices [n,d,d] and vectors [n,d] computes the entrywise product
    return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) 