import torch

def gradient(y, x):
    " Compute dy/dx @ grad_outputs "
    " train_points: [B, DIM]"
    " model: R^{DIM} --> R"
    " grads: [B, DIM]"
    grads = torch.autograd.grad(y.sum(), x, 
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)[0]
    return grads

def partial_t_j(f, x, t, j):
    """
    :param s: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: (dsdt)_j (torch.tensor) of shape [B, 1]
    """
    assert j <= x.shape[-1]
    s = f(x, t)
    v = torch.zeros_like(s)
    v[:, j] = 1.
    dy_j_dx = torch.autograd.grad(
                   s,
                   t,
                   grad_outputs=v,
                   retain_graph=True,
                   create_graph=True,
                   allow_unused=False)[0]  # shape [B, N]
    return dy_j_dx

def batch_div_exact(f,x,t):
    div = 0
    for i in range(x.shape[-1]):
        div += torch.autograd.grad(f[:,i].sum(),x,
                                   create_graph=True,
                                   allow_unused=True,
                                   retain_graph=True)[0][:,i:i+1]
    return div

def hutch_div(score_model, sample, time_steps, already_evaluated=True):
    """Compute the divergence of the score-based model with Skilling-Hutchinson.
        The already evaluated parameter should be switched to False when the
        score_model is a function of (sample,time_steps) instead of the output of that fn
    """
    with torch.enable_grad():
        sample.requires_grad_(True)
        repeat = 1
        divs = torch.zeros((sample.shape[0],), device=sample.device, requires_grad=False) #div: [B,]
        for _ in range(repeat):
            epsilon = torch.randn_like(sample)
            s = score_model if already_evaluated else score_model(sample, time_steps)
            score_e = torch.sum(s * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample,
                                                retain_graph=True,
                                                create_graph=True,
                                                allow_unused=False)[0]
            divs += torch.sum(grad_score_e * epsilon, dim=(1))  
        divs = divs/repeat
    return divs.unsqueeze(-1)


def t_finite_diff(fn, x, t, hs=0.001, hd=0.0005):
    up = hs**2 * fn(x, t+hd) + (hd**2 - hs**2) * fn(x, t) - hd**2 * fn(x, t-hs)
    low = hs * hd * (hd+hs)
    return up/low  
