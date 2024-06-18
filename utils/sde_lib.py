"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import torch.nn as nn
from math import pi, log

from utils.diff import batch_div_exact, hutch_div
from utils.misc import batch_matrix_product

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self):
    super().__init__()

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass
  
  @abc.abstractmethod
  def drift(self, x,t, forward=True):
    """Drift of the SDE"""
    pass
  
  @abc.abstractmethod
  def diffusion(self, x,t, forward=True):
    """Diffusion of the SDE"""
    pass
  
  @abc.abstractmethod
  def probability_flow_drift(self, xt, t):
    """Probability flow drift"""
    pass
  
  def sample(self, shape, device, backward=True, in_cond=None, prob_flow=False):
    with torch.no_grad():
      xt = self.prior_sampling(shape,device) if backward else in_cond
      assert xt is not None
      n_time_pts = 100      
      time_pts = torch.linspace(0., self.T(), n_time_pts, device=device)
      trajectories = torch.empty((xt.shape[0], n_time_pts-1, *xt.shape[1:]),device=xt.device) 

      for i, t in enumerate(time_pts):
        if i == n_time_pts - 1:
          break
        dt = time_pts[i+1] - t 
        dt = -dt if backward else dt 
        t_shape = self.T() - t if backward else t
        t_shape = t_shape.unsqueeze(-1).expand(xt.shape[0],1)
        if prob_flow:
          drift = self.probability_flow_drift(xt,t_shape)
          xt = xt + drift * dt
        else:
          drift = self.drift(xt,t_shape, forward=(not backward))
          xt = xt + drift * dt + torch.randn_like(xt) * self.diffusion(xt,t) * dt.abs().sqrt()
        trajectories[:,i] = xt
      return xt, trajectories

class LinearSDE(SDE):
  def __init__(self):
    super().__init__()
    
  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """ Returns the marginal prob dist """
    pass  
class VP(LinearSDE):

  def __init__(self,T=1.,delta=1e-3, beta_min=0.1, beta_max=5, model_backward=None):
    # dX = - .5 (beta_min + beta_max * t) X_t dt + (...) dW
    super().__init__()
    self._T = T
    self.delta = delta
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.backward_score = model_backward

  def T(self):
    return self._T
  
  def beta(self, t):
    return self.beta_max
  
  def beta_int(self, t):
    return self.beta_max * t
  
  def marginal_prob(self, x, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    big_beta = self.beta_int(t)
    return torch.exp(-big_beta/2) * x, 1 - torch.exp(-big_beta)
  
  def marginal_prob_std(self, t):
    return (1 - torch.exp(-self.beta_int(t)))**.5
  
  def drift(self, x,t, forward=True):
    if forward:
      return -.5 * self.beta(t) * x
    else:
      return -.5 * self.beta(t) * x - self.beta(t) * self.backward_score(x,t)
  
  def probability_flow_drift(self, xt, t):
    beta = self.beta(t)
    return -.5 * beta * (xt + self.backward_score(xt,t))
  
  def diffusion(self, x,t):
    return self.beta(t)**.5

  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)

class EDM(LinearSDE):

  def __init__(self,T=80.,delta=1e-3, model_backward=None):
    # dX = - .5 (beta_min + beta_max * t) X_t dt + (...) dW
    super().__init__()
    self._T = T
    self.delta = delta
    self.backward_score = model_backward

  def T(self):
    return self._T
  
  def beta(self, t):
    return 2 * t
  
  def beta_int(self, t):
    return t**2
  
  def marginal_prob(self, x, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    return  x, self.beta_int(t)
  
  def marginal_prob_std(self, t):
    return self.beta_int(t)**.5
  
  def drift(self, x,t, forward=True):
    if forward:
      return 0.
    else:
      return 0. - self.beta(t) * self.backward_score(x,t)
  
  def probability_flow_drift(self, xt, t):
    return - .5 * self.beta(t) * self.backward_score(xt,t)
  
  def diffusion(self, x,t):
    return self.beta(t)**.5

  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device) * self.marginal_prob_std(self.T())


class SchrodingerBridge(SDE):
  """ 
    Note that this is not a general SB, it is implemented so that after optimized
    the linear drift transports to a standard normal
  """
  def __init__(self, T=1.,delta=1e-3, beta_min=0.1, beta_max=5, forward_score=None, backward_score=None):
    super().__init__()
    self._T = T
    self.delta = delta
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.forward_score = forward_score
    self.backward_score = backward_score

  def T(self):
    return self._T
  
  def beta(self, t):
    return self.beta_max
  
  def beta_int(self, t):
    return self.beta_max * t
  
  def drift(self, x,t, forward=True):
    beta = self.beta(t)
    if forward:
      return -.5 * beta * x + beta * self.forward_score(x,t)
    else:
      return -.5 * beta * x - beta * self.backward_score(x,t)
  
  def diffusion(self, x,t):
    return self.beta(t)**.5
  
  def eval_sb_loss(self, in_cond, time_pts):
    # We assume that time_pts is a discretization of the interval [0,T]
    n_time_pts = time_pts.shape[0]
    
    xt = in_cond.detach().clone().requires_grad_(True)
    batch_size, d = xt.shape[0], xt.shape[-1]
    loss = 0
    trajectories = torch.empty((in_cond.shape[0], n_time_pts-1, *in_cond.shape[1:]),device=in_cond.device) 
    forward_scores = torch.empty_like(trajectories)

    for i, t in enumerate(time_pts):
      if i == n_time_pts - 1:
        break
      t_shape = t.unsqueeze(-1).expand(batch_size,1)
      
      bt = self.beta(t)
      forward_score = bt * self.forward_score(xt,t_shape) # beta * fw_score
        
      trajectories[:,i] = xt
      forward_scores[:,i] = forward_score
      
      # First we compute everything, we then take an euler step
      dt = time_pts[i+1] - t
      xt = xt + (-.5 * bt * xt + forward_score) * dt + torch.randn_like(xt) * self.diffusion(xt,t) * dt.abs().sqrt()
    
    # We now compute the loss fn, we first need to do some reshaping
    time_pts_shaped = time_pts[:-1].repeat(batch_size)
    bt = self.beta(time_pts_shaped)
    flat_traj = trajectories.reshape(-1,xt.shape[-1])
    backward_scores = bt * self.backward_score(flat_traj,time_pts_shaped)
    
    # div_term = bt * batch_div_exact(backward_scores,flat_traj,time_pts_shaped) + .5 * d * bt 
    div_term = bt * hutch_div(backward_scores,flat_traj,time_pts_shaped) + .5 * d * bt 
    loss = .5 * torch.sum((backward_scores + forward_scores.view(-1,xt.shape[-1]))**2)/batch_size \
      + torch.sum(div_term)/batch_size
    loss = dt * loss
    loss += .5 * torch.sum(xt**2)/batch_size + .5 * d * log(2*pi)
    return loss
  
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)

  def probability_flow_drift(self, xt, t):
    beta = self.beta(t)
    return -.5 * beta * (xt - self.forward_score(xt,t) \
      + self.backward_score(xt, t))

class VariationaLinearlDrift():
  
  def __init__(self,dim, net):
    super().__init__()
    self.dim = dim
    self.A = net
    self.identity = torch.eye(dim).unsqueeze(0)
    # 0 Initialization is important so that it pushes to a standard normal
    # torch.nn.init.constant_(self.A.weight,0)
    # torch.nn.init.constant_(self.A.bias,0)
  
  def forward(self, t):
    mat = self.A(t).reshape(-1, self.dim, self.dim) 
    return self.identity - 2 * (mat + mat.mT)

class LinearSchrodingerBridge(LinearSDE, SchrodingerBridge):
  """ 
    Note that this is not a general SB, it is implemented so that after optimized
    the linear drift transports to a standard normal
  """
  def __init__(self,dim, device, T=1.,delta=1e-3, beta_min=0.1, beta_max=5, forward_model=None, backward_model=None):
    """ Here the backward model is a standard backwards score
        The forward model is such that it receives t of shape [bs,1] and outputs a matrix [bs, d,d]
    """
    super().__init__()
    self._T = T
    self.delta = delta
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.D = VariationaLinearlDrift(dim,forward_model).to(device=device).requires_grad_(True)
    self.dim = self.D.dim
    self.backward_score = backward_model
    
  def T(self):
    return self._T
  
  def beta(self, t):
    return self.beta_max
  
  def beta_int(self, t):
    return self.beta_max * t
  
  def int_beta_ds(self, t):
    # Curently using Simpsons Method
    num_pts = 1000
    t_shape = t.unsqueeze(-1).expand(-1,num_pts,-1)
    dt = t/num_pts
    time_pts = torch.arange(num_pts,device=t.device).unsqueeze(-1) * t_shape/num_pts
    multipliers = torch.ones(num_pts, device=t.device)
    multipliers[1:-1:2] = 4
    multipliers[2:-1:2] = 2
    multipliers = multipliers.view(1,-1,1,1)
    Ats = self.D(time_pts.view(-1,1))
    Ats = Ats.view(-1,num_pts, Ats.shape[-1], Ats.shape[-1])
    betas = self.beta(time_pts).unsqueeze(-1)
    return torch.sum(betas * Ats * multipliers,dim=1) * dt.unsqueeze(-1)/3

  def compute_variance(self, t):
    int_mat = self.int_beta_ds(t)
    dim = int_mat.shape[-1]
    ch_power = torch.zeros((t.shape[0], 2 * dim, 2 * dim),device=int_mat.device)
    ch_power[:,:dim, :dim] = -.5 * int_mat
    ch_power[:,dim:, dim:] = .5 * int_mat.mT
    ch_power[:, :dim, dim:] = self.beta_int(t).view(-1,1,1) * torch.eye(dim,device=int_mat.device).unsqueeze(0).expand(t.shape[0],-1,-1)
    ch_pair = torch.linalg.matrix_exp(ch_power)
    C = ch_pair[:, :dim, dim:]
    H_inv = ch_pair[:, :dim, :dim]
    cov = C @ H_inv
    diag, Q = torch.linalg.eigh(cov)
    L = Q @ torch.diag_embed(diag.sqrt()) @ Q.mH
    invL = Q @ torch.diag_embed(1/(diag.sqrt())) @ Q.mH
    max_eig = diag[:,-1].unsqueeze(-1)
    return cov, L, invL, max_eig
  
  def marginal_prob_std(self, t):
    return self.compute_variance(t)[1]
  
  def marginal_prob(self, x, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    big_beta = (-.5 * self.int_beta_ds(t)).matrix_exp()
    cov, L, invL, max_eig = self.compute_variance(t)
    return batch_matrix_product(big_beta, x), L, invL, max_eig
  
  def unscaled_marginal_prob_std(self, t):
    mat = (.5 * self.int_beta_ds(t)).matrix_exp()
    cov, L, invL, _ = self.compute_variance(t)
    return mat @ L
  
  def drift(self, x,t):
    return - .5 * self.beta(t) * batch_matrix_product(self.D(t), x) 
  
  def diffusion(self, x,t):
    return self.beta(t)**.5
  
  def prior_sampling(self, shape, device):
    # return torch.randn(*shape, dtype=torch.float, device=device)
    L = self.compute_variance(torch.tensor([[self.T()]],device=device))[1][0]
    return (L @ torch.randn(*shape, dtype=torch.float, device=device).T).T


class CLD(SDE):
  # We assume that images have shape [B, C, H, W] 
  # Additionally there has been added channels as momentum
  def __init__(self,T=1.,delta=1e-3, gamma=2, beta_min=4., beta_max=0.):
    super().__init__()
    self._T = T
    self.delta = delta
    self.gamma = gamma
    self.is_augmented = True
    self.beta_min = beta_min
    self.beta_max = beta_max

  def T(self):
    return 1.
  
  def beta(self, t):
    return self.beta_min + (self.beta_max - self.beta_min) * t 
  
  def beta_int(self, t):
    return (self.beta_min * t + (self.beta_max - self.beta_min) * t**2/2)/2
  
  
  def get_exp_At_components(self, t):
    b = self.beta_int(t)
    exp  = torch.exp(-b) 
    return exp * (b + 1), exp * b, exp * -b, exp * (1-b)
  
  def marginal_prob_mean(self, z, t):
    x, v = torch.chunk(z,2,dim=1) # Decompose in channels    
    a,b,c,d = self.get_exp_At_components(t)
    exp  = torch.exp(-b) 
    new_x = exp * ((b+1) * x + b * v)
    new_v = exp * (-b * x + (1-b) * v)
    return torch.cat((new_x,new_v),dim=1)
    
  
  def get_sig_components(self, t):
    t_aux = t.clone().to(dtype=torch.float64)
    b = self.beta_int(t_aux)
    exp = torch.exp(2 * b)
    sig_xx = (2 * b**2 - 2 * b + 1) * exp - 1 
    sig_xv = - 2 * b **2 * exp
    sig_vv = (2*b**2 + 2 * b + 1) * exp -1 
    
    return sig_xx.to(t.dtype), sig_xv.to(t.dtype), sig_vv.to(t.dtype)
  
  def marginal_prob_var_components(self, t):
    a,b,c,d = self.get_exp_At_components(t)
    xx, xv, vv = self.get_sig_components(t)
    
    temp_xx = a * xx + b * xv
    temp_xv = c * xx + d * xv
    temp_vx = a * xv + b * vv
    temp_vv = c * xv + d * vv
    
    return a * temp_xx + b * temp_vx, a * temp_xv + b * temp_vv, \
           c * temp_xx + d * temp_vx, c * temp_xv + d * temp_vv
    
  def marginal_prob_std(self, t):
    # Returns mean, std
    a,b,c,d = self.marginal_prob_var_components(t)
    return (a**.5  ,  # 0 is  ommitted \
            c/a**.5,(d-c**2/a)**.5)
  
  def multiply_std(self, z, t):
    a, c, d = self.marginal_prob_std(t)
    x, v = z.chunk(2,dim=1)
    return torch.cat((a * x, c * x  + d * v), dim=1)

  def multiply_inv_std(self, z, t):
    a, c, d = self.marginal_prob_std(t)
    x, v = z.chunk(2,dim=1)
    return torch.cat((a * v, d * x - c * v), dim=1)/ (a*d)

  def marginal_prob(self, z, t):
    # Returns mean, std
    mean = self.marginal_prob_mean(z,t)
    a,c,d = self.marginal_prob_std(t)
    
    return mean, (a, c, d)
  
  def drift(self, z,t):
    x,v = torch.chunk(z,2,dim=-1)
    
    d_x = v
    d_v = -x - self.gamma * v
    return torch.cat((d_x,d_v),dim=-1)
  
  def diffusion(self, z,t):
    x,v = torch.chunk(z,2,dim=-1)
    # TODO : DO 
    return 2**.5
  
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)
  

def get_sde(sde_name):
  if sde_name == 'vp':
    return VP()
  if sde_name == 'edm':
    return EDM()
  if sde_name == 'sb':
    return SchrodingerBridge()
  elif sde_name == 'cld':
    return CLD()