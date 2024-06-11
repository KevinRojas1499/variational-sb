"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import torch.nn as nn

from utils.misc import batch_matrix_product

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  # @abc.abstractmethod
  # def prior_logp(self, z):
  #   """Compute log-density of the prior distribution.

  #   Useful for computing the log-likelihood via probability flow ODE.

  #   Args:
  #     z: latent code
  #   Returns:
  #     log probability density
  #   """
  #   pass

class VP(SDE):

  def __init__(self,T=1.,delta=1e-3, beta_min=0.1, beta_max=20):
    # dX = - .5 (beta_min + beta_max * t) X_t dt + (...) dW
    super().__init__()
    self._T = T
    self.delta = delta
    self.beta_min = beta_min
    self.beta_max = beta_max

  def T(self):
    return self._T
  
  def beta(self, t):
    return self.beta_min + (self.beta_max - self.beta_min) * t 
  
  def beta_int(self, t):
    return self.beta_min * t + (self.beta_max - self.beta_min) * t**2/2
  
  def marginal_prob(self, x, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    big_beta = self.beta_int(t)
    return torch.exp(-big_beta/2) * x, 1 - torch.exp(-big_beta)
  
  def marginal_prob_std(self, t):
    return (1 - torch.exp(-self.beta_int(t)))**.5
  
  def drift(self, x,t):
    return - .5 * self.beta(t) * x
  
  def diffusion(self, x,t):
    return self.beta(t)**.5
  
  def time_steps(self, n, device):
    from math import exp, log
    c = 1.6 * (exp(log(self.T()/self.delta)/n) - 1)
    t_steps = torch.zeros(n,device=device)
    t_steps[0] = self.delta
    exp_step = True
    for i in range(1,n):
      if exp_step:
        t_steps[i] = t_steps[i-1] + c * t_steps[i-1]
        if t_steps[i] >= 1:
          c = (self.T() - t_steps[i-1])/(n-i)
          t_steps[i] = t_steps[i-1] + c
          exp_step = False
      else:
        t_steps[i] = t_steps[i-1] + c
    
    t_steps[-1] = self.T()
    t_steps = self.T() - t_steps  
    t_steps = torch.flip(t_steps,dims=(0,))
    return t_steps.to(dtype=torch.float)
  
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)



class VariationaLinearlDrift(nn.Module):
  
  def __init__(self,dim):
    super().__init__()
    self.dim = dim
    self.A = nn.Linear(1,dim * dim)
    self.register_buffer('identity',torch.eye(dim).unsqueeze(0))
    # torch.nn.init.xavier_uniform_(self.A.weight)
    torch.nn.init.constant_(self.A.weight,0)
    # torch.nn.init.constant_(self.A.bias,0)
    print(self.A.bias.view(dim,dim))
  
  def forward(self, t):
    mat = self.A(t).reshape(-1, self.dim, self.dim) 
    return self.identity - 2 * (mat + mat.mT)

class LinearSchrodingerBridge(SDE):

  def __init__(self,dim, device, T=1.,delta=1e-3, beta_min=0.1, beta_max=20):
    # dX = - .5 (beta_min + beta_max * t) X_t dt + (...) dW
    super().__init__()
    self._T = T
    self.delta = delta
    self.beta_min = beta_min
    self.beta_max = beta_max
    
    self.A = VariationaLinearlDrift(dim).requires_grad_(False).to(device=device)

  def T(self):
    return self._T
  
  def beta(self, t):
    return self.beta_min + (self.beta_max - self.beta_min) * t 
  
  def beta_int(self, t):
    return self.beta_min * t + (self.beta_max - self.beta_min) * t**2/2
  
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
    Ats = self.A(time_pts.view(-1,1))
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
    return - .5 * self.beta(t) * batch_matrix_product(self.A(t), x) 
  
  def diffusion(self, x,t):
    return self.beta(t)**.5
  
  def time_steps(self, n, device):
    from math import exp, log
    c = 1.6 * (exp(log(self.T()/self.delta)/n) - 1)
    t_steps = torch.zeros(n,device=device)
    t_steps[0] = self.delta
    exp_step = True
    for i in range(1,n):
      if exp_step:
        t_steps[i] = t_steps[i-1] + c * t_steps[i-1]
        if t_steps[i] >= 1:
          c = (self.T() - t_steps[i-1])/(n-i)
          t_steps[i] = t_steps[i-1] + c
          exp_step = False
      else:
        t_steps[i] = t_steps[i-1] + c
    
    t_steps[-1] = self.T()
    t_steps = self.T() - t_steps  
    t_steps = torch.flip(t_steps,dims=(0,))
    return t_steps.to(dtype=torch.float)
  
  def prior_sampling(self, shape, device):
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
  
  def time_steps(self, n, device):
    from math import exp, log
    c = 1.6 * (exp(log(self.T()/self.delta)/n) - 1)
    t_steps = torch.zeros(n,device=device)
    t_steps[0] = self.delta
    exp_step = True
    for i in range(1,n):
      if exp_step:
        t_steps[i] = t_steps[i-1] + c * t_steps[i-1]
        if t_steps[i] >= 1:
          c = (self.T() - t_steps[i-1])/(n-i)
          t_steps[i] = t_steps[i-1] + c
          exp_step = False
      else:
        t_steps[i] = t_steps[i-1] + c
    
    t_steps[-1] = self.T()
    t_steps = self.T() - t_steps  
    t_steps = torch.flip(t_steps,dims=(0,))
    return t_steps.to(dtype=torch.float)
  
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)
  
  
def get_sde(sde_name):
  if sde_name == 'vp':
    return VP()
  if sde_name == 'sb':
    return LinearSchrodingerBridge()
  elif sde_name == 'cld':
    return CLD()