"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
from tqdm import tqdm

def reshape_t(t, xt):
  ones = [1] * len(xt.shape[1:])
  return t.reshape(-1, *ones)

class SDE(abc.ABC):
  def __init__(self, is_augmented):
    # Augmentation refers to adding momentum
    super().__init__()
    self.is_augmented = is_augmented 
    

  @property
  @abc.abstractmethod
  def T(self):
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    pass
  
  @abc.abstractmethod
  def drift(self, x,t, forward=True,cond=None):
    pass
  
  @abc.abstractmethod
  def diffusion(self, x,t, forward=True):
    pass
  
  @abc.abstractmethod
  def probability_flow_drift(self, xt, t,cond=None):
    pass
  
  @torch.no_grad()
  def sample(self, shape, device, backward=True, 
             in_cond=None, prob_flow=True, 
             cond=None, n_time_pts=1000, return_traj=False):
    xt = self.prior_sampling(shape,device) if backward else in_cond
    assert xt is not None
    time_pts = torch.linspace(0. if backward else self.delta, self.T - self.delta, n_time_pts, device=device)
    time_pts = torch.cat((time_pts, torch.ones_like(time_pts[:1]) * self.T))
    if return_traj:
      trajectories = torch.empty((xt.shape[0], n_time_pts, *xt.shape[1:]),device=xt.device) 
    
    N = time_pts.shape[0]
    for i, t in tqdm(enumerate(time_pts), total=N, leave=False):
      if return_traj:
        trajectories[:,i] = xt
      if i == N - 1:
        break
      dt = time_pts[i+1] - t 
      dt = -dt if backward else dt 
      t_shape = self.T - t if backward else t
      t_shape = t_shape.unsqueeze(-1).expand(xt.shape[0])
      if prob_flow:
        drift = self.probability_flow_drift(xt,t_shape, cond)
        if backward and i+1 != N-1:
          xt_hat = xt + drift * dt
          t_hat = self.T - time_pts[i+1] if backward else time_pts[i+1]
          t_hat = t_hat.unsqueeze(-1).expand(xt.shape[0])
          xt = xt + .5 * dt * (drift + self.probability_flow_drift(xt_hat,t_hat,cond))
        else:
          xt = xt + drift * dt
      else:
        drift = self.drift(xt,t_shape, forward=(not backward),cond=cond)
        xt = xt + drift * dt 
        if i +1 != N-1:
          xt += torch.randn_like(xt) * self.diffusion(xt,self.T - t) * dt.abs().sqrt()
    
    xt = xt.chunk(2,dim=1)[0] if self.is_augmented else xt
    return xt, (trajectories if return_traj else None)
class LinearSDE(SDE):
  
  def __init__(self, backward_model, is_augmented):
    SDE.__init__(self, is_augmented)
    self.backward_score = backward_model
    
  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """ Returns the marginal prob dist """
    pass  
class VP(LinearSDE):

  def __init__(self,T=1.,delta=1e-3, beta_max=5, backward_model=None):
    LinearSDE.__init__(self,backward_model=backward_model,is_augmented=False)
    self._T = T
    self.delta = delta
    self.beta_max = 19.9

  @property
  def T(self):
    return self._T
  
  def beta(self, t):
    b_min = 0.1
    return b_min+ t*(self.beta_max-b_min) # self.beta_max
  
  def beta_int(self, t):
    b_min = 0.1
    return  b_min * t +(self.beta_max-b_min) * t**2/2
  
  def marginal_prob(self, x, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    big_beta = self.beta_int(t)
    return torch.exp(-big_beta/2) * x, 1 - torch.exp(-big_beta)
  
  def marginal_prob_std(self, t):
    return (1 - torch.exp(-self.beta_int(t)))**.5
  
  def drift(self, x,t, forward=True,cond=None):
    beta = self.beta(t)
    if forward:
      return -.5 * beta * x
    else:
      return -.5 * beta * x - beta * self.backward_score(x,t,cond)
  
  def probability_flow_drift(self, xt, t,cond=None):
    beta = self.beta(t.reshape(-1,*([1] * len(xt.shape[1:]))))
    return -.5 * beta * (xt + self.backward_score(xt,t,cond))
  
  def diffusion(self, x,t):
    return self.beta(t)**.5

  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)

class EDM(LinearSDE):

  def __init__(self,T=80.,delta=1e-3, model_backward=None):
    super().__init__(backward_model=model_backward, is_augmented=False)
    self._T = T
    self.delta = delta

  @property
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
    return torch.randn(*shape, dtype=torch.float, device=device) * self.marginal_prob_std(self.T)

class VSDM(LinearSDE):
  """ 
    Note that this is not a general SB, it is implemented so that after optimized
    the linear drift transports to a standard normal
  """
  def __init__(self,T=1.,delta=1e-3, beta_max=10, forward_model=None, backward_model=None):
    """ Here the backward model is a standard backwards score
        The forward model is such that it receives t of shape [bs] and outputs a matrix [bs, shape of input]
        The dimension is infered from the forward model, so if it doesn't behave in this way it won't work
        We internally assign the forward model to be the multiplication against this matrix
    """
    LinearSDE.__init__(self,backward_model=backward_model, is_augmented=False)
    self.forward_score = forward_model
    self._T = T
    self.delta = delta
    self.beta_max = beta_max
    self.beta_min = 0.1
    self.beta_r = 1.7

    
  @property
  def T(self):
    return self._T

  def beta(self, t):
    return self.beta_max
    return self.beta_min+ t**self.beta_r *(self.beta_max-self.beta_min) # self.beta_max
  
  def beta_int(self, t):
    return self.beta_max * t
    return  self.beta_min * t + t**(self.beta_r+1)/(self.beta_r+1) * (self.beta_max-self.beta_min)
  
  @property
  def forward_score(self):
    return lambda x,t, cond=None : self.At(t) *  x
  
  @forward_score.setter
  def forward_score(self,forward_model):
    self.At = forward_model
    
  def diffusion(self, x, t, forward=True):
    return self.beta(t)**.5
    
  def drift(self, x,t, forward=True, cond=None):
    beta = self.beta(t)
    if forward:
      return -.5 * beta * x + beta * self.forward_score(x,t,cond)
    else:
      return -.5 * beta * x + beta * self.forward_score(x,t,cond) - beta * self.backward_score(x,t,cond)

  def probability_flow_drift(self, xt, t, cond=None):
    beta = self.beta(reshape_t(t,xt))
    return -.5 * beta * (xt - 2 * self.forward_score(xt,t,cond) \
      + self.backward_score(xt, t,cond))

  def get_trajectories_for_loss(self, in_cond, time_pts,forward=True, cond=None):
    n_time_pts = time_pts.shape[0]
    
    xt = in_cond.detach().clone().requires_grad_(True)
    batch_size = xt.shape[0]
    trajectories = torch.empty((batch_size, n_time_pts, *in_cond.shape[1:]),device=in_cond.device) 
    policies = torch.empty_like(trajectories)
    cur_score = self.forward_score if forward else self.backward_score
    for i, t in enumerate(time_pts):
      if not forward:
        t = self.T - t
      t_shape = t.expand(batch_size)
      
      bt = self.beta(t)
      save_idx = i if forward else -(i+1)
      policy = bt**.5 * cur_score(xt,t_shape,cond) # g * fw_score
      policies[:,save_idx] = policy
      trajectories[:,save_idx] = xt
      
      if i == n_time_pts - 1:
        break
      
      # First we compute everything, we then take an euler step
      dt = time_pts[i+1] - time_pts[i]
      dt = dt if forward else -dt
      if forward:
        xt = xt + (-.5 * bt * xt + bt**.5 * policy) * dt + torch.randn_like(xt) * self.diffusion(xt,t) * dt.abs().sqrt()
      else:
        xt = xt + (-.5 * bt * xt + bt * self.forward_score(xt,t_shape,cond) - bt**.5 * policy) * dt + torch.randn_like(xt) * self.diffusion(xt,t) * dt.abs().sqrt()
        
    return xt,trajectories,policies
  
  def get_transition_params(self, x, t):
    A = self.At(t)
    scale = torch.exp(-.5 * self.beta_int(t) * (1 - 2 * A))
    
    return scale, ((1-scale**2)/(1 - 2 * A)).sqrt()

  def marginal_prob(self, x, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    scale, std = self.get_transition_params(x,t)
    return scale * x, std

  def prior_sampling(self, shape, device):
    noise = torch.randn(shape,device=device)
    t = torch.ones(noise.shape[0], device=device) * self.T
    t = reshape_t(t, noise)
    
    scale, std = self.get_transition_params(noise,t)
    noise = std * noise 
    # print("COVARIANCE")
    # print(noise.var(dim=0))
    return  noise

class LinearMomentumSchrodingerBridge(LinearSDE):
  """ 
    Note that this is not a general SB, it is implemented so that after optimized
    the linear drift transports to a standard normal
  """
  def __init__(self,T=1.,delta=1e-3, gamma=2., beta_max=10, forward_model=None, backward_model=None):
    """ Here the backward model is a standard backwards score
        The forward model is such that it receives t of shape [bs,1] and outputs a matrix [bs, d,2d]
        The dimension is infered from the forward model, so if it doesn't behave in this way it won't work
        We internally assign the forward model to be the multiplication against this matrix
    """
    LinearSDE.__init__(self,backward_model=backward_model, is_augmented=True)
    self.forward_score = forward_model
    self._T = T
    self.beta_max = beta_max
    self.delta = delta
    self.gamma = gamma
    
  @property
  def T(self):
    return self._T

  def beta(self, t):
    return self.beta_max
    b_min = 0.01
    return b_min+ t*(self.beta_max-b_min) # self.beta_max
  
  def beta_int(self, t):
    return self.beta_max * t
    b_min = 0.01
    return  b_min * t +(self.beta_max-b_min) * t**2/2
  
  @property
  def forward_score(self):
    def f_score(z,t,cond=None):
      x,v = z.chunk(2,dim=1)
      at,ct = self.At(t).chunk(2,dim=1)
      return at * x + ct * v
    return f_score
  
  @forward_score.setter
  def forward_score(self,forward_model):
    self.At = forward_model
  
  def probability_flow_drift(self,z,t, cond=None):
    beta = self.beta(reshape_t(t,z))
    xt,vt = z.chunk(2,dim=1)
    v_drift = -xt -self.gamma * vt + self.gamma * (2 * self.forward_score(z,t,cond) - self.backward_score(z,t,cond))
    return .5 * beta * torch.cat((vt,v_drift),dim=1)
      
  def drift(self,z,t, forward=True,cond=None):
    beta = self.beta(t)
    xt,vt = z.chunk(2,dim=1)
    if forward:
      v_drift = -xt - self.gamma * vt + 2 * self.gamma * self.forward_score(z,t,cond)
      return .5 * beta * torch.cat((vt, v_drift),dim=1)
    else:
      v_drift = -xt - self.gamma * vt + 2 * self.gamma * self.forward_score(z,t,cond) \
        - 2 * self.gamma * self.backward_score(z,t,cond)
      return .5 * beta * torch.cat((vt,v_drift),dim=1)
    
  
  def get_trajectories_for_loss(self, in_cond, time_pts,forward=True,cond=None):
    n_time_pts = time_pts.shape[0]
    
    zt = in_cond.detach().clone().requires_grad_(True)
    batch_size = zt.shape[0]
    trajectories = torch.empty((in_cond.shape[0], n_time_pts, *in_cond.shape[1:]),device=in_cond.device) 
    policies = torch.empty((in_cond.shape[0], n_time_pts, in_cond.shape[1]//2, *in_cond.shape[2:]),device=in_cond.device) 
    cur_score = self.forward_score if forward else self.backward_score
    for i, t in enumerate(time_pts):
      if not forward:
        t = self.T - t
      t_shape = t.expand(batch_size)
      
      bt = self.beta(t)
      policy = (self.gamma * bt)**.5 * cur_score(zt,t_shape,cond) # g * fw_score
      save_idx = i if forward else -(i+1)
      trajectories[:,save_idx] = zt
      policies[:,save_idx] = policy
      
      if i == n_time_pts - 1:
        break
      # First we compute everything, we then take an euler step
      dt = time_pts[i+1] - time_pts[i]
      dt = dt if forward else -dt
      xt,vt = zt.chunk(2,dim=1)
      x_drift = .5 * bt * vt * dt
      if forward:
        v_drift = (.5 * bt * (-xt - self.gamma * vt) + (self.gamma * bt)**.5 * policy) * dt \
            + torch.randn_like(vt) * (self.gamma * bt * dt).abs().sqrt()
      else:
        v_drift = (.5 * bt * (-xt - self.gamma * vt) + self.gamma * bt * self.forward_score(zt,t_shape,cond) \
          - (self.gamma * bt)**.5 * policy) * dt \
          + torch.randn_like(vt) * (self.gamma * bt * dt).abs().sqrt()
          
        
      zt = torch.cat((xt + x_drift,vt + v_drift),dim=1)
    # Forward scores really are the forward policy as described in the FBSDE paper
    return zt,trajectories,policies
  
  def integrate_forward_score(self, t):
    # Curently using Simpsons Method\
    num_pts = 1000
    t_shape = t.flatten().view(-1,1,1).expand(-1,num_pts,-1)
    dt = t/num_pts
    time_pts = torch.arange(num_pts,device=t.device).unsqueeze(-1) * t_shape/num_pts
    multipliers = torch.ones(num_pts, device=t.device)
    multipliers[1:-1:2] = 4
    multipliers[2:-1:2] = 2
    Ats = self.At(time_pts.view(-1,1)) * self.beta(time_pts.view(-1,1))
    Ats = Ats.view(-1,num_pts, *Ats.shape[1:])
    multipliers = multipliers.view(1,-1,*([1] * len(Ats.shape[2:])))
    res = torch.sum(Ats * multipliers,dim=1) * dt.view(-1,*([1] * (len(Ats.shape)-2)))/3
    # res = self.At(t) * self.beta_int(t)
    return res
    
  def diffusion(self, z,t):
    # This was done in an effort to unify the sampling for all the methods
    x,v = torch.chunk(z,2,dim=1)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(v)
    return (self.beta(t) * self.gamma)**.5 * torch.cat((zeros,ones),dim=1)
  
  def get_M_matrix(self, x, t):
    x_dim = np.prod(x.shape)//2 #Divide by 2 because is augmented
    M = torch.zeros((x_dim,2,2))
    at,ct = self.integrate_forward_score(t).chunk(2,dim=1)
    beta_int = self.beta_int(t).expand_as(at).flatten()
    at = at.flatten()
    ct = ct.flatten()

    M[:,0,1] = -beta_int
    M[:,1,0] = beta_int - 2 * self.gamma * at
    M[:,1,1] = self.gamma * beta_int - 2 * self.gamma * ct
  
    M/= -2
    return M , beta_int


  def get_transition_params(self, z, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    M, beta_int = self.get_M_matrix(z, t)
    ch_power = torch.zeros((M.shape[0], 4, 4),device=z.device)
    ch_power[:,:2, :2] = M
    ch_power[:, 2:,2:] = -M.mT
    ch_power[:, 1, 3] = self.gamma * beta_int
    ch_pair = torch.linalg.matrix_exp(ch_power)
    C = ch_pair[:, :2, 2:]
    H_inv = ch_pair[:, :2, :2].mT
    cov = C @ H_inv
    L = torch.linalg.cholesky(cov)
    out_shape = (z.shape[0],z.shape[1]//2,*z.shape[2:],2,2)
    return H_inv.mT.reshape(out_shape), L.reshape(out_shape)
  
  def marginal_prob(self, z, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    H_inv, L = self.get_transition_params(z,t)
    
    x,v = z.chunk(2,dim=1)
    new_x = H_inv[...,0,0] * x + H_inv[...,0,1] * v
    new_v = H_inv[...,1,0] * x + H_inv[...,1,1] * v
    
    return torch.cat((new_x,new_v),dim=1), L

  def prior_sampling(self, shape, device):
    noise = torch.randn(shape, device=device)
    ones = [1] * len(noise.shape[1:])
    t = (torch.ones(1, device=device) * self.T).view(-1,*ones)
    H, L = self.get_transition_params(noise[:1],t)
    L = L.expand((noise.shape[0], *([-1] * len(L.shape[1:]))))
    noise_x, noise_v = torch.chunk(noise,2, dim=1)
    n_noise_x = L[...,0,0] * noise_x
    n_noise_v = L[...,0,1] * noise_x + L[...,1,1] * noise_v
    noise = torch.cat((n_noise_x,n_noise_v),dim=1) 
    # print("COVARIANCE")
    # print(noise.var(dim=0))
    return noise

class CLD(SDE):
  # We assume that images have shape [B, C, H, W] 
  # Additionally there has been added channels as momentum
  def __init__(self,T=1.,delta=1e-3, gamma=2,beta_max=5., backward_model=None):
    # Beta max = 5 here to  match that we didn't divide by 2 as in other set ups
    super().__init__(is_augmented=True)
    self._T = T
    self.delta = delta
    self.gamma = gamma
    self.beta_max = beta_max
    self.backward_score = backward_model

  @property
  def T(self):
    return self._T
  
  def beta(self, t):
    return self.beta_max
    b_min = 0.01
    return (b_min+ t*(self.beta_max-b_min))/2 # self.beta_max
  
  def beta_int(self, t):
    return self.beta_max * t
    b_min = 0.01
    return  (b_min * t +(self.beta_max-b_min) * t**2/2)/2
  
  
  def get_exp_At_components(self, t):
    b = self.beta_int(t)
    exp  = torch.exp(-b) 
    return exp * (b + 1), exp * b, exp * -b, exp * (1-b)
  
  def marginal_prob_mean(self, z, t):
    x, v = torch.chunk(z,2,dim=1) # Decompose in channels    
    a,b,c,d = self.get_exp_At_components(t)
    new_x = a * x + b * v
    new_v = c * x + d * v
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

  def marginal_prob(self, z, t):
    # Returns mean, std
    mean = self.marginal_prob_mean(z,t)
    a,c,d = self.marginal_prob_std(t)
    
    return mean, (a, c, d)
  
  def drift(self, z,t, forward=True,cond=None):
    x,v = torch.chunk(z,2,dim=1)
    beta = self.beta(reshape_t(t,z))
    d_x = beta * v
    d_v = beta * (-x - self.gamma * v)
    if forward:
      return torch.cat((d_x,d_v),dim=1)
    else:
      return torch.cat((d_x, d_v - 2 * beta * self.gamma * self.backward_score(z,t,cond)),dim=1)

  def probability_flow_drift(self, z,t,cond=None):
    x,v = torch.chunk(z,2,dim=1)
    beta = self.beta(reshape_t(t,z))
    d_x =  beta * v
    d_v =  beta * (-x - self.gamma * v)
    return torch.cat((d_x, d_v - self.gamma * beta * self.backward_score(z,t,cond)),dim=1)
  
  def diffusion(self, z,t):
    # This was done in an effort to unify the sampling for all the methods
    x,v = torch.chunk(z,2,dim=1)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(v)
    return (2 * self.beta(t) * self.gamma)**.5 * torch.cat((zeros,ones),dim=1)
  
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)
  
def get_sde(sde_name, **kwargs):
  if sde_name == 'vp':
    return VP(**kwargs)
  elif sde_name == 'edm':
    return EDM(**kwargs)
  elif sde_name == 'vsdm':
    return VSDM(**kwargs)
  elif sde_name == 'cld':
    return CLD(**kwargs)
  elif sde_name == 'linear-momentum-sb':
    return LinearMomentumSchrodingerBridge(**kwargs)