"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch

from utils.misc import batch_matrix_product

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
  
  def sample(self, shape, device, backward=True, 
             in_cond=None, prob_flow=True, 
             cond=None, return_traj=False):
    with torch.no_grad():
      xt = self.prior_sampling(shape,device) if backward else in_cond
      assert xt is not None
      n_time_pts = 100      
      # step_indices = torch.arange(n_time_pts, device=device)
      # rho = 7
      # time_pts = (self.T ** (1 / rho) + step_indices / (n_time_pts - 1) * (self.delta ** (1 / rho) - self.T ** (1 / rho))) ** rho
      # time_pts = torch.cat([time_pts, torch.zeros_like(time_pts[:1])]) # t_N = 0
      # time_pts = time_pts.flip(dims=(0,))

      time_pts = torch.linspace(self.delta, self.T, n_time_pts, device=device)
      if return_traj:
        trajectories = torch.empty((xt.shape[0], n_time_pts, *xt.shape[1:]),device=xt.device) 

      for i, t in enumerate(time_pts):
        if return_traj:
          trajectories[:,i] = xt
        if i == n_time_pts - 1:
          break
        dt = time_pts[i+1] - t 
        dt = -dt if backward else dt 
        t_shape = self.T - t if backward else t
        t_shape = t_shape.unsqueeze(-1).expand(xt.shape[0],1)
        if prob_flow:
          drift = self.probability_flow_drift(xt,t_shape, cond)
          if backward and i+1 != n_time_pts - 1:
            xt_hat = xt + drift * dt
            t_hat = self.T - time_pts[i+1] if backward else time_pts[i+1]
            t_hat = t_hat.unsqueeze(-1).expand(xt.shape[0],1)
            xt = xt + .5 * dt * (drift + self.probability_flow_drift(xt_hat,t_hat,cond))
          else:
            xt = xt + drift * dt
        else:
          drift = self.drift(xt,t_shape, forward=(not backward),cond=cond)
          xt = xt + drift * dt + torch.randn_like(xt) * self.diffusion(xt,t) * dt.abs().sqrt()
      return xt, (trajectories if return_traj else None)
class LinearSDE(SDE):
  
  def __init__(self, backward_score, is_augmented):
    SDE.__init__(self, is_augmented)
    self.backward_score = backward_score
    
  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """ Returns the marginal prob dist """
    pass  
class VP(LinearSDE):

  def __init__(self,T=1.,delta=1e-3, beta_max=10, model_backward=None):
    LinearSDE.__init__(self,backward_score=model_backward,is_augmented=False)
    self._T = T
    self.delta = delta
    self.beta_max = beta_max

  @property
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
  
  def drift(self, x,t, forward=True,cond=None):
    beta = self.beta(t)
    if forward:
      return -.5 * beta * x
    else:
      return -.5 * beta * x - beta * self.backward_score(x,t,cond)
  
  def probability_flow_drift(self, xt, t,cond=None):
    beta = self.beta(t)
    return -.5 * beta * (xt + self.backward_score(xt,t,cond))
  
  def diffusion(self, x,t):
    return self.beta(t)**.5

  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)

class EDM(LinearSDE):

  def __init__(self,T=80.,delta=1e-3, model_backward=None):
    super().__init__(backward_score=model_backward, is_augmented=False)
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

class SchrodingerBridge(SDE):
  """ 
    Note that this is not a general SB, it is implemented so that after optimized
    the linear drift transports to a standard normal, it also only works for f(Xt,t) = -.5 bt Xt
  """
  def __init__(self, T=1.,delta=1e-3, beta_max=10, forward_score=None, backward_score=None, is_augmented=False):
    SDE.__init__(self,is_augmented=is_augmented)
    self._T = T
    self.delta = delta
    self.beta_max = beta_max
    self.forward_score = forward_score
    self.backward_score = backward_score

  @property
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
   
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)

  def probability_flow_drift(self, xt, t):
    beta = self.beta(t)
    return -.5 * beta * (xt - self.forward_score(xt,t) \
      + self.backward_score(xt, t))

  def get_trajectories_for_loss(self, in_cond, time_pts,forward=True):
    n_time_pts = time_pts.shape[0]
    
    xt = in_cond.detach().clone().requires_grad_(True)
    batch_size = xt.shape[0]
    trajectories = torch.empty((batch_size, n_time_pts, *in_cond.shape[1:]),device=in_cond.device) 
    policies = torch.empty_like(trajectories)
    cur_score = self.forward_score if forward else self.backward_score
    for i, t in enumerate(time_pts):
      if not forward:
        t = self.T - t
      t_shape = t.unsqueeze(-1).expand(batch_size,1)
      
      bt = self.beta(t)
      save_idx = i if forward else -(i+1)
      policy = bt**.5 * cur_score(xt,t_shape) # g * fw_score
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
        xt = xt + (-.5 * bt * xt - bt**.5 * policy) * dt + torch.randn_like(xt) * self.diffusion(xt,t) * dt.abs().sqrt()
        
    return xt,trajectories,policies
    
class MomentumSchrodingerBridge(SchrodingerBridge):
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
    SchrodingerBridge.__init__(self, T, delta, beta_max, forward_model, backward_model, is_augmented=True)
    self.gamma = gamma

  @property
  def T(self):
    return self._T
  
  def drift(self,z,t, forward=True):
    beta = self.beta(t)
    xt,vt = z.chunk(2,dim=1)
    if forward:
      v_drift = -xt - self.gamma * vt + 2 * self.gamma * self.forward_score(z,t)
      return .5 * beta * torch.cat((vt, v_drift),dim=1)
    else:
      v_drift = -xt - self.gamma * vt - 2 * self.gamma * self.backward_score(z,t)
      return .5 * beta * torch.cat((vt,v_drift),dim=1)
    
  def diffusion(self, z,t):
    # This was done in an effort to unify the sampling for all the methods
    x,v = torch.chunk(z,2,dim=1)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(v)
    return (self.beta(t) * self.gamma)**.5 * torch.cat((zeros,ones),dim=1)
  
  
  def get_trajectories_for_loss(self, in_cond, time_pts,forward=True):
    n_time_pts = time_pts.shape[0]
    
    zt = in_cond.detach().clone().requires_grad_(True)
    batch_size = zt.shape[0]
    trajectories = torch.empty((in_cond.shape[0], n_time_pts, *in_cond.shape[1:]),device=in_cond.device) 
    policies = torch.empty((in_cond.shape[0], n_time_pts, in_cond.shape[1]//2, *in_cond.shape[2:]),device=in_cond.device) 
    cur_score = self.forward_score if forward else self.backward_score
    for i, t in enumerate(time_pts):
      if not forward:
        t = self.T - t
      t_shape = t.unsqueeze(-1).expand(batch_size,1)
      
      bt = self.beta(t)
      policy = (self.gamma * bt)**.5 * cur_score(zt,t_shape) # g * fw_score
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
        v_drift = (.5 * bt * (-xt - self.gamma * vt) - (self.gamma * bt)**.5 * policy) * dt \
          + torch.randn_like(vt) * (self.gamma * bt * dt).abs().sqrt()
      zt = torch.cat((xt + x_drift,vt + v_drift),dim=1)
    # Forward scores really are the forward policy as described in the FBSDE paper
    return zt,trajectories,policies


class GeneralLinearizedSB(SchrodingerBridge, LinearSDE):
  """ 
    Note that this is not a general SB, it is implemented so that after optimized
    the linear drift transports to a standard normal
    To instantiate a class you just have to implement how the D matrix works
  """
  def __init__(self,T=1.,delta=1e-3, beta_max=10, forward_model=None, backward_model=None, is_augmented=False):
    """ Here the backward model is a standard backwards score
        The forward model is such that it receives t of shape [bs,1] and outputs a matrix [bs, d,d]
        The dimension is infered from the forward model, so if it doesn't behave in this way it won't work
        We internally assign the forward model to be the multiplication against this matrix
    """
    SchrodingerBridge.__init__(self,T,delta,beta_max,is_augmented=is_augmented)
    LinearSDE.__init__(self,backward_score=backward_model, is_augmented=is_augmented)

    self.At = forward_model

  @property
  def forward_score(self):
    return lambda x,t : batch_matrix_product(self.At(t), x)
  
  @forward_score.setter
  def forward_score(self,forward_model):
    self.At = forward_model
  
  @abc.abstractmethod
  def D(self,t):
    pass

  @property
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
    betas = self.beta(time_pts)#.unsqueeze(-1)
    return torch.sum(betas * Ats * multipliers,dim=1) * dt.unsqueeze(-1)/3

  def compute_variance(self, t):
    int_mat = self.int_beta_ds(t)
    dim = int_mat.shape[-1]
    ch_power = torch.zeros((t.shape[0], 2 * dim, 2 * dim),device=int_mat.device)
    ch_power[:,:dim, :dim] = -.5 * int_mat
    ch_power[:,dim:, dim:] = .5 * int_mat.mH
    if self.is_augmented:
      k = dim//2
      ch_power[:, k:dim, dim+k:] = self.gamma * self.beta_int(t).view(-1,1,1) * torch.eye(k,device=int_mat.device).unsqueeze(0).expand(t.shape[0],-1,-1)
    else:
      ch_power[:, :dim, dim:] = self.beta_int(t).view(-1,1,1) * torch.eye(dim,device=int_mat.device).unsqueeze(0).expand(t.shape[0],-1,-1)
    ch_pair = torch.linalg.matrix_exp(ch_power)
    C = ch_pair[:, :dim, dim:]
    H_inv = ch_pair[:, :dim, :dim].mH
    cov = C @ H_inv
    L = torch.linalg.cholesky(cov)
    return cov, L, H_inv.mH # Cov, L, exp([-.5 bD]_t)
  
  def marginal_prob(self, x, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    cov, L, big_beta = self.compute_variance(t)
    return batch_matrix_product(big_beta, x), L
  
   
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)
    L = self.compute_variance(torch.tensor([[self.T]],device=device))[1][0]
    return (L @ torch.randn(*shape, dtype=torch.float, device=device).T).T


class LinearSchrodingerBridge(GeneralLinearizedSB):
  """ 
    Note that this is not a general SB, it is implemented so that after optimized
    the linear drift transports to a standard normal
  """
  def __init__(self,T=1.,delta=1e-3, beta_max=10, forward_model=None, backward_model=None):
    """ Here the backward model is a standard backwards score
        The forward model is such that it receives t of shape [bs,1] and outputs a matrix [bs, d,d]
        The dimension is infered from the forward model, so if it doesn't behave in this way it won't work
        We internally assign the forward model to be the multiplication against this matrix
    """
    GeneralLinearizedSB.__init__(self, T, delta, beta_max, forward_model, backward_model, is_augmented=False)

  @property
  def forward_score(self):
    return lambda x,t : batch_matrix_product(self.At(t), x)
  
  @forward_score.setter
  def forward_score(self,forward_model):
    self.At = forward_model
  
  def D(self,t):
    mat = self.At(t) 
    id = torch.eye(mat.shape[-1], device=mat.device).unsqueeze(0)
    return id - 2 * (mat + mat.mT)

class LinearMomentumSchrodingerBridge(MomentumSchrodingerBridge, GeneralLinearizedSB):
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
    MomentumSchrodingerBridge.__init__(self,T,delta,gamma,beta_max,forward_model,backward_model)
    GeneralLinearizedSB.__init__(self,T,delta,beta_max,forward_model,backward_model,is_augmented=True)

  @property
  def forward_score(self):
    return lambda x,t : batch_matrix_product(self.At(t), x)
  
  @forward_score.setter
  def forward_score(self,forward_model):
    self.At = forward_model
  
  def D(self,t):
    mat = self.At(t) # Has shape [bs, d, 2d]
    dim = mat.shape[-2]
    Dt = torch.cat((torch.zeros_like(mat),- 2 * self.gamma * mat),dim=-2) # [bs,2d,2d]
    id = torch.eye(dim, device=mat.device).unsqueeze(0)
    Dt[:,:dim, -dim:] -= id
    Dt[:,-dim:, :dim] += id
    Dt[:,-dim:, -dim:] += self.gamma * id
    return Dt

  @property
  def T(self):
    return self._T
  
  def drift(self,z,t, forward=True):
    beta = self.beta(t)
    if forward:
      return -.5 * beta * batch_matrix_product(self.D(t) , z)
    else:
      xt,vt = z.chunk(2,dim=1)
      d_x = .5 * beta * vt
      d_v = .5 * beta * (-xt - self.gamma * vt) - self.gamma * beta * self.backward_score(z,t)
      return torch.cat((d_x,d_v),dim=1)
    
  def diffusion(self, z,t):
    # This was done in an effort to unify the sampling for all the methods
    x,v = torch.chunk(z,2,dim=1)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(v)
    return (self.beta(t) * self.gamma)**.5 * torch.cat((zeros,ones),dim=1)
  
  

class CLD(SDE):
  # We assume that images have shape [B, C, H, W] 
  # Additionally there has been added channels as momentum
  def __init__(self,T=1.,delta=1e-3, gamma=2,beta_max=5., model_backward=None):
    super().__init__(is_augmented=True)
    self._T = T
    self.delta = delta
    self.gamma = gamma
    self.beta_max = beta_max
    self.backward_score = model_backward

  @property
  def T(self):
    return self._T
  
  def beta(self, t):
    return self.beta_max 
  
  def beta_int(self, t):
    return self.beta_max  * t
  
  
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
    beta = self.beta(t)
    d_x = beta * v
    d_v = beta * (-x - self.gamma * v)
    if forward:
      return torch.cat((d_x,d_v),dim=-1)
    else:
      return torch.cat((d_x, d_v - 2 * beta * self.gamma * self.backward_score(z,t,cond)),dim=1)

  def probability_flow_drift(self, z,t,cond=None):
    x,v = torch.chunk(z,2,dim=-1)
    beta = self.beta(t)
    d_x =  beta * v
    d_v =  beta * (-x - self.gamma * v)
    return torch.cat((d_x, d_v - self.gamma * beta * self.backward_score(z,t,cond)),dim=-1)
  
  def diffusion(self, z,t):
    # This was done in an effort to unify the sampling for all the methods
    x,v = torch.chunk(z,2,dim=1)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(v)
    return (2 * self.beta(t) * self.gamma)**.5 * torch.cat((zeros,ones),dim=1)
  
  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)
  
def get_sde(sde_name):
  if sde_name == 'vp':
    return VP()
  elif sde_name == 'edm':
    return EDM()
  elif sde_name == 'sb':
    return SchrodingerBridge()
  elif sde_name == 'linear-sb':
    return LinearSchrodingerBridge()
  elif sde_name == 'cld':
    return CLD()
  elif sde_name == 'momentum-sb':
    return MomentumSchrodingerBridge()
  elif sde_name == 'linear-momentum-sb':
    return LinearMomentumSchrodingerBridge()