import torch
import numpy as np

# Define sigma(t) mapping
def get_sigma_time(sigma_min, sigma_max):
    def sigma_time(t):
        return sigma_min * (sigma_max / sigma_min) ** t
    return sigma_time

# Define time uniform sampling
def get_sample_time(sampling_eps, T):
    def sample_time(shape):
        return (sampling_eps - T) * torch.rand(shape) + T
    return sample_time



class VESDE():
  def __init__(self, sigma_min, sigma_max, N, T = 1, eps=1e-5):
    super().__init__()
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.N = N
    self.T = T
    self.eps = eps

    self.timesteps = torch.linspace(T, eps, N)

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def sample_time(self, shape):
    return (self.eps - self.T) * torch.rand(shape) + self.T

  def sigma_fn(self, t):
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

  def sde(self, x, t):
    sigma = self.sigma_fn(t)
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def rsde(self, x, t, model_output):
    """Create the drift and diffusion functions for the reverse SDE/ODE."""
    drift, diffusion = self.sde(x, t)
    score = self.score_fn(t, model_output)
    drift = drift - diffusion[:, None, None, None, None] ** 2 * score
    return drift, diffusion

  def score_fn(self, t, model_output):
    return model_output/self.sigma_fn(t)[:,None,None,None,None]

  def update_fn(self, x, t, model_output):
    dt = -self.T / self.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde(x, t, model_output)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


