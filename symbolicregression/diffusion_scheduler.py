import torch
import torch.nn.functional as F
import math
import sys

BETA_START = 0.0001
BETA_END = 0.02

def linear_beta_schedule(timesteps, beta_start=BETA_START, beta_end=BETA_END):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)




class LatentDiffusion:
    """Lightweight latent (Gaussian) diffusion scheduler for z vectors.
    Provides q_sample and helper scalars (sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).
    """
    def __init__(self, num_timesteps=100, latent_dim=None, device='cuda', schedule_type='linear'):
        self.num_timesteps = num_timesteps
        self.latent_dim = latent_dim
        self.device = device

        if schedule_type == 'linear':
            betas = linear_beta_schedule(num_timesteps).to(device)
        elif schedule_type == 'cosine':
            betas = cosine_beta_schedule(num_timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.betas = betas.float()
        self.alphas = (1.0 - self.betas).float()
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

    def q_sample(self, z0, t, noise=None):
        """z0: (B, D), t: (B,) long
        returns z_t and noise used
        """
        device = z0.device
        B = z0.size(0)
        if t.dim() == 0:
            t = t.view(1).expand(B)
        t = t.to(device)

        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)  # (B,1)
        b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)

        if noise is None:
            noise = torch.randn_like(z0, device=device)

        z_t = a * z0 + b * noise
        return z_t, noise

    def get_scalars(self, t):
        if t.dim() == 0:
            t = t.view(1).expand(1)
        return self.sqrt_alphas_cumprod[t], self.sqrt_one_minus_alphas_cumprod[t]
