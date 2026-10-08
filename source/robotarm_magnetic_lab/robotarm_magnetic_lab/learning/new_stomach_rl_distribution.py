"""Bounded continuous command density; no discrete/Beta TASK-010 head."""
import math
import torch
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F


class BoundedNormal:
    def __init__(self, mean, log_std):
        if mean.shape != log_std.shape or mean.shape[-1] not in (9, 36):
            raise ValueError('continuous 9/36 dimensional parameters required')
        if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
            raise RuntimeError('nonfinite continuous parameters')
        self.base = Normal(mean, log_std.exp())
        self.dimension = mean.shape[-1]

    def sample(self):
        latent = self.base.rsample()
        return latent.tanh(), latent

    def mode(self):
        return self.base.loc.tanh()

    def log_prob(self, action, latent=None):
        if not torch.isfinite(action).all() or (action.abs() >= 1).any():
            raise ValueError('finite action strictly inside [-1,1] required')
        if latent is None:
            latent = torch.atanh(action)
        # Exact numerically stable log(1-tanh(z)^2), no epsilon bias.
        jacobian = 2 * (math.log(2.) - latent - F.softplus(-2*latent))
        return (self.base.log_prob(latent) - jacobian).sum(-1)

    def entropy_mc(self):
        action, latent = self.sample()
        return -self.log_prob(action, latent)

    def kl_per_dimension(self, other):
        # The same bijective tanh transform leaves KL unchanged.
        return kl_divergence(self.base, other.base).mean(-1)
