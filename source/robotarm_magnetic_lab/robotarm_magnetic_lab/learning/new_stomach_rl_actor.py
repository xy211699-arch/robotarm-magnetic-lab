"""One explicit-state recurrent trunk for A/B/C/D; only 548 allowed features."""
import math
import torch
from torch import nn
from .new_stomach_rl_distribution import BoundedNormal


class ContinuousGRUActor(nn.Module):
    observation_dim, hidden_dim = 548, 256

    def __init__(self, action_dim):
        super().__init__()
        if action_dim not in (9,36):
            raise ValueError('9 or 36 action dimensions required')
        self.action_dim = action_dim
        self.visual = nn.Sequential(nn.Linear(512,256), nn.LayerNorm(256), nn.SiLU())
        self.history = nn.Sequential(nn.Linear(36,32), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(288,256), nn.LayerNorm(256), nn.SiLU())
        self.gru = nn.GRU(256,256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2.))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mean_head.weight, .01)
        for name, parameter in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(parameter)
            elif 'weight_hh' in name:
                for block in parameter.chunk(3):
                    nn.init.orthogonal_(block)
            else:
                nn.init.zeros_(parameter)

    def parameters_sequence(self, observations, hidden, masks=None):
        if observations.ndim not in (2,3) or observations.shape[-1] != 548:
            raise ValueError('only [N,548] or [T,N,548] Actor input allowed')
        if not torch.isfinite(observations).all():
            raise RuntimeError('nonfinite Actor inputs')
        single = observations.ndim == 2
        if single:
            observations = observations[None]
        t,n,_ = observations.shape
        if hidden is None:
            hidden = observations.new_zeros((1,n,256))
        if hidden.shape != (1,n,256):
            raise ValueError('recurrent hidden must match environment rows')
        if masks is None:
            masks = torch.zeros((t,n), dtype=torch.bool, device=observations.device)
        if masks.shape != (t,n):
            raise ValueError('reset-before-step masks must be [T,N]')
        encoded = self.fusion(torch.cat((self.visual(observations[...,:512]),
                                        self.history(observations[...,512:])), -1))
        outputs = []
        for index in range(t):
            hidden = hidden * (~masks[index]).to(encoded.dtype)[None,:,None]
            value, hidden = self.gru(encoded[index:index+1], hidden)
            outputs.append(value)
        mean = self.mean_head(torch.cat(outputs))
        return (mean[0] if single else mean), hidden

    def distribution(self, observations, hidden, masks=None):
        mean, hidden = self.parameters_sequence(observations, hidden, masks)
        return BoundedNormal(mean, self.log_std.clamp(-5.,2.).expand_as(mean)), hidden
