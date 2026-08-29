"""Independent asymmetric Critic and selective running normalization."""

from __future__ import annotations

import math

import torch
from torch import nn


def task010_continuous_mask(device=None) -> torch.Tensor:
    mask = torch.zeros(65, dtype=torch.bool, device=device)
    for start, stop in ((0, 15), (21, 22), (23, 27), (28, 58), (62, 65)):
        mask[start:stop] = True
    return mask


class Task010SelectiveNormalizer(nn.Module):
    def __init__(self, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("continuous_mask", task010_continuous_mask())
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("running_mean", torch.zeros(65, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(65, dtype=torch.float64))
        self.register_buffer("frozen", torch.tensor(False, dtype=torch.bool))

    def freeze(self) -> None:
        self.frozen.fill_(True)

    def unfreeze(self) -> None:
        self.frozen.fill_(False)

    def _update(self, value: torch.Tensor) -> None:
        batch = value.detach().to(dtype=torch.float64)
        batch_count = float(batch.shape[0])
        if batch_count <= 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        old_count = float(self.count.item())
        new_count = old_count + batch_count
        if old_count == 0.0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var.clamp_min(self.epsilon))
        else:
            delta = batch_mean - self.running_mean
            combined_mean = self.running_mean + delta * (batch_count / new_count)
            old_m2 = self.running_var * old_count
            batch_m2 = batch_var * batch_count
            combined_m2 = old_m2 + batch_m2 + delta.square() * old_count * batch_count / new_count
            self.running_mean.copy_(combined_mean)
            self.running_var.copy_((combined_m2 / new_count).clamp_min(self.epsilon))
        self.count.fill_(new_count)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != 65:
            raise ValueError("TASK-010 Critic normalizer expected 65 features")
        flat = observation.reshape(-1, 65)
        if self.training and not bool(self.frozen.item()):
            self._update(flat)
        output = observation.clone()
        mask = self.continuous_mask
        mean = self.running_mean[mask].to(device=observation.device, dtype=observation.dtype)
        std = torch.sqrt(self.running_var[mask].to(device=observation.device, dtype=observation.dtype).clamp_min(self.epsilon))
        output[..., mask] = (observation[..., mask] - mean) / std
        if not torch.isfinite(output).all().item():
            raise RuntimeError("TASK-010 normalized Critic observation is non-finite")
        return output


class Task010Critic(nn.Module):
    observation_dim = 65

    def __init__(self) -> None:
        super().__init__()
        self.normalizer = Task010SelectiveNormalizer()
        self.network = nn.Sequential(
            nn.Linear(65, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1),
        )
        linear = [module for module in self.network if isinstance(module, nn.Linear)]
        for layer in linear[:-1]:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linear[-1].weight, gain=1.0)
        nn.init.zeros_(linear[-1].bias)

    def freeze_normalizer(self) -> None:
        self.normalizer.freeze()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(f"TASK-010 Critic expected 65 features, got {observation.shape[-1]}")
        value = self.network(self.normalizer(observation))
        if not torch.isfinite(value).all().item():
            raise RuntimeError("TASK-010 Critic value is non-finite")
        return value
