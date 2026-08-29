"""Six-mode categorical and five conditional-Beta joint distribution."""

from __future__ import annotations

import torch
from torch.distributions import Beta, Categorical, kl_divergence
from torch.nn import functional as F


def _finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"TASK-010 {name} is non-finite")


class Task010ModeBetaDistribution:
    def __init__(self, logits: torch.Tensor, concentration_raw: torch.Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != 6:
            raise ValueError("TASK-010 logits must have shape [B,6]")
        if concentration_raw.shape != (logits.shape[0], 5, 2):
            raise ValueError("TASK-010 concentration_raw must have shape [B,5,2]")
        _finite("logits", logits)
        _finite("concentration_raw", concentration_raw)
        self.logits = logits
        self.concentration_raw = concentration_raw
        self.concentration = 1.0 + F.softplus(concentration_raw)
        self.categorical = Categorical(logits=logits)
        self.beta = Beta(self.concentration[..., 0], self.concentration[..., 1])

    def _assemble(self, modes: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
        strengths = torch.where(modes == 0, torch.zeros_like(strengths), strengths.clamp(0.0, 1.0))
        result = torch.stack((modes.to(dtype=self.logits.dtype), strengths), dim=1)
        _finite("action", result)
        return result

    def sample(self) -> torch.Tensor:
        modes = self.categorical.sample()
        all_strengths = self.beta.sample()
        index = (modes - 1).clamp(0, 4).unsqueeze(1)
        strengths = all_strengths.gather(1, index).squeeze(1)
        return self._assemble(modes, strengths)

    def mode(self) -> torch.Tensor:
        modes = torch.argmax(self.logits, dim=1)
        means = self.concentration[..., 0] / self.concentration.sum(dim=-1)
        strengths = means.gather(1, (modes - 1).clamp(0, 4).unsqueeze(1)).squeeze(1)
        return self._assemble(modes, strengths)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape != (self.logits.shape[0], 2):
            raise ValueError("TASK-010 actions must have shape [B,2]")
        modes = actions[:, 0].to(dtype=torch.int64)
        if torch.any((modes < 0) | (modes > 5)).item():
            raise ValueError("TASK-010 action mode is outside [0,5]")
        mode_log_prob = self.categorical.log_prob(modes)
        epsilon = torch.finfo(actions.dtype).eps
        strength = actions[:, 1].clamp(epsilon, 1.0 - epsilon)
        all_log_prob = self.beta.log_prob(strength[:, None].expand(-1, 5))
        selected = all_log_prob.gather(1, (modes - 1).clamp(0, 4).unsqueeze(1)).squeeze(1)
        result = mode_log_prob + torch.where(modes == 0, torch.zeros_like(selected), selected)
        _finite("joint log probability", result)
        return result.unsqueeze(1)

    def entropy(self) -> torch.Tensor:
        result = self.categorical.entropy() + (
            self.categorical.probs[:, 1:] * self.beta.entropy()
        ).sum(dim=1)
        _finite("joint entropy", result)
        return result.unsqueeze(1)

    def kl(self, other: "Task010ModeBetaDistribution") -> torch.Tensor:
        if not isinstance(other, Task010ModeBetaDistribution):
            raise TypeError("TASK-010 KL requires another Task010ModeBetaDistribution")
        result = kl_divergence(self.categorical, other.categorical) + (
            self.categorical.probs[:, 1:] * kl_divergence(self.beta, other.beta)
        ).sum(dim=1)
        _finite("joint KL", result)
        return result.unsqueeze(1)

    def parameters_for_storage(self) -> dict[str, torch.Tensor]:
        return {
            "logits": self.logits.detach().clone(),
            "concentration_raw": self.concentration_raw.detach().clone(),
            "concentration": self.concentration.detach().clone(),
        }
