"""Pure-visual recurrent TASK-010 Actor."""

from __future__ import annotations

import math

import torch
from torch import nn

from .task010_distribution import Task010ModeBetaDistribution


def _orthogonal(layer: nn.Linear, gain: float, bias: float = 0.0) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)


class Task010Actor(nn.Module):
    observation_dim = 519
    hidden_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.visual_projection = nn.Linear(512, 256)
        self.visual_norm = nn.LayerNorm(256)
        self.action_projection = nn.Linear(7, 32)
        self.fusion = nn.Linear(288, 256)
        self.fusion_norm = nn.LayerNorm(256)
        self.activation = nn.SiLU()
        self.gru = nn.GRU(256, 256, num_layers=1)
        self.mode_head = nn.Linear(256, 6)
        self.beta_head = nn.Linear(256, 10)
        gain = math.sqrt(2.0)
        _orthogonal(self.visual_projection, gain)
        _orthogonal(self.action_projection, gain)
        _orthogonal(self.fusion, gain)
        _orthogonal(self.mode_head, 0.01)
        beta_bias = math.log(math.expm1(1.0))
        _orthogonal(self.beta_head, 0.01, beta_bias)
        for name, parameter in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                for block in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(block)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        self._hidden_state: torch.Tensor | None = None
        self._last_distribution: Task010ModeBetaDistribution | None = None

    def _features(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(f"TASK-010 Actor expected 519 features, got {observations.shape[-1]}")
        if not torch.isfinite(observations).all().item():
            raise RuntimeError("TASK-010 Actor observation is non-finite")
        visual = self.activation(self.visual_norm(self.visual_projection(observations[..., :512])))
        action = self.activation(self.action_projection(observations[..., 512:]))
        return self.activation(self.fusion_norm(self.fusion(torch.cat((visual, action), dim=-1))))

    def evaluate_parameters(
        self,
        observations: torch.Tensor,
        *,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = observations.ndim == 3
        if observations.ndim not in (2, 3):
            raise ValueError("TASK-010 Actor observations must be [B,519] or [T,B,519]")
        features = self._features(observations)
        if not sequence:
            features = features.unsqueeze(0)
        batch = features.shape[1]
        if hidden_state is None:
            hidden_state = self._hidden_state
        if hidden_state is None or hidden_state.shape != (1, batch, self.hidden_dim):
            hidden_state = torch.zeros((1, batch, self.hidden_dim), device=features.device, dtype=features.dtype)
        if masks is None:
            output, next_hidden = self.gru(features, hidden_state)
        else:
            masks = masks.to(device=features.device, dtype=torch.bool)
            if not sequence:
                masks = masks.reshape(1, batch)
            if masks.shape != features.shape[:2]:
                raise ValueError("TASK-010 recurrent masks must have shape [T,B]")
            outputs = []
            next_hidden = hidden_state
            for index in range(features.shape[0]):
                next_hidden = next_hidden * (~masks[index]).to(features.dtype).view(1, batch, 1)
                step, next_hidden = self.gru(features[index : index + 1], next_hidden)
                outputs.append(step)
            output = torch.cat(outputs, dim=0)
        logits = self.mode_head(output)
        raw = self.beta_head(output).reshape(*output.shape[:-1], 5, 2)
        if not sequence:
            logits = logits.squeeze(0)
            raw = raw.squeeze(0)
            self._hidden_state = next_hidden
        if not (torch.isfinite(logits).all() and torch.isfinite(raw).all()).item():
            raise RuntimeError("TASK-010 Actor output is non-finite")
        return logits, raw, next_hidden

    def forward(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        logits, raw, _ = self.evaluate_parameters(observations, masks=masks, hidden_state=hidden_state)
        if logits.ndim != 2:
            raise ValueError("TASK-010 action sampling expects one [B,519] boundary")
        self._last_distribution = Task010ModeBetaDistribution(logits, raw)
        return self._last_distribution.sample() if stochastic_output else self._last_distribution.mode()

    def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._last_distribution is None:
            raise RuntimeError("TASK-010 Actor has no current distribution")
        return self._last_distribution.log_prob(actions)

    def output_distribution_params(self) -> dict[str, torch.Tensor]:
        if self._last_distribution is None:
            raise RuntimeError("TASK-010 Actor has no current distribution")
        return self._last_distribution.parameters_for_storage()

    def output_entropy(self) -> torch.Tensor:
        if self._last_distribution is None:
            raise RuntimeError("TASK-010 Actor has no current distribution")
        return self._last_distribution.entropy()

    def get_kl_divergence(self, old_parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._last_distribution is None:
            raise RuntimeError("TASK-010 Actor has no current distribution")
        old = Task010ModeBetaDistribution(old_parameters["logits"], old_parameters["concentration_raw"])
        return old.kl(self._last_distribution)

    def get_hidden_state(self) -> torch.Tensor:
        if self._hidden_state is None:
            return torch.zeros((1, 0, self.hidden_dim), device=self.mode_head.weight.device)
        return self._hidden_state

    def detach_hidden_state(self) -> None:
        if self._hidden_state is not None:
            self._hidden_state = self._hidden_state.detach()

    def reset(self, dones: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None) -> None:
        state = self._hidden_state if hidden_state is None else hidden_state
        if state is None:
            return
        if dones is None:
            state.zero_()
        else:
            rows = dones.to(device=state.device, dtype=torch.bool).reshape(-1)
            if rows.shape[0] != state.shape[1]:
                raise ValueError("TASK-010 reset mask batch mismatch")
            state[:, rows] = 0.0
        if hidden_state is None:
            self._hidden_state = state
