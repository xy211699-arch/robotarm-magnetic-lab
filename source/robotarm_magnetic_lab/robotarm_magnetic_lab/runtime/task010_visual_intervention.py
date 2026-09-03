"""Pure tensor interventions for TASK-010 visual-dependence conditions."""

from __future__ import annotations

import torch


VALID_TRAINING_VISUAL_CONDITIONS = ("normal", "blind")
VALID_EVALUATION_VISUAL_CONDITIONS = ("normal", "blind", "donor", "first_frame")


def replace_actor_visual_features(
    actor_observation: torch.Tensor,
    replacement: torch.Tensor,
) -> torch.Tensor:
    if actor_observation.ndim != 2 or actor_observation.shape[1] != 519:
        raise ValueError("actor observation must have shape [B,519]")
    if replacement.ndim != 2 or replacement.shape[0] != actor_observation.shape[0] or replacement.shape[1] != 512:
        raise ValueError("visual replacement must have shape [B,512]")
    if not (
        torch.isfinite(actor_observation).all()
        and torch.isfinite(replacement).all()
    ).item():
        raise ValueError("visual intervention inputs must be finite")
    output = actor_observation.clone()
    output[:, :512] = replacement.to(
        device=actor_observation.device,
        dtype=actor_observation.dtype,
    )
    return output


class Task010VisualIntervention:
    """Apply one deterministic visual condition without mutating encoder caches."""

    def __init__(
        self,
        condition: str,
        *,
        num_envs: int,
        feature_dim: int = 512,
    ) -> None:
        self.condition = str(condition)
        if self.condition not in (
            VALID_TRAINING_VISUAL_CONDITIONS + VALID_EVALUATION_VISUAL_CONDITIONS
        ):
            raise ValueError(f"unknown visual condition: {self.condition}")
        self.num_envs = int(num_envs)
        self.feature_dim = int(feature_dim)
        if self.num_envs <= 0 or self.feature_dim != 512:
            raise ValueError("num_envs must be positive and feature_dim must be 512")
        self._first_features: torch.Tensor | None = None
        self._initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device="cpu"
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._first_features is None:
            return
        if env_ids is None:
            self._initialized.fill_(False)
            self._first_features.zero_()
            return
        rows = env_ids.to(device=self._initialized.device, dtype=torch.int64).reshape(-1)
        self._initialized[rows] = False
        self._first_features[rows] = 0.0

    def _ensure_state(self, features: torch.Tensor) -> None:
        if self._first_features is None or self._first_features.shape != features.shape:
            self._first_features = torch.zeros_like(features)
            self._initialized = torch.zeros(
                features.shape[0], dtype=torch.bool, device=features.device
            )
        elif self._first_features.device != features.device:
            self._first_features = self._first_features.to(features.device)
            self._initialized = self._initialized.to(features.device)

    def apply(
        self,
        features: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
        donor_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del env_ids
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("visual features must have shape [B,512]")
        if self.condition == "normal":
            return features.clone()
        if self.condition == "blind":
            return torch.zeros_like(features)
        if self.condition == "donor":
            if donor_features is None:
                raise ValueError("donor condition requires donor_features")
            if donor_features.shape != features.shape:
                raise ValueError("donor_features must have shape [B,512]")
            return donor_features.clone()
        if self.condition == "first_frame":
            self._ensure_state(features)
            assert self._first_features is not None
            if not torch.isfinite(features).all().item():
                raise ValueError("first-frame visual features must be finite")
            output = self._first_features.clone()
            rows = torch.nonzero(~self._initialized, as_tuple=False).reshape(-1)
            if rows.numel():
                output[rows] = features[rows]
                self._first_features[rows] = features[rows]
                self._initialized[rows] = True
            return output
        raise AssertionError(f"unhandled visual condition: {self.condition}")
