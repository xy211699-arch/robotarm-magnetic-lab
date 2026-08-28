"""Independent area-weighted coverage state for vectorized environments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchedCoverageUpdate:
    updated: torch.Tensor
    visible_count: torch.Tensor
    newly_covered_count: torch.Tensor
    visible_area_m2: torch.Tensor
    newly_covered_area_m2: torch.Tensor
    cumulative_area_m2: torch.Tensor
    coverage_fraction: torch.Tensor


class BatchedCoverageAccumulator:
    """Accumulate one monotonic boolean coverage mask per environment."""

    def __init__(self, weights: torch.Tensor, num_envs: int, device: str) -> None:
        self._weights = weights.to(device=device, dtype=torch.float64).reshape(-1)
        if (
            self._weights.numel() == 0
            or torch.any(~torch.isfinite(self._weights)).item()
            or torch.any(self._weights < 0).item()
            or self._weights.sum().item() <= 0.0
        ):
            raise ValueError(
                "coverage weights must be finite and nonnegative with positive total area"
            )
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        self._mask = torch.zeros(
            (int(num_envs), self._weights.numel()), dtype=torch.bool, device=device
        )
        self._last_frame = torch.full(
            (int(num_envs),), -1, dtype=torch.int64, device=device
        )

    @property
    def mask(self) -> torch.Tensor:
        return self._mask.clone()

    @property
    def last_frame_ids(self) -> torch.Tensor:
        return self._last_frame.clone()

    @property
    def total_area_m2(self) -> torch.Tensor:
        return self._weights.sum()

    @property
    def weights(self) -> torch.Tensor:
        return self._weights.clone()

    def update(
        self, frame_ids: torch.Tensor, visible_mask: torch.Tensor
    ) -> BatchedCoverageUpdate:
        frame_ids = frame_ids.to(
            device=self._mask.device, dtype=torch.int64
        ).reshape(-1)
        visible = visible_mask.to(device=self._mask.device, dtype=torch.bool)
        if visible.shape != self._mask.shape or frame_ids.shape != self._last_frame.shape:
            raise ValueError(
                "coverage frame and visibility shapes must match accumulator state"
            )
        if torch.any(frame_ids < self._last_frame).item():
            raise RuntimeError("coverage frame IDs decreased")
        updated = frame_ids > self._last_frame
        effective_visible = visible & updated[:, None]
        previous = self._mask.clone()
        previous_area = (
            previous.to(torch.float64) * self._weights[None, :]
        ).sum(dim=1)
        self._mask |= effective_visible
        newly = self._mask & ~previous
        self._last_frame = torch.where(updated, frame_ids, self._last_frame)
        visible_count = effective_visible.sum(dim=1)
        newly_count = newly.sum(dim=1)
        visible_area = (
            effective_visible.to(torch.float64) * self._weights[None, :]
        ).sum(dim=1)
        newly_area = (
            newly.to(torch.float64) * self._weights[None, :]
        ).sum(dim=1)
        cumulative_area = (
            self._mask.to(torch.float64) * self._weights[None, :]
        ).sum(dim=1)
        if torch.any(cumulative_area < previous_area).item():
            raise RuntimeError("cumulative coverage area decreased")
        return BatchedCoverageUpdate(
            updated=updated,
            visible_count=visible_count,
            newly_covered_count=newly_count,
            visible_area_m2=visible_area,
            newly_covered_area_m2=newly_area,
            cumulative_area_m2=cumulative_area,
            coverage_fraction=cumulative_area / self._weights.sum(),
        )

    def reset_rows(self, env_ids: torch.Tensor) -> None:
        rows = env_ids.to(device=self._mask.device, dtype=torch.int64).reshape(-1)
        self._mask[rows] = False
        self._last_frame[rows] = -1
