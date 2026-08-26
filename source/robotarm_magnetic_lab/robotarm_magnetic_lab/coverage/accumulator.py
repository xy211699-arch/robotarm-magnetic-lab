"""One-update-per-recorded-frame monotonic coverage accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np


@dataclass(frozen=True)
class CoverageUpdate:
    frame_id: Hashable
    updated: bool
    visible_count: int
    newly_covered_count: int
    cumulative_count: int
    coverage_fraction: float
    visible_area_m2: float
    newly_covered_area_m2: float
    cumulative_area_m2: float
    total_area_m2: float


class CoverageAccumulator:
    def __init__(self, vertex_count: int, vertex_weights: np.ndarray | None = None) -> None:
        if int(vertex_count) <= 0:
            raise ValueError("vertex_count must be positive")
        self._mask = np.zeros(int(vertex_count), dtype=np.bool_)
        if vertex_weights is None:
            weights = np.ones(int(vertex_count), dtype=np.float64)
        else:
            weights = np.asarray(vertex_weights, dtype=np.float64).reshape(-1)
        if len(weights) != int(vertex_count):
            raise ValueError("vertex weight count must equal vertex_count")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("vertex weights must be finite and nonnegative")
        if float(weights.sum()) <= 0.0:
            raise ValueError("vertex weights must have positive total area")
        self._weights = weights.copy()
        self._frame_ids: set[Hashable] = set()

    @property
    def mask(self) -> np.ndarray:
        result = self._mask.copy()
        result.setflags(write=False)
        return result

    @property
    def recorded_frame_count(self) -> int:
        return len(self._frame_ids)

    @property
    def vertex_weights(self) -> np.ndarray:
        result = self._weights.copy()
        result.setflags(write=False)
        return result

    @property
    def total_area_m2(self) -> float:
        return float(self._weights.sum())

    @property
    def cumulative_area_m2(self) -> float:
        return float(self._weights[self._mask].sum())

    @property
    def coverage_fraction(self) -> float:
        return self.cumulative_area_m2 / self.total_area_m2

    def update(self, frame_id: Hashable, visible_vertex_indices: Iterable[int]) -> CoverageUpdate:
        if frame_id in self._frame_ids:
            cumulative = int(self._mask.sum())
            cumulative_area = self.cumulative_area_m2
            return CoverageUpdate(
                frame_id,
                False,
                0,
                0,
                cumulative,
                cumulative_area / self.total_area_m2,
                0.0,
                0.0,
                cumulative_area,
                self.total_area_m2,
            )
        indices = np.asarray(list(visible_vertex_indices), dtype=np.int64).reshape(-1)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= len(self._mask)):
            raise ValueError("visible vertex index is out of range")
        indices = np.unique(indices)
        previous = self._mask.copy()
        self._mask[indices] = True
        self._frame_ids.add(frame_id)
        newly_mask = self._mask & ~previous
        newly = int(np.count_nonzero(newly_mask))
        cumulative = int(self._mask.sum())
        visible_area = float(self._weights[indices].sum())
        newly_area = float(self._weights[newly_mask].sum())
        cumulative_area = self.cumulative_area_m2
        return CoverageUpdate(
            frame_id=frame_id,
            updated=True,
            visible_count=int(len(indices)),
            newly_covered_count=newly,
            cumulative_count=cumulative,
            coverage_fraction=cumulative_area / self.total_area_m2,
            visible_area_m2=visible_area,
            newly_covered_area_m2=newly_area,
            cumulative_area_m2=cumulative_area,
            total_area_m2=self.total_area_m2,
        )

    def reset(self) -> None:
        self._mask.fill(False)
        self._frame_ids.clear()
