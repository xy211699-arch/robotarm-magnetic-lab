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


class CoverageAccumulator:
    def __init__(self, vertex_count: int) -> None:
        if int(vertex_count) <= 0:
            raise ValueError("vertex_count must be positive")
        self._mask = np.zeros(int(vertex_count), dtype=np.bool_)
        self._frame_ids: set[Hashable] = set()

    @property
    def mask(self) -> np.ndarray:
        result = self._mask.copy()
        result.setflags(write=False)
        return result

    @property
    def recorded_frame_count(self) -> int:
        return len(self._frame_ids)

    def update(self, frame_id: Hashable, visible_vertex_indices: Iterable[int]) -> CoverageUpdate:
        if frame_id in self._frame_ids:
            cumulative = int(self._mask.sum())
            return CoverageUpdate(frame_id, False, 0, 0, cumulative, cumulative / len(self._mask))
        indices = np.asarray(list(visible_vertex_indices), dtype=np.int64).reshape(-1)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= len(self._mask)):
            raise ValueError("visible vertex index is out of range")
        indices = np.unique(indices)
        previous = self._mask.copy()
        self._mask[indices] = True
        self._frame_ids.add(frame_id)
        newly = int(np.count_nonzero(self._mask & ~previous))
        cumulative = int(self._mask.sum())
        return CoverageUpdate(
            frame_id=frame_id,
            updated=True,
            visible_count=int(len(indices)),
            newly_covered_count=newly,
            cumulative_count=cumulative,
            coverage_fraction=cumulative / len(self._mask),
        )

    def reset(self) -> None:
        self._mask.fill(False)
        self._frame_ids.clear()
