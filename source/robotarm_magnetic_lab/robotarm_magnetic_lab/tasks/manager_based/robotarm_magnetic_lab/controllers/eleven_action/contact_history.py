"""Physics-substep contact history for TASK-005 action preconditions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .types import immutable_vector


class ContactRegion(str, Enum):
    CAMERA_HEMISPHERE = "camera_hemisphere"
    SIDEWALL = "sidewall"
    NONCAMERA_HEMISPHERE = "noncamera_hemisphere"


@dataclass(frozen=True)
class ContactSample:
    physics_substep: int
    point_world: np.ndarray
    normal_world: np.ndarray
    axial_coordinate_m: float
    impulse_n_s: float | None = None
    force_world_n: np.ndarray | None = None
    cylinder_half_length_m: float = 0.006

    def __post_init__(self) -> None:
        if self.physics_substep < 0:
            raise ValueError("physics_substep must be nonnegative")
        object.__setattr__(self, "point_world", immutable_vector(self.point_world, 3))
        object.__setattr__(self, "normal_world", immutable_vector(self.normal_world, 3))
        if not math.isfinite(float(self.axial_coordinate_m)):
            raise ValueError("axial coordinate must be finite")
        if self.impulse_n_s is not None and not math.isfinite(float(self.impulse_n_s)):
            raise ValueError("contact impulse must be finite when present")
        if self.force_world_n is not None:
            object.__setattr__(self, "force_world_n", immutable_vector(self.force_world_n, 3))
        if self.cylinder_half_length_m <= 0.0:
            raise ValueError("cylinder half length must be positive")

    @property
    def region(self) -> ContactRegion:
        from .geometry import classify_contact_region

        return classify_contact_region(self.axial_coordinate_m, self.cylinder_half_length_m)


class SideContactHistory:
    """Bounded contact samples indexed only by absolute physics substep."""

    def __init__(self, *, capacity_substeps: int = 12) -> None:
        if capacity_substeps < 1:
            raise ValueError("capacity_substeps must be positive")
        self.capacity_substeps = int(capacity_substeps)
        self._samples: deque[ContactSample] = deque()

    def __len__(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def append(self, sample: ContactSample) -> None:
        if self._samples and sample.physics_substep < self._samples[-1].physics_substep:
            raise ValueError("contact substeps must be monotonic")
        self._samples.append(sample)
        cutoff = sample.physics_substep - self.capacity_substeps + 1
        while self._samples and self._samples[0].physics_substep < cutoff:
            self._samples.popleft()

    def _recent(self, *, current_substep: int, last_n_substeps: int) -> tuple[ContactSample, ...]:
        if current_substep < 0 or last_n_substeps < 1:
            raise ValueError("contact window must be positive")
        cutoff = int(current_substep) - int(last_n_substeps)
        return tuple(
            sample
            for sample in self._samples
            if cutoff < sample.physics_substep < int(current_substep)
        )

    def recent_contacts(
        self, *, current_substep: int, last_n_substeps: int = 12
    ) -> tuple[ContactSample, ...]:
        return self._recent(current_substep=current_substep, last_n_substeps=last_n_substeps)

    def had_sidewall_contact(self, *, current_substep: int, last_n_substeps: int = 12) -> bool:
        return any(
            sample.region is ContactRegion.SIDEWALL
            for sample in self._recent(current_substep=current_substep, last_n_substeps=last_n_substeps)
        )

    def camera_constraints(
        self, *, current_substep: int, last_n_substeps: int = 12
    ) -> tuple[ContactSample, ...]:
        return tuple(
            sample
            for sample in self._recent(current_substep=current_substep, last_n_substeps=last_n_substeps)
            if sample.region is ContactRegion.CAMERA_HEMISPHERE
        )
