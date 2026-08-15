"""Exact support geometry for the preflight-confirmed capsule collider."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import normalized


@dataclass(frozen=True)
class Spherocylinder:
    radius_m: float
    cylinder_half_length_m: float

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0 or self.cylinder_half_length_m <= 0.0:
            raise ValueError("spherocylinder dimensions must be positive")

    @property
    def cylinder_length_m(self) -> float:
        return 2.0 * float(self.cylinder_half_length_m)

    @property
    def tip_to_tip_length_m(self) -> float:
        return 2.0 * (float(self.radius_m) + float(self.cylinder_half_length_m))

    def support_distance(self, axis_world: np.ndarray, normal_world: np.ndarray) -> float:
        axis = normalized(axis_world, name="capsule axis")
        normal = normalized(normal_world, name="support normal")
        return float(self.radius_m) + float(self.cylinder_half_length_m) * abs(float(axis @ normal))

    def effective_roll_radius(self) -> float:
        return float(self.radius_m)

    def axial_end_centers(self, center_world: np.ndarray, axis_world: np.ndarray) -> np.ndarray:
        center = np.asarray(center_world, dtype=np.float64).reshape(3)
        axis = normalized(axis_world, name="capsule axis")
        offset = float(self.cylinder_half_length_m) * axis
        return np.stack((center - offset, center + offset))

