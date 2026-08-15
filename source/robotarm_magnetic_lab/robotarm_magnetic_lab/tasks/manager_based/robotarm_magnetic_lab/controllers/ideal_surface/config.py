"""Configuration frozen by the ``ideal_surface_v1`` action contract."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class IdealSurfaceConfig:
    schema_version: str = "ideal_surface_v1"
    action_duration_s: float = 1.0
    tilt_step_rad: float = math.radians(15.0)
    precession_step_rad: float = math.radians(15.0)
    roll_arc_length_m: float = 0.004
    upright_enter_rad: float = math.radians(5.0)
    upright_exit_rad: float = math.radians(8.0)
    logical_stability_s: float = 0.1
    side_contact_separation_fraction: float = 0.25
    contact_clearance_radius_fraction: float = 0.02
    planned_penetration_radius_fraction: float = 0.01
    hard_penetration_radius_fraction: float = 0.05
    recovery_query_radius_scale: float = 2.0
    capsule_radius_m: float = 0.0065
    capsule_cylinder_half_length_m: float = 0.006000000052154064

    def __post_init__(self) -> None:
        if self.schema_version != "ideal_surface_v1":
            raise ValueError("schema_version is frozen as ideal_surface_v1")
        if self.action_duration_s <= 0.0:
            raise ValueError("action_duration_s must be positive")
        if not 0.0 < self.upright_enter_rad < self.upright_exit_rad < math.pi / 2:
            raise ValueError("upright thresholds must have enter < exit < pi/2")
        if self.tilt_step_rad <= 0.0 or self.precession_step_rad <= 0.0:
            raise ValueError("tilt and precession steps must be positive")
        if self.roll_arc_length_m <= 0.0:
            raise ValueError("roll_arc_length_m must be positive")
        if self.logical_stability_s <= 0.0:
            raise ValueError("logical_stability_s must be positive")
        if self.capsule_radius_m <= 0.0 or self.capsule_cylinder_half_length_m <= 0.0:
            raise ValueError("preflight-confirmed capsule dimensions must be positive")
        for name in (
            "contact_clearance_radius_fraction",
            "planned_penetration_radius_fraction",
            "hard_penetration_radius_fraction",
            "recovery_query_radius_scale",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.planned_penetration_radius_fraction >= self.hard_penetration_radius_fraction:
            raise ValueError("planned penetration threshold must be below hard penetration")

