"""Frozen shared configuration for flat and stomach primitive tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalPrimitiveControllerCfg:
    """All tunable values allowed by the TASK-004 contract."""

    capsule_mass_kg: float = 0.0057349997
    capsule_half_total_length_m: float = 0.0125
    capsule_half_cylinder_length_m: float = 0.006
    gravity_m_s2: float = 9.81
    axis_kp_nm_per_rad: float = 3.0e-5
    axis_kd_nms_per_rad: float = 8.0e-6
    roll_damping_nms_per_rad: float = 1.0e-6
    torque_limit_nm: float = 3.0e-5
    anchor_kp_n_per_m: float = 3.0
    anchor_kd_ns_per_m: float = 0.15
    horizontal_force_limit_weight_ratio: float = 1.0
    downward_preload_weight_ratio: float = 0.15
    stable_duration_s: float = 0.4
    max_stable_linear_speed_m_s: float = 0.02
    max_stable_angular_speed_rad_s: float = 0.15
    transition_tolerance_rad: float = 0.05235987755982989
    tilt_tolerance_rad: float = 0.03490658503988659
    cone_tilt_rmse_limit_rad: float = 0.08726646259971647
    cone_coverage_tolerance_rad: float = 0.17453292519943295
    side_start_min_rad: float = 1.3089969389957472
    side_start_max_rad: float = 1.8325957145940461
    upright_start_max_rad: float = 0.08726646259971647
    cone_start_tolerance_rad: float = 0.05235987755982989
    target_tilt_rad: float = 0.5235987755982988
    motion_duration_s: tuple[float, float, float, float] = (5.5, 4.5, 3.5, 8.0)
    hard_timeout_s: tuple[float, float, float, float] = (8.0, 7.0, 6.0, 9.5)

    def __post_init__(self) -> None:
        finite_positive = (
            self.capsule_mass_kg, self.capsule_half_total_length_m,
            self.capsule_half_cylinder_length_m, self.gravity_m_s2,
            self.axis_kp_nm_per_rad, self.axis_kd_nms_per_rad,
            self.roll_damping_nms_per_rad, self.torque_limit_nm,
            self.anchor_kp_n_per_m, self.anchor_kd_ns_per_m,
            self.horizontal_force_limit_weight_ratio,
        )
        if not all(__import__("math").isfinite(v) and v > 0.0 for v in finite_positive):
            raise ValueError("controller physical and gain values must be finite and positive")
        if len(self.motion_duration_s) != 4 or len(self.hard_timeout_s) != 4:
            raise ValueError("duration tuples must contain exactly four values")
        if any(not (0.0 < motion < timeout < 10.0) for motion, timeout in zip(self.motion_duration_s, self.hard_timeout_s)):
            raise ValueError("each duration must satisfy 0 < motion < timeout < 10 seconds")

    @property
    def weight_n(self) -> float:
        return self.capsule_mass_kg * self.gravity_m_s2

    @property
    def xy_force_limit_n(self) -> float:
        return self.horizontal_force_limit_weight_ratio * self.weight_n

    @property
    def downward_preload_n(self) -> float:
        return self.downward_preload_weight_ratio * self.weight_n


def make_local_primitive_controller_cfg() -> LocalPrimitiveControllerCfg:
    """Return the exact shared controller configuration."""

    return LocalPrimitiveControllerCfg()


LocalPrimitiveActionCfg = LocalPrimitiveControllerCfg


def make_local_primitive_action_cfg() -> LocalPrimitiveControllerCfg:
    """Compatibility factory; task action configs wrap the controller factory."""

    return make_local_primitive_controller_cfg()
