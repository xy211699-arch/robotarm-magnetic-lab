"""Frozen shared configuration for flat and stomach primitive tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalPrimitiveActionCfg:
    """All tunable values allowed by the TASK-004 contract."""

    capsule_mass_kg: float = 0.0057349997
    half_total_length_m: float = 0.0125
    half_cylinder_length_m: float = 0.006
    gravity_m_s2: float = 9.81
    axis_kp_nm_per_rad: float = 1.2e-5
    axis_kd_nm_s_per_rad: float = 2.0e-6
    roll_damping_nm_s_per_rad: float = 1.0e-6
    torque_limit_nm: float = 2.0e-5
    anchor_kp_n_per_m: float = 0.8
    anchor_kd_n_s_per_m: float = 0.03
    xy_force_weight_ratio_limit: float = 0.5
    downward_preload_weight_ratio: float = 0.15
    stable_duration_s: float = 0.4
    stable_linear_speed_m_s: float = 0.02
    stable_angular_speed_rad_s: float = 0.15
    transition_tolerance_rad: float = 0.05235987755982989
    tilt_tolerance_rad: float = 0.03490658503988659
    cone_tilt_rmse_limit_rad: float = 0.08726646259971647
    cone_coverage_tolerance_rad: float = 0.17453292519943295
    side_start_min_rad: float = 1.3089969389957472
    side_start_max_rad: float = 1.8325957145940461
    upright_start_max_rad: float = 0.08726646259971647
    cone_start_tolerance_rad: float = 0.05235987755982989
    target_tilt_rad: float = 0.5235987755982988
    motion_durations_s: tuple[float, float, float, float] = (5.5, 4.5, 3.5, 8.0)
    timeout_durations_s: tuple[float, float, float, float] = (8.0, 7.0, 6.0, 9.5)

    @property
    def weight_n(self) -> float:
        return self.capsule_mass_kg * self.gravity_m_s2

    @property
    def xy_force_limit_n(self) -> float:
        return self.xy_force_weight_ratio_limit * self.weight_n

    @property
    def downward_preload_n(self) -> float:
        return self.downward_preload_weight_ratio * self.weight_n


def make_local_primitive_action_cfg() -> LocalPrimitiveActionCfg:
    """Return the single shared action configuration used by both scenes."""

    return LocalPrimitiveActionCfg()
