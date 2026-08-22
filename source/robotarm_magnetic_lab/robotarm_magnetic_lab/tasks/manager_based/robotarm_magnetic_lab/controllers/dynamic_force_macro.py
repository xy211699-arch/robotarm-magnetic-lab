"""Pure contracts for TASK-008 fourteen-level force macros."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


GRAVITY_M_S2 = 9.81
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


class NumericalContractError(ValueError):
    """Raised before physics when an action frame cannot be defined safely."""


class DynamicForceMacroActionId(IntEnum):
    # IDs 0..5 are retained exactly for dataset and log compatibility.
    HOLD = 0
    MOVE_POS = 1
    MOVE_NEG = 2
    VIEW_POS = 3
    VIEW_NEG = 4
    UP = 5
    MOVE_POS_MEDIUM = 6
    MOVE_NEG_MEDIUM = 7
    MOVE_POS_HIGH = 8
    MOVE_NEG_HIGH = 9
    VIEW_POS_MEDIUM = 10
    VIEW_NEG_MEDIUM = 11
    VIEW_POS_HIGH = 12
    VIEW_NEG_HIGH = 13


MOVE_ACTION_IDS = frozenset(
    {
        DynamicForceMacroActionId.MOVE_POS,
        DynamicForceMacroActionId.MOVE_NEG,
        DynamicForceMacroActionId.MOVE_POS_MEDIUM,
        DynamicForceMacroActionId.MOVE_NEG_MEDIUM,
        DynamicForceMacroActionId.MOVE_POS_HIGH,
        DynamicForceMacroActionId.MOVE_NEG_HIGH,
    }
)
VIEW_ACTION_IDS = frozenset(
    {
        DynamicForceMacroActionId.VIEW_POS,
        DynamicForceMacroActionId.VIEW_NEG,
        DynamicForceMacroActionId.VIEW_POS_MEDIUM,
        DynamicForceMacroActionId.VIEW_NEG_MEDIUM,
        DynamicForceMacroActionId.VIEW_POS_HIGH,
        DynamicForceMacroActionId.VIEW_NEG_HIGH,
    }
)
NEGATIVE_ACTION_IDS = frozenset(
    {
        DynamicForceMacroActionId.MOVE_NEG,
        DynamicForceMacroActionId.MOVE_NEG_MEDIUM,
        DynamicForceMacroActionId.MOVE_NEG_HIGH,
        DynamicForceMacroActionId.VIEW_NEG,
        DynamicForceMacroActionId.VIEW_NEG_MEDIUM,
        DynamicForceMacroActionId.VIEW_NEG_HIGH,
    }
)


@dataclass(frozen=True)
class DynamicForceMacroConfig:
    physics_hz: int = 240
    environment_hz: int = 60
    camera_hz: int = 30
    actor_hz: int = 1
    move_force_ratio: float = 0.40
    move_force_ratio_medium: float = 0.50
    move_force_ratio_high: float = 0.60
    view_force_ratio: float = 0.25
    view_force_ratio_medium: float = 0.35
    view_force_ratio_high: float = 0.45
    up_force_ratio: float = 0.85
    max_force_ratio: float = 3.0
    camera_side_local_axis_sign: int = -1
    wait_substeps: int = 48
    force_substeps: int = 144
    action_substeps: int = 240
    capsule_radius_m: float = 0.0065
    cylinder_height_m: float = 0.012

    def __post_init__(self) -> None:
        if (self.physics_hz, self.environment_hz, self.camera_hz, self.actor_hz) != (240, 60, 30, 1):
            raise ValueError("TASK-008 clocks are frozen at 240/60/30/1 Hz")
        if self.camera_side_local_axis_sign != -1:
            raise ValueError("TASK-008 camera side is capsule local -Z")
        for name in (
            "move_force_ratio",
            "move_force_ratio_medium",
            "move_force_ratio_high",
            "view_force_ratio",
            "view_force_ratio_medium",
            "view_force_ratio_high",
            "up_force_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= self.max_force_ratio:
                raise ValueError(f"{name} must be finite and in (0, {self.max_force_ratio}]")
        if (self.wait_substeps, self.force_substeps, self.action_substeps) != (48, 144, 240):
            raise ValueError("TASK-008 timing is frozen at 48/144/48 physics substeps")


def action_force_ratio(action: DynamicForceMacroActionId | int, config: DynamicForceMacroConfig) -> float:
    """Return the force-to-weight ratio encoded by one action ID."""
    action = DynamicForceMacroActionId(int(action))
    if action in (DynamicForceMacroActionId.MOVE_POS, DynamicForceMacroActionId.MOVE_NEG):
        return config.move_force_ratio
    if action in (DynamicForceMacroActionId.MOVE_POS_MEDIUM, DynamicForceMacroActionId.MOVE_NEG_MEDIUM):
        return config.move_force_ratio_medium
    if action in (DynamicForceMacroActionId.MOVE_POS_HIGH, DynamicForceMacroActionId.MOVE_NEG_HIGH):
        return config.move_force_ratio_high
    if action in (DynamicForceMacroActionId.VIEW_POS, DynamicForceMacroActionId.VIEW_NEG):
        return config.view_force_ratio
    if action in (DynamicForceMacroActionId.VIEW_POS_MEDIUM, DynamicForceMacroActionId.VIEW_NEG_MEDIUM):
        return config.view_force_ratio_medium
    if action in (DynamicForceMacroActionId.VIEW_POS_HIGH, DynamicForceMacroActionId.VIEW_NEG_HIGH):
        return config.view_force_ratio_high
    if action == DynamicForceMacroActionId.UP:
        return config.up_force_ratio
    return 0.0


@dataclass(frozen=True)
class MacroPhase:
    name: str
    force_active: bool


@dataclass(frozen=True)
class PointForce:
    endpoint: str
    position_world: np.ndarray
    force_world: np.ndarray


def _unit(vector, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not math.isfinite(norm) or norm <= 1.0e-12:
        raise NumericalContractError(f"{name} must be finite and non-zero")
    return value / norm


def phase_for_substep(action: DynamicForceMacroActionId | int, substep: int) -> MacroPhase:
    action = DynamicForceMacroActionId(int(action))
    index = int(substep)
    if not 0 <= index < 240:
        raise ValueError("macro substep must be in [0, 239]")
    if action == DynamicForceMacroActionId.HOLD:
        return MacroPhase("hold", False)
    if action == DynamicForceMacroActionId.UP:
        return MacroPhase("force", True)
    if index < 48:
        return MacroPhase("wait_before", False)
    if index < 192:
        return MacroPhase("force", True)
    return MacroPhase("wait_after", False)


def lateral_direction_world(camera_axis_world, *, positive: bool = True) -> np.ndarray:
    direction = _unit(np.cross(WORLD_UP, _unit(camera_axis_world, name="camera axis")), name="lateral direction")
    return direction if positive else -direction


def camera_sphere_centers_local(
    camera_offset_local: np.ndarray,
    cylinder_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the physical hemisphere center nearest the mounted camera.

    The capsule rigid body has two hemisphere centers at +/- half the straight
    cylinder height.  Camera-side identity is derived from the actual CameraCfg
    local offset instead of assuming a fixed local-axis sign.
    """
    camera_offset = np.asarray(camera_offset_local, dtype=np.float64).reshape(3)
    height = float(cylinder_height_m)
    if not np.isfinite(camera_offset).all() or not math.isfinite(height) or height <= 0.0:
        raise NumericalContractError("camera offset and cylinder height must be finite")
    half = 0.5 * height
    candidates = np.asarray([[0.0, 0.0, -half], [0.0, 0.0, half]], dtype=np.float64)
    distances = np.linalg.norm(candidates - camera_offset[None, :], axis=1)
    if abs(float(distances[0] - distances[1])) <= 1.0e-12:
        raise NumericalContractError("camera is equidistant from both capsule hemisphere centers")
    camera_index = int(np.argmin(distances))
    return candidates[camera_index].copy(), candidates[1 - camera_index].copy()


def point_forces_for_action(
    action: DynamicForceMacroActionId | int,
    *,
    mass_kg: float,
    lateral_direction_world: np.ndarray,
    camera_center_world: np.ndarray,
    other_center_world: np.ndarray,
    config: DynamicForceMacroConfig,
) -> tuple[PointForce, ...]:
    action = DynamicForceMacroActionId(int(action))
    mass = float(mass_kg)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    camera = np.asarray(camera_center_world, dtype=np.float64).reshape(3)
    other = np.asarray(other_center_world, dtype=np.float64).reshape(3)
    if action == DynamicForceMacroActionId.HOLD:
        return ()
    if action in MOVE_ACTION_IDS:
        direction = _unit(lateral_direction_world, name="lateral direction")
        if action in NEGATIVE_ACTION_IDS:
            direction = -direction
        force = 0.5 * action_force_ratio(action, config) * mass * GRAVITY_M_S2 * direction
        return (PointForce("camera", camera, force), PointForce("other", other, force.copy()))
    if action in VIEW_ACTION_IDS:
        direction = _unit(lateral_direction_world, name="lateral direction")
        if action in NEGATIVE_ACTION_IDS:
            direction = -direction
        force = action_force_ratio(action, config) * mass * GRAVITY_M_S2 * direction
        return (PointForce("camera", camera, force),)
    # UP is the original single world-up force at the physical camera-side
    # hemisphere center.  Camera-side selection is performed from the live
    # camera mounting config by ``camera_sphere_centers_local``.
    if action == DynamicForceMacroActionId.UP:
        force = config.up_force_ratio * mass * GRAVITY_M_S2 * WORLD_UP
        return (PointForce("camera", camera, force),)
    raise AssertionError(f"unhandled force macro action: {action}")


def equivalent_com_wrench(point_forces: tuple[PointForce, ...], com_world) -> tuple[np.ndarray, np.ndarray]:
    com = np.asarray(com_world, dtype=np.float64).reshape(3)
    force = np.zeros(3, dtype=np.float64)
    torque = np.zeros(3, dtype=np.float64)
    for item in point_forces:
        force += np.asarray(item.force_world, dtype=np.float64)
        torque += np.cross(np.asarray(item.position_world, dtype=np.float64) - com, item.force_world)
    return force, torque


def resolved_force_levels_n(mass_kg: float, config: DynamicForceMacroConfig) -> dict[str, float]:
    """Resolve all MOVE/VIEW tiers and UP to auditable Newton values."""
    mass = float(mass_kg)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    weight = mass * GRAVITY_M_S2
    move_total = config.move_force_ratio * weight
    return {
        "mass_kg": mass,
        "weight_n": weight,
        "move_force_ratio": config.move_force_ratio,
        "move_total_force_n": move_total,
        "move_force_per_endpoint_n": 0.5 * move_total,
        "move_force_ratio_medium": config.move_force_ratio_medium,
        "move_medium_total_force_n": config.move_force_ratio_medium * weight,
        "move_force_ratio_high": config.move_force_ratio_high,
        "move_high_total_force_n": config.move_force_ratio_high * weight,
        "view_force_ratio": config.view_force_ratio,
        "view_camera_endpoint_force_n": config.view_force_ratio * weight,
        "view_force_ratio_medium": config.view_force_ratio_medium,
        "view_medium_camera_endpoint_force_n": config.view_force_ratio_medium * weight,
        "view_force_ratio_high": config.view_force_ratio_high,
        "view_high_camera_endpoint_force_n": config.view_force_ratio_high * weight,
        "up_force_ratio": config.up_force_ratio,
        "up_camera_endpoint_force_n": config.up_force_ratio * weight,
    }


def move_projected_displacement_m(position_onset, position_end, direction_onset) -> float:
    return float(np.dot(np.asarray(position_end) - np.asarray(position_onset), _unit(direction_onset, name="onset direction")))


def view_signed_angle_deg(axis_onset, axis_end, direction_onset) -> float:
    u0 = _unit(axis_onset, name="onset axis")
    u1 = _unit(axis_end, name="end axis")
    d0 = _unit(direction_onset, name="onset direction")
    k0 = _unit(np.cross(u0, d0), name="commanded-plane normal")
    return math.degrees(math.atan2(float(np.dot(k0, np.cross(u0, u1))), float(np.dot(u0, u1))))


def up_elevation_and_crossing(axis_end, initial_horizontal_axis, sampled_axes=()) -> tuple[float, bool]:
    end = _unit(axis_end, name="end axis")
    reference = _unit(initial_horizontal_axis, name="initial horizontal axis")
    elevation = math.degrees(math.asin(float(np.clip(np.dot(end, WORLD_UP), -1.0, 1.0))))
    crossed = any(float(np.dot(_unit(axis, name="sampled axis"), reference)) < -1.0e-6 for axis in sampled_axes)
    return elevation, crossed
