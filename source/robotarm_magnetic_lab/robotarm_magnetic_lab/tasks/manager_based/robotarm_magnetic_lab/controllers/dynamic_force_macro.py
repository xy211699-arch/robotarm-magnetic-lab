"""Pure contracts for TASK-008 six-action force macros."""

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
    HOLD = 0
    MOVE_POS = 1
    MOVE_NEG = 2
    VIEW_POS = 3
    VIEW_NEG = 4
    UP = 5


@dataclass(frozen=True)
class DynamicForceMacroConfig:
    physics_hz: int = 240
    environment_hz: int = 60
    camera_hz: int = 30
    actor_hz: int = 1
    move_force_ratio: float = 0.9
    view_force_ratio: float = 0.9
    up_force_ratio: float = 0.9
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
        for name in ("move_force_ratio", "view_force_ratio", "up_force_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= self.max_force_ratio:
                raise ValueError(f"{name} must be finite and in (0, {self.max_force_ratio}]")
        if (self.wait_substeps, self.force_substeps, self.action_substeps) != (48, 144, 240):
            raise ValueError("TASK-008 timing is frozen at 48/144/48 physics substeps")


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
    if action in (DynamicForceMacroActionId.MOVE_POS, DynamicForceMacroActionId.MOVE_NEG):
        direction = _unit(lateral_direction_world, name="lateral direction")
        if action == DynamicForceMacroActionId.MOVE_NEG:
            direction = -direction
        force = 0.5 * config.move_force_ratio * mass * GRAVITY_M_S2 * direction
        return (PointForce("camera", camera, force), PointForce("other", other, force.copy()))
    if action in (DynamicForceMacroActionId.VIEW_POS, DynamicForceMacroActionId.VIEW_NEG):
        direction = _unit(lateral_direction_world, name="lateral direction")
        if action == DynamicForceMacroActionId.VIEW_NEG:
            direction = -direction
        force = config.view_force_ratio * mass * GRAVITY_M_S2 * direction
        return (PointForce("camera", camera, force),)
    # UP is a pure endpoint couple, not a net upward body force.  The previous
    # camera-only vertical force produced the desired moment but also lifted the
    # COM; on a changing stomach-wall contact patch, the resulting contact
    # reaction could pivot the capsule about the camera end and visibly raise
    # the wrong end.  Resolve the same orientation moment into equal/opposite
    # endpoint forces: the camera end is driven toward world +Z while the other
    # end is explicitly driven into the support surface.
    camera_axis = _unit(camera - other, name="camera endpoint axis")
    camera_lift = WORLD_UP - float(np.dot(WORLD_UP, camera_axis)) * camera_axis
    if float(np.linalg.norm(camera_lift)) <= 1.0e-12:
        # At exactly camera-down the shortest rise plane is geometrically
        # ambiguous.  Use the already deterministic lateral direction to tip
        # away from that unstable pole.  Camera-up needs no further moment.
        camera_lift = (
            _unit(lateral_direction_world, name="camera-down lift direction")
            if float(np.dot(WORLD_UP, camera_axis)) < 0.0
            else np.zeros(3, dtype=np.float64)
        )
    force = 0.5 * config.up_force_ratio * mass * GRAVITY_M_S2 * camera_lift
    return (
        PointForce("camera", camera, force),
        PointForce("other", other, -force),
    )


def equivalent_com_wrench(point_forces: tuple[PointForce, ...], com_world) -> tuple[np.ndarray, np.ndarray]:
    com = np.asarray(com_world, dtype=np.float64).reshape(3)
    force = np.zeros(3, dtype=np.float64)
    torque = np.zeros(3, dtype=np.float64)
    for item in point_forces:
        force += np.asarray(item.force_world, dtype=np.float64)
        torque += np.cross(np.asarray(item.position_world, dtype=np.float64) - com, item.force_world)
    return force, torque


def resolved_force_levels_n(mass_kg: float, config: DynamicForceMacroConfig) -> dict[str, float]:
    """Resolve the three CLI force ratios to auditable Newton values."""
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
        "view_force_ratio": config.view_force_ratio,
        "view_camera_endpoint_force_n": config.view_force_ratio * weight,
        "up_force_ratio": config.up_force_ratio,
        "up_couple_force_ratio": config.up_force_ratio,
        "up_force_per_endpoint_max_n": 0.5 * config.up_force_ratio * weight,
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
