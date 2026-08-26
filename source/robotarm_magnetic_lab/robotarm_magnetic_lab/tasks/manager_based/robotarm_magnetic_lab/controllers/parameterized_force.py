"""Pure 10 Hz parameterized-force contract.

One command is a discrete motion mode plus a continuous normalized strength.
The environment holds that pair for exactly one 0.1 s control period while
the geometry executor recomputes world directions at every 240 Hz physics
step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


PHYSICS_HZ = 240
CONTROL_HZ = 10
PHYSICS_STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ
GRAVITY_M_S2 = 9.81
WORLD_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


class ParameterizedForceMode(IntEnum):
    HOLD = 0
    MOVE_POS = 1
    MOVE_NEG = 2
    VIEW_POS = 3
    VIEW_NEG = 4
    UP = 5


MOVE_MODES = frozenset((ParameterizedForceMode.MOVE_POS, ParameterizedForceMode.MOVE_NEG))
VIEW_MODES = frozenset((ParameterizedForceMode.VIEW_POS, ParameterizedForceMode.VIEW_NEG))
NEGATIVE_MODES = frozenset((ParameterizedForceMode.MOVE_NEG, ParameterizedForceMode.VIEW_NEG))


@dataclass(frozen=True)
class ParameterizedForceConfig:
    move_min_ratio: float = 0.70
    move_max_ratio: float = 1.40
    view_min_ratio: float = 0.20
    view_max_ratio: float = 0.50
    up_min_ratio: float = 0.80
    up_max_ratio: float = 1.05
    physics_hz: int = PHYSICS_HZ
    control_hz: int = CONTROL_HZ

    def __post_init__(self) -> None:
        if self.physics_hz != PHYSICS_HZ or self.control_hz != CONTROL_HZ:
            raise ValueError("parameterized-force clocks are fixed at 240/10 Hz")
        for low_name, high_name in (
            ("move_min_ratio", "move_max_ratio"),
            ("view_min_ratio", "view_max_ratio"),
            ("up_min_ratio", "up_max_ratio"),
        ):
            low = float(getattr(self, low_name))
            high = float(getattr(self, high_name))
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low <= high <= 3.0):
                raise ValueError(f"{low_name}/{high_name} must be finite, ordered, and in (0, 3]")


@dataclass(frozen=True)
class EndpointForceCommand:
    mode: ParameterizedForceMode
    alpha: float
    force_ratio: float
    target_total_force_n: float
    camera_force_world: np.ndarray
    other_force_world: np.ndarray
    direction_world: np.ndarray


def validate_parameterized_command(
    mode: ParameterizedForceMode | int,
    alpha: float,
) -> tuple[ParameterizedForceMode, float]:
    resolved_mode = ParameterizedForceMode(int(mode))
    resolved_alpha = float(alpha)
    if not math.isfinite(resolved_alpha) or not 0.0 <= resolved_alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    return resolved_mode, resolved_alpha


def parameterized_force_ratio(
    mode: ParameterizedForceMode | int,
    alpha: float,
    config: ParameterizedForceConfig = ParameterizedForceConfig(),
) -> float:
    mode, alpha = validate_parameterized_command(mode, alpha)
    if mode == ParameterizedForceMode.HOLD:
        return 0.0
    if mode in MOVE_MODES:
        low, high = config.move_min_ratio, config.move_max_ratio
    elif mode in VIEW_MODES:
        low, high = config.view_min_ratio, config.view_max_ratio
    else:
        low, high = config.up_min_ratio, config.up_max_ratio
    return float(low + (high - low) * alpha)


def horizontal_lateral_direction(camera_axis_world, *, negative: bool = False) -> np.ndarray:
    axis = np.asarray(camera_axis_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis).all() or not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("camera axis must be finite and non-zero")
    axis /= norm
    direction = np.cross(WORLD_UP, axis)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("MOVE/VIEW direction is undefined for a vertical capsule axis")
    direction /= norm
    return -direction if negative else direction


def parameterized_endpoint_forces(
    mode: ParameterizedForceMode | int,
    alpha: float,
    *,
    mass_kg: float,
    camera_axis_world,
    config: ParameterizedForceConfig = ParameterizedForceConfig(),
) -> EndpointForceCommand:
    mode, alpha = validate_parameterized_command(mode, alpha)
    mass = float(mass_kg)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    ratio = parameterized_force_ratio(mode, alpha, config)
    target = ratio * mass * GRAVITY_M_S2
    zero = np.zeros(3, dtype=np.float64)
    if mode == ParameterizedForceMode.HOLD:
        return EndpointForceCommand(mode, alpha, 0.0, 0.0, zero.copy(), zero.copy(), zero.copy())
    if mode == ParameterizedForceMode.UP:
        return EndpointForceCommand(mode, alpha, ratio, target, target * WORLD_UP, zero.copy(), WORLD_UP.copy())
    direction = horizontal_lateral_direction(camera_axis_world, negative=mode in NEGATIVE_MODES)
    if mode in MOVE_MODES:
        endpoint = 0.5 * target * direction
        return EndpointForceCommand(mode, alpha, ratio, target, endpoint, endpoint.copy(), direction)
    return EndpointForceCommand(mode, alpha, ratio, target, target * direction, zero.copy(), direction)
