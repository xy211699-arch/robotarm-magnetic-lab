"""Pure contracts for bounded TASK-003 world-frame capsule forces."""

from __future__ import annotations

import math

import numpy as np


GRAVITY_M_S2 = 9.81
DEFAULT_FORCE_WEIGHT_RATIO = 0.9
DEFAULT_VERTICAL_FORCE_WEIGHT_RATIO = 1.1
MAXIMUM_FORCE_WEIGHT_RATIO = 2.0


def validate_force_weight_ratio(value: float) -> float:
    """Return one finite ratio in the frozen ``(0, 2]`` interval."""
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 < ratio <= MAXIMUM_FORCE_WEIGHT_RATIO:
        raise ValueError("force_weight_ratio must be finite and in (0, 2]")
    return ratio


def normalize_force_direction(value: np.ndarray) -> np.ndarray:
    """Clip a three-vector componentwise and then limit its Euclidean norm."""
    vector = np.asarray(value, dtype=np.float64).reshape(3).copy()
    if not bool(np.isfinite(vector).all()):
        raise ValueError("force direction must be finite")
    vector = np.clip(vector, -1.0, 1.0)
    norm = float(np.linalg.norm(vector))
    return vector if norm <= 1.0 else vector / norm


def force_world_from_action(
    action: np.ndarray,
    mass_kg: float,
    force_weight_ratio: float = DEFAULT_FORCE_WEIGHT_RATIO,
    vertical_force_weight_ratio: float = DEFAULT_VERTICAL_FORCE_WEIGHT_RATIO,
) -> np.ndarray:
    """Scale normalized world XY/Z components by fractions of live body weight."""
    ratio = validate_force_weight_ratio(force_weight_ratio)
    vertical_ratio = validate_force_weight_ratio(vertical_force_weight_ratio)
    mass = float(mass_kg)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("capsule mass must be finite and positive")
    component_ratios = np.array([ratio, ratio, vertical_ratio], dtype=np.float64)
    return normalize_force_direction(action) * component_ratios * mass * GRAVITY_M_S2
