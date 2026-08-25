"""Metrics for stable-contact MOVE calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MoveDisplacementMeasurement:
    """One paired 0.1 s MOVE/HOLD displacement measurement."""

    active_signed_m: float
    hold_signed_m: float
    corrected_signed_m: float


def corrected_move_displacement(
    active_start_com,
    active_end_com,
    hold_start_com,
    hold_end_com,
    command_direction_world,
) -> MoveDisplacementMeasurement:
    """Return command-aligned MOVE displacement after paired HOLD correction.

    Both trajectories must start from the same stable-contact state.  The
    command direction points toward the requested MOVE direction, so a
    physically correct MOVE_POS or MOVE_NEG response is positive.
    """

    direction = np.asarray(command_direction_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("command direction must be finite and non-zero")
    direction /= norm

    active_delta = np.asarray(active_end_com, dtype=np.float64).reshape(3) - np.asarray(
        active_start_com, dtype=np.float64
    ).reshape(3)
    hold_delta = np.asarray(hold_end_com, dtype=np.float64).reshape(3) - np.asarray(
        hold_start_com, dtype=np.float64
    ).reshape(3)
    if not (np.isfinite(active_delta).all() and np.isfinite(hold_delta).all()):
        raise ValueError("trajectory positions must be finite")

    active_signed = float(np.dot(active_delta, direction))
    hold_signed = float(np.dot(hold_delta, direction))
    return MoveDisplacementMeasurement(
        active_signed_m=active_signed,
        hold_signed_m=hold_signed,
        corrected_signed_m=active_signed - hold_signed,
    )
