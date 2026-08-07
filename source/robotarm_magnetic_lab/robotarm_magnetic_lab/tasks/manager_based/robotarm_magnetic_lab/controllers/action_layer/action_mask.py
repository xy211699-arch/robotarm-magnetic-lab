"""Deterministic deployable-state action masks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .command_state import roll_direction_from_field
from .config import ActionLayerConfig
from .types import AtomicAction, DeviceSnapshot, MagnetCommandState


@dataclass(frozen=True)
class ActionMask:
    values: np.ndarray
    reasons: tuple[str, ...]

    def allows(self, action: AtomicAction | int) -> bool:
        return bool(self.values[int(action)])


def _inside_workspace(position: np.ndarray, cfg: ActionLayerConfig) -> bool:
    value = np.asarray(position, dtype=np.float64)
    return bool(
        np.all(value >= np.asarray(cfg.workspace_min_world_m))
        and np.all(value <= np.asarray(cfg.workspace_max_world_m))
    )


def compute_action_mask(
    command: MagnetCommandState,
    snapshot: DeviceSnapshot,
    cfg: ActionLayerConfig,
    *,
    busy: bool = False,
) -> ActionMask:
    """Return the 11-action mask without capsule/contact ground truth."""
    mask = np.zeros(len(AtomicAction), dtype=np.bool_)
    reasons = ["available"] * len(AtomicAction)
    finite = np.isfinite(snapshot.joint_position_rad).all() and np.isfinite(
        snapshot.joint_velocity_rad_s
    ).all()
    if busy:
        return ActionMask(mask, tuple("executor_busy" for _ in AtomicAction))
    if not snapshot.controller_connected or not finite or snapshot.environment_terminated:
        reason = (
            "controller_disconnected"
            if not snapshot.controller_connected
            else "environment_terminated"
            if snapshot.environment_terminated
            else "nonfinite_device_state"
        )
        return ActionMask(mask, tuple(reason for _ in AtomicAction))

    mask[:] = True
    theta_step = cfg.tilt_increment_rad
    theta_min = cfg.theta_min_rad
    theta_max = cfg.theta_max_rad
    if command.theta_rad + theta_step > theta_max:
        mask[int(AtomicAction.TILT_POS)] = False
        reasons[int(AtomicAction.TILT_POS)] = "tilt_upper_bound"
    if command.theta_rad - theta_step < theta_min:
        mask[int(AtomicAction.TILT_NEG)] = False
        reasons[int(AtomicAction.TILT_NEG)] = "tilt_lower_bound"

    if abs(math.sin(command.theta_rad)) < cfg.azimuth_min_sine:
        for action in (AtomicAction.AZIMUTH_POS, AtomicAction.AZIMUTH_NEG):
            mask[int(action)] = False
            reasons[int(action)] = "azimuth_singularity"

    roll_band = (
        cfg.roll_theta_min_rad
        <= command.theta_rad
        <= cfg.roll_theta_max_rad
    )
    roll_direction = roll_direction_from_field(command.field_direction_world)
    for action, sign in (
        (AtomicAction.ROLL_POS, 1.0),
        (AtomicAction.ROLL_NEG, -1.0),
    ):
        target = (
            command.magnet_position_target_world_m
            + sign * cfg.roll_displacement_m * roll_direction
        )
        if not roll_band:
            mask[int(action)] = False
            reasons[int(action)] = "roll_tilt_band"
        elif not _inside_workspace(target, cfg):
            mask[int(action)] = False
            reasons[int(action)] = "workspace_limit"

    for action in (AtomicAction.TURN_POS, AtomicAction.TURN_NEG):
        if not roll_band:
            mask[int(action)] = False
            reasons[int(action)] = "turn_tilt_band"

    approach_direction = np.asarray(cfg.approach_direction_world, dtype=np.float64)
    approach_direction /= max(float(np.linalg.norm(approach_direction)), 1.0e-12)
    for action, sign in (
        (AtomicAction.APPROACH, 1.0),
        (AtomicAction.RETREAT, -1.0),
    ):
        target = (
            command.magnet_position_target_world_m
            + sign * cfg.approach_displacement_m * approach_direction
        )
        if not _inside_workspace(target, cfg):
            mask[int(action)] = False
            reasons[int(action)] = "workspace_limit"

    return ActionMask(mask, tuple(reasons))
