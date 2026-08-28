"""Simulator-independent batched parameterized-force mathematics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .parameterized_force import GRAVITY_M_S2, ParameterizedForceConfig


@dataclass(frozen=True)
class BatchedEndpointForceCommand:
    modes: torch.Tensor
    alpha: torch.Tensor
    force_ratios: torch.Tensor
    target_total_forces_n: torch.Tensor
    camera_forces_world: torch.Tensor
    other_forces_world: torch.Tensor
    directions_world: torch.Tensor


def batched_parameterized_endpoint_forces(
    modes: torch.Tensor,
    alpha: torch.Tensor,
    masses_kg: torch.Tensor,
    camera_axes_world: torch.Tensor,
    config: ParameterizedForceConfig = ParameterizedForceConfig(),
) -> BatchedEndpointForceCommand:
    """Resolve one parameterized command per row entirely on the input device."""
    modes = modes.to(dtype=torch.int64).reshape(-1)
    alpha = alpha.to(device=modes.device).reshape(-1)
    masses = masses_kg.to(device=modes.device).reshape(-1)
    axes = camera_axes_world.to(device=modes.device).reshape(-1, 3)
    if not (len(modes) == len(alpha) == len(masses) == len(axes)):
        raise ValueError("batched parameterized-force rows must match")
    dtype = torch.promote_types(alpha.dtype, masses.dtype)
    dtype = torch.promote_types(dtype, axes.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float32
    alpha = alpha.to(dtype=dtype)
    masses = masses.to(dtype=dtype)
    axes = axes.to(dtype=dtype)
    if torch.any((modes < 0) | (modes > 5)).item():
        raise ValueError("mode IDs must be in [0, 5]")
    if torch.any(~torch.isfinite(alpha) | (alpha < 0) | (alpha > 1)).item():
        raise ValueError("alpha must be finite and in [0, 1]")
    if torch.any(~torch.isfinite(masses) | (masses <= 0)).item():
        raise ValueError("masses must be finite and positive")
    axis_norm = torch.linalg.vector_norm(axes, dim=1)
    if torch.any(~torch.isfinite(axes)).item() or torch.any(axis_norm <= 1.0e-12).item():
        raise ValueError("camera axes must be finite and non-zero")
    axes = axes / axis_norm[:, None]
    world_up = torch.tensor(
        [0.0, 0.0, 1.0], dtype=dtype, device=modes.device
    )
    lateral = torch.linalg.cross(world_up.expand_as(axes), axes)
    lateral_norm = torch.linalg.vector_norm(lateral, dim=1)
    move = (modes == 1) | (modes == 2)
    view = (modes == 3) | (modes == 4)
    lateral_modes = move | view
    bad_rows = torch.nonzero(
        lateral_modes & (lateral_norm <= 1.0e-12), as_tuple=False
    ).reshape(-1)
    if bad_rows.numel():
        raise ValueError(
            f"undefined lateral direction at environment rows {bad_rows.tolist()}"
        )
    lateral = lateral / lateral_norm.clamp_min(1.0e-12)[:, None]
    negative = (modes == 2) | (modes == 4)
    lateral = torch.where(negative[:, None], -lateral, lateral)
    ratios = torch.zeros_like(alpha)
    ratios = torch.where(
        move,
        config.move_min_ratio
        + alpha * (config.move_max_ratio - config.move_min_ratio),
        ratios,
    )
    ratios = torch.where(
        view,
        config.view_min_ratio
        + alpha * (config.view_max_ratio - config.view_min_ratio),
        ratios,
    )
    ratios = torch.where(
        modes == 5,
        config.up_min_ratio + alpha * (config.up_max_ratio - config.up_min_ratio),
        ratios,
    )
    target = ratios * masses * GRAVITY_M_S2
    directions = torch.where((modes == 5)[:, None], world_up.expand_as(axes), lateral)
    directions = torch.where(
        (modes == 0)[:, None], torch.zeros_like(directions), directions
    )
    camera_scale = torch.where(move, 0.5 * target, target)
    camera = camera_scale[:, None] * directions
    other = torch.where(
        move[:, None],
        0.5 * target[:, None] * directions,
        torch.zeros_like(camera),
    )
    return BatchedEndpointForceCommand(
        modes=modes,
        alpha=alpha,
        force_ratios=ratios,
        target_total_forces_n=target,
        camera_forces_world=camera,
        other_forces_world=other,
        directions_world=directions,
    )


def batched_equivalent_com_wrench(
    camera_forces_world: torch.Tensor,
    other_forces_world: torch.Tensor,
    camera_positions_world: torch.Tensor,
    other_positions_world: torch.Tensor,
    com_positions_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert two endpoint forces per environment to an equivalent COM wrench."""
    values = (
        camera_forces_world,
        other_forces_world,
        camera_positions_world,
        other_positions_world,
        com_positions_world,
    )
    shape = camera_forces_world.shape
    if len(shape) != 2 or shape[1] != 3 or any(value.shape != shape for value in values):
        raise ValueError("batched COM-wrench inputs must all have shape [E, 3]")
    if any(torch.any(~torch.isfinite(value)).item() for value in values):
        raise ValueError("batched COM-wrench inputs must be finite")
    resultant = camera_forces_world + other_forces_world
    torque = torch.linalg.cross(
        camera_positions_world - com_positions_world,
        camera_forces_world,
    ) + torch.linalg.cross(
        other_positions_world - com_positions_world,
        other_forces_world,
    )
    return resultant, torque
