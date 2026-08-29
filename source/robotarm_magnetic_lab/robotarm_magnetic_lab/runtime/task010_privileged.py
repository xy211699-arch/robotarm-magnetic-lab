"""Fixed 65-dimensional asymmetric Critic observation for TASK-010."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


_WIDTHS = OrderedDict(
    (
        ("position_local", 3),
        ("rotation_6d", 6),
        ("linear_velocity", 3),
        ("angular_velocity", 3),
        ("previous_action", 7),
        ("contact", 2),
        ("wall_normal", 3),
        ("wall_normal_valid", 1),
        ("coverage", 2),
        ("coverage_grid", 27),
        ("remaining_time", 1),
        ("recovery", 7),
    )
)


def _slices() -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0
    for name, width in _WIDTHS.items():
        result[name] = slice(offset, offset + width)
        offset += width
    if offset != 65:
        raise AssertionError(f"TASK-010 Critic schema is {offset}, expected 65")
    return result


TASK010_CRITIC_SLICES = _slices()


def quaternion_xyzw_to_rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=1, keepdim=True).clamp_min(1.0e-12)
    x, y, z, w = q.unbind(dim=1)
    # First two columns of the rotation matrix, concatenated column-major.
    first = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), dim=1)
    second = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=1)
    return torch.cat((first, second), dim=1)


@dataclass
class Task010PrivilegedBuilder:
    contact_threshold: float = 1.0e-4

    def build(self, env, recovery_step=None) -> torch.Tensor:
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.task009d0_terms import task009d0_runtime
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.task010_terms import task010_recovery_step

        runtime = task009d0_runtime(env)
        latest = runtime.latest_update
        if latest is None:
            raise RuntimeError("TASK-010 privileged observation requires an initial coverage snapshot")
        recovery_step = task010_recovery_step(env) if recovery_step is None else recovery_step
        capsule = env.scene["capsule"]
        pose = capsule.data.root_pose_w.torch.to(dtype=torch.float32)
        velocity = capsule.data.root_com_vel_w.torch.to(dtype=torch.float32)
        origins = env.scene.env_origins.to(device=pose.device, dtype=pose.dtype)
        previous = env.action_manager.get_term("parameterized_force").previous_action_features.to(dtype=torch.float32)
        sensor = env.scene["capsule_contact"]
        net = getattr(sensor.data.net_forces_w, "torch", sensor.data.net_forces_w).to(device=pose.device, dtype=torch.float32)
        while net.ndim > 2:
            net = net.sum(dim=1)
        intensity = torch.linalg.vector_norm(net, dim=1)
        valid = intensity >= self.contact_threshold
        normal = torch.where(valid[:, None], net / intensity[:, None].clamp_min(1.0e-12), torch.zeros_like(net))
        recovery = torch.cat(
            (
                recovery_step.phase_one_hot_4,
                recovery_step.stagnation_progress[:, None],
                recovery_step.max_escape_progress[:, None],
                recovery_step.timer_fraction[:, None],
            ),
            dim=1,
        ).to(dtype=torch.float32)
        coverage = torch.stack((latest.reachable.coverage_fraction, latest.raw.coverage_fraction), dim=1).to(dtype=torch.float32)
        remaining = torch.full(
            (env.num_envs, 1),
            max(0.0, 1.0 - float(getattr(env, "_formal_step", 0)) / 1200.0),
            device=pose.device,
            dtype=torch.float32,
        )
        parts = (
            pose[:, :3] - origins,
            quaternion_xyzw_to_rotation_6d(pose[:, 3:7]),
            velocity[:, :3],
            velocity[:, 3:6],
            previous,
            torch.stack((valid.to(torch.float32), intensity), dim=1),
            normal,
            valid.to(torch.float32)[:, None],
            coverage,
            runtime.coverage_grid_3x3x3().to(dtype=torch.float32),
            remaining,
            recovery,
        )
        observation = torch.cat(parts, dim=1)
        if observation.shape != (env.num_envs, 65) or not torch.isfinite(observation).all().item():
            raise RuntimeError("TASK-010 Critic observation must be finite [N,65]")
        return observation
