"""Manager terms for the synchronous TASK-009D0 vector environment."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from robotarm_magnetic_lab.runtime.task009d0_config import (
    TASK009D0_CONFIG_PATH,
    load_task009d0_config,
)
from robotarm_magnetic_lab.runtime.task009d0_coverage_runtime import (
    Task009D0CoverageRuntime,
)

from .vision import capsule_rgb


def task009d0_runtime(env) -> Task009D0CoverageRuntime:
    runtime = getattr(env, "_task009d0_coverage_runtime", None)
    if runtime is None:
        config = load_task009d0_config(TASK009D0_CONFIG_PATH)
        root = TASK009D0_CONFIG_PATH.parents[2]
        runtime = Task009D0CoverageRuntime.from_environment(
            env,
            unreachable_region_path=root / config["unreachable_region"]["path"],
        )
        env._task009d0_coverage_runtime = runtime
    return runtime


def task009d0_rgb(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("capsule_camera"),
    field_of_view_deg: float = 120.0,
) -> torch.Tensor:
    runtime = task009d0_runtime(env)
    camera = env.scene.sensors[sensor_cfg.name]
    _ = camera.data.output["rgb"]
    runtime.rgb_sync.observe(int(env.common_step_counter), camera)
    return capsule_rgb(
        env,
        sensor_cfg=sensor_cfg,
        field_of_view_deg=field_of_view_deg,
        require_new_control_boundary_frame=False,
    )


def task009d0_previous_action(env) -> torch.Tensor:
    term = env.action_manager.get_term("parameterized_force")
    return term.previous_action_features


def task009d0_privileged_capsule_state(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("capsule")
) -> torch.Tensor:
    capsule = env.scene[asset_cfg.name]
    return torch.cat(
        (capsule.data.root_pose_w.torch, capsule.data.root_com_vel_w.torch), dim=1
    )


def task009d0_privileged_coverage(env) -> torch.Tensor:
    runtime = task009d0_runtime(env)
    latest = runtime.latest_update
    if latest is None:
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    return torch.stack(
        (latest.reachable.coverage_fraction, latest.raw.coverage_fraction), dim=1
    ).to(dtype=torch.float32)


def task009d0_new_coverage(env) -> torch.Tensor:
    runtime = task009d0_runtime(env)
    update = runtime.update_boundary(
        boundary=int(env.common_step_counter),
        stabilizing=bool(getattr(env, "_task009d0_stabilizing", False)),
    )
    return (
        update.new_coverage_reward_m2 / runtime.reachable_accumulator.total_area_m2
    ).to(dtype=torch.float32)
