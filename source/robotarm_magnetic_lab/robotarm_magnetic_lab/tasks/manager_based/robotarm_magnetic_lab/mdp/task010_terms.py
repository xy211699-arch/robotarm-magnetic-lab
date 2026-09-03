"""Manager terms used only by the TASK-010 training environment."""

from __future__ import annotations

import torch

from robotarm_magnetic_lab.runtime.task010_visual_encoder import FrozenResNet18Encoder
from robotarm_magnetic_lab.runtime.task010_recovery import Task010RecoveryTracker
from robotarm_magnetic_lab.runtime.task010_privileged import Task010PrivilegedBuilder
from robotarm_magnetic_lab.runtime.task010_visual_intervention import (
    VALID_TRAINING_VISUAL_CONDITIONS,
    replace_actor_visual_features,
)

from .task009d0_terms import task009d0_rgb, task009d0_runtime


def task010_visual_encoder(env) -> FrozenResNet18Encoder:
    encoder = getattr(env, "_task010_visual_encoder", None)
    if encoder is None:
        encoder = FrozenResNet18Encoder().to(env.device)
        env._task010_visual_encoder = encoder
    return encoder


def task010_actor_observation(env) -> torch.Tensor:
    rgb = task009d0_rgb(env)
    frame_ids = task009d0_runtime(env).rgb_sync.latest
    features = task010_visual_encoder(env)(rgb, frame_ids)
    condition = getattr(env.cfg, "task010_visual_condition", "normal")
    if condition not in VALID_TRAINING_VISUAL_CONDITIONS:
        raise ValueError(
            "donor and first_frame are validation-only visual conditions; "
            "the training environment accepts only normal or blind"
        )
    if condition == "blind":
        features = torch.zeros_like(features)
    elif condition == "normal":
        features = features.clone()
    previous = env.action_manager.get_term("parameterized_force").previous_action_features
    observation = replace_actor_visual_features(
        torch.cat((features, previous.to(dtype=torch.float32)), dim=1),
        features,
    )
    if observation.shape != (env.num_envs, 519) or not torch.isfinite(observation).all().item():
        raise RuntimeError("TASK-010 Actor observation must be finite [N,519]")
    return observation


def task010_recovery_step(env):
    boundary = int(env.common_step_counter)
    if getattr(env, "_task010_recovery_boundary", None) == boundary:
        return env._task010_recovery_step
    runtime = task009d0_runtime(env)
    update = runtime.update_boundary(
        boundary=boundary,
        stabilizing=bool(getattr(env, "_task009d0_stabilizing", False)),
    )
    tracker = getattr(env, "_task010_recovery_tracker", None)
    if tracker is None:
        tracker = Task010RecoveryTracker(env.num_envs, env.device)
        env._task010_recovery_tracker = tracker
    pose = env.scene["capsule"].data.root_pose_w.torch
    step = tracker.update(
        pose[:, :3], pose[:, 3:7], update.reachable.coverage_fraction, dt_s=0.1
    )
    env._task010_recovery_boundary = boundary
    env._task010_recovery_step = step
    return step


def task010_total_reward(env) -> torch.Tensor:
    step = task010_recovery_step(env)
    # Keep the exact reward decomposition alive across the synchronous
    # horizon reset, whose first observation initializes the next episode.
    env._task010_last_reward_step = step
    return step.total_reward


def task010_privileged_observation(env) -> torch.Tensor:
    if task009d0_runtime(env).latest_update is None:
        return torch.zeros((env.num_envs, 65), device=env.device, dtype=torch.float32)
    builder = getattr(env, "_task010_privileged_builder", None)
    if builder is None:
        builder = Task010PrivilegedBuilder()
        env._task010_privileged_builder = builder
    return builder.build(env, task010_recovery_step(env))
