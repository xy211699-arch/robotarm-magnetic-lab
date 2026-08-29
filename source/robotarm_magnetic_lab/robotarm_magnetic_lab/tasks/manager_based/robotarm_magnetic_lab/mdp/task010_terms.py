"""Manager terms used only by the TASK-010 training environment."""

from __future__ import annotations

import torch

from robotarm_magnetic_lab.runtime.task010_visual_encoder import FrozenResNet18Encoder

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
    previous = env.action_manager.get_term("parameterized_force").previous_action_features
    observation = torch.cat((features, previous.to(dtype=torch.float32)), dim=1)
    if observation.shape != (env.num_envs, 519) or not torch.isfinite(observation).all().item():
        raise RuntimeError("TASK-010 Actor observation must be finite [N,519]")
    return observation
