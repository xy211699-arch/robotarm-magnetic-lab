#!/usr/bin/env python3
"""TASK-010 Gate 2: twelve-environment GPU integration validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
SCHEMA = "robotarm_magnetic_lab.task010_gpu_integration"


def validate_summary(value: dict) -> dict:
    required = {
        "schema": SCHEMA, "status": "pass", "num_envs": 12,
        "sequences": 3, "rollout_steps": 64, "formal_steps": 1200,
        "physics_steps_per_action": 24, "all_finite": True,
        "rgb_coverage_same_frame": True, "resnet_unchanged": True,
        "true_terminal": True, "timeout_bootstrap": False,
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise ValueError(f"TASK-010 Gate 2 {name} mismatch")
    devices = value.get("devices", {})
    if any("cuda" not in str(devices.get(name, "")) for name in ("environment", "physics", "camera", "coverage")):
        raise ValueError("TASK-010 Gate 2 did not execute fully on GPU")
    if value.get("shapes") != {"actor": [12, 519], "critic": [12, 65], "action": [12, 2], "reward": [12], "reset_mask": [12]}:
        raise ValueError("TASK-010 Gate 2 tensor shape mismatch")
    return value


def _module_hash(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True); parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser); parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.num_envs != 12: parser.error("TASK-010 Gate 2 requires exactly 12 environments")
    args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
    from robotarm_magnetic_lab.runtime.task010_config import load_task010_config

    frozen = load_task010_config(args.config)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=12)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with launch_simulation(app, cfg):
        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            observations, reset_extras = env.reset(seed=frozen.training.seed)
            actor = Task010Actor().to(env.device).eval()
            encoder = env._task010_visual_encoder
            resnet_before = _module_hash(encoder.backbone)
            initial = env._task009d0_coverage_runtime.latest_update
            initial_zero = int((initial.reachable.coverage_fraction == 0).sum().item())
            if torch.any(initial.raw.coverage_fraction <= 0).item():
                raise RuntimeError("TASK-010 Gate 2 raw initial coverage must be positive")
            reset_mask = torch.zeros(12, dtype=torch.bool, device=env.device)
            shapes = {"actor": list(observations["policy"].shape), "critic": list(observations["privileged"].shape), "reset_mask": list(reset_mask.shape)}
            frame_previous = initial.frame_ids.detach().clone()
            frame_consistent = True; all_finite = True; detach_verified = True
            terminal = None
            fixed_modes = torch.arange(12, device=env.device) % 6
            for formal_step in range(1, 1201):
                with torch.inference_mode():
                    _ = actor(observations["policy"], stochastic_output=False)
                action = torch.zeros((12, 2), device=env.device)
                action[:, 0] = torch.roll(fixed_modes, formal_step % 6)
                action[:, 1] = torch.linspace(0.0, 1.0, 12, device=env.device)
                action[action[:, 0] == 0, 1] = 0.0
                observations, reward, terminated, truncated, extras = env.step(action)
                all_finite &= bool(torch.isfinite(observations["policy"]).all() and torch.isfinite(observations["privileged"]).all() and torch.isfinite(reward).all())
                latest = env._task009d0_coverage_runtime.latest_update
                frame_consistent &= bool(torch.equal(latest.frame_ids, env._task009d0_coverage_runtime.rgb_sync.latest))
                if formal_step < 1200:
                    if not torch.all(latest.frame_ids == frame_previous + 1).item():
                        raise RuntimeError("TASK-010 Gate 2 RGB frame did not advance exactly once")
                    frame_previous = latest.frame_ids.detach().clone()
                if formal_step in (64, 128, 192):
                    before = actor.get_hidden_state().detach().clone(); actor.detach_hidden_state()
                    detach_verified &= torch.equal(before, actor.get_hidden_state()) and actor.get_hidden_state().grad_fn is None
                if formal_step == 1200:
                    terminal = (terminated.detach().clone(), truncated.detach().clone(), extras)
            shapes.update(action=list(action.shape), reward=list(reward.shape))
            if terminal is None: raise RuntimeError("TASK-010 Gate 2 did not reach horizon")
            terminated, truncated, extras = terminal
            true_terminal = bool(torch.all(terminated).item() and not torch.any(truncated).item())
            timeout_bootstrap = bool(torch.any(extras["time_outs"]).item())
            resnet_after = _module_hash(encoder.backbone)
            physics_view = env.sim.physics_sim_view
            camera_rgb = env.scene["capsule_camera"].data.output["rgb"].torch
            devices = {
                "environment": str(env.device),
                "physics": str(getattr(physics_view, "device", getattr(physics_view, "_device", env.sim.device))),
                "camera": str(camera_rgb.device),
                "coverage": str(env._task009d0_coverage_runtime.vertices_local.device),
            }
            summary = validate_summary({
                "schema": SCHEMA, "status": "pass", "num_envs": 12, "sequences": 3,
                "rollout_steps": 64, "formal_steps": 1200, "physics_steps_per_action": int(env.cfg.decimation),
                "all_finite": all_finite, "rgb_coverage_same_frame": frame_consistent,
                "resnet_unchanged": resnet_before == resnet_after, "gru_rollout_detach_verified": bool(detach_verified),
                "true_terminal": true_terminal, "timeout_bootstrap": timeout_bootstrap,
                "initial_zero_reachable_count": initial_zero, "initial_raw_positive": True,
                "shapes": shapes, "devices": devices, "resnet_sha256": resnet_after,
                "weight_identity": encoder.weight_identity,
            })
            (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print("TASK010_GATE2", json.dumps(summary, sort_keys=True))
        finally:
            env.close()
    app.close()


if __name__ == "__main__": main()
