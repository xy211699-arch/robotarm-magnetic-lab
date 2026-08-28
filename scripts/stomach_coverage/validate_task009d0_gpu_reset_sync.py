#!/usr/bin/env python3
"""Gate 4: validate repeated GPU pose, HOLD, RGB and C0 reset synchronization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
TASK_ID = "Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0"


def main() -> None:
    from isaaclab.app import AppLauncher

    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resets", type=int, default=20)
    parser.add_argument("--output_directory", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.resets < 20:
        parser.error("Gate 4 requires at least 20 synchronized resets")
    args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    args.output_directory.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=2)
    try:
        with launch_simulation(app, cfg):
            env = gym.make(TASK_ID, cfg=cfg).unwrapped
            evidence = []
            for reset_index in range(args.resets):
                observation, extras = env.reset(seed=990009 if reset_index == 0 else None)
                info = extras["task009d0_reset"]
                trace = info["hold_cycles"]
                if len(trace) != 10:
                    raise RuntimeError("reset did not execute ten HOLD boundaries")
                frames = [row["frame_ids"] for row in trace]
                for before, after in zip(frames, frames[1:], strict=False):
                    if any(int(b) + 1 != int(a) for b, a in zip(before, after, strict=True)):
                        raise RuntimeError("HOLD frame vectors did not increment exactly once")
                if len(set(info["pose_ids"])) != 2:
                    raise RuntimeError("Gate 4 requires two distinct train poses")
                rgb = observation["policy"]["rgb"]
                if not torch.isfinite(rgb).all().item():
                    raise RuntimeError("reset RGB is non-finite")
                runtime = env._task009d0_coverage_runtime
                c0 = runtime.latest_update.reachable.coverage_fraction
                if torch.any(c0 <= 0).item():
                    raise RuntimeError("reset C0 must be positive in both rows")
                if torch.any(env.episode_length_buf != 0).item():
                    raise RuntimeError("episode length was not cleared after HOLD")
                term = env.action_manager.get_term("parameterized_force")
                if torch.any(term.previous_action_features != 0).item():
                    raise RuntimeError("previous action was not cleared after reset")
                composer = env.scene["capsule"].permanent_wrench_composer
                if torch.any(composer.out_force_b.torch != 0).item() or torch.any(
                    composer.out_torque_b.torch != 0
                ).item():
                    raise RuntimeError("wrench composer retained force after reset")
                devices = {
                    "environment": str(env.device),
                    "physics": str(env.scene["capsule"].root_view._backend.device),
                    "camera": str(rgb.device),
                    "coverage": str(runtime.vertices_local.device),
                }
                if any("cuda:0" not in value for value in devices.values()):
                    raise RuntimeError(f"Gate 4 device mismatch: {devices}")
                evidence.append({
                    "reset_index": reset_index,
                    "pose_ids": info["pose_ids"],
                    "hold_frame_vectors": frames,
                    "C0": c0.detach().cpu().tolist(),
                    "episode_length": env.episode_length_buf.cpu().tolist(),
                    "devices": devices,
                    "rgb_finite": True,
                    "composer_zero": True,
                })
            manifest = {
                "status": "pass",
                "task_id": TASK_ID,
                "branch": "feature/TASK-009D0-vectorized-training-infrastructure",
                "commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
                "reset_count": len(evidence),
                "zero_retries": True,
                "partial_camera_advancement": 0,
                "resets": evidence,
            }
            target = args.output_directory / "task009d0_gate4_gpu_reset_sync.json"
            target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print("TASK009D0_GATE4", json.dumps({"status":"pass","resets":len(evidence),"path":str(target.resolve())}, sort_keys=True))
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
