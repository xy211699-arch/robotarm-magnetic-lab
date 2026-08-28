#!/usr/bin/env python3
"""Run one isolated TASK-009D0 Gate 5 throughput measurement process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
TASK_ID = "Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    from isaaclab.app import AppLauncher

    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, required=True, choices=(1, 2, 4, 8, 12))
    parser.add_argument("--repeat_index", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--output_directory", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short 2-warmup/5-measurement candidate smoke test.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.baselines.random_policies import build_policy, load_random_baseline_config
    from robotarm_magnetic_lab.runtime.task009d0_config import TASK009D0_CONFIG_PATH, load_task009d0_config

    output_name = (
        f"task009d0a_smoke_env{args.num_envs}.json"
        if args.smoke
        else f"task009d0_throughput_env{args.num_envs}_repeat{args.repeat_index}.json"
    )
    output = args.output_directory / output_name
    config = load_task009d0_config(TASK009D0_CONFIG_PATH)
    policy_config = load_random_baseline_config(ROOT / "configs/task009c/random_baseline_preexperiment_v1.json")
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs)
    faults: list[str] = []
    measurements: list[dict] = []
    try:
        with launch_simulation(app, cfg):
            env = gym.make(TASK_ID, cfg=cfg).unwrapped
            env.reset(seed=int(config["training_seed"]) + args.repeat_index)
            runtime = env._task009d0_coverage_runtime
            policies = [
                build_policy("R3", int(config["training_seed"]) + 1000 * args.repeat_index + index, policy_config)
                for index in range(args.num_envs)
            ]

            timing = {"physics": 0.0, "rgb": 0.0, "coverage": 0.0}
            original_sim_step = env.sim.step
            original_render = env.sim.render
            original_observe = runtime.rgb_sync.observe
            original_update = runtime.update_boundary

            def timed_sim_step(*call_args, **call_kwargs):
                start = time.perf_counter()
                result = original_sim_step(*call_args, **call_kwargs)
                timing["physics"] += time.perf_counter() - start
                return result

            def timed_render(*call_args, **call_kwargs):
                start = time.perf_counter()
                result = original_render(*call_args, **call_kwargs)
                timing["rgb"] += time.perf_counter() - start
                return result

            def timed_observe(*call_args, **call_kwargs):
                start = time.perf_counter()
                result = original_observe(*call_args, **call_kwargs)
                timing["rgb"] += time.perf_counter() - start
                return result

            def timed_update(*call_args, **call_kwargs):
                rgb_before = timing["rgb"]
                start = time.perf_counter()
                result = original_update(*call_args, **call_kwargs)
                total = time.perf_counter() - start
                timing["coverage"] += max(0.0, total - (timing["rgb"] - rgb_before))
                return result

            env.sim.step = timed_sim_step
            env.sim.render = timed_render
            runtime.rgb_sync.observe = timed_observe
            runtime.update_boundary = timed_update

            warmup_steps = 2 if args.smoke else config["benchmark"]["warmup_steps"]
            measured_steps = 5 if args.smoke else config["benchmark"]["measured_steps"]
            total_steps = warmup_steps + measured_steps
            for step in range(total_steps):
                action = torch.tensor(
                    [policy.act().as_pair() for policy in policies], device=env.device, dtype=torch.float32
                )
                timing.update(physics=0.0, rgb=0.0, coverage=0.0)
                started = time.perf_counter()
                observation, _, terminated, truncated, _ = env.step(action)
                total_s = time.perf_counter() - started
                if torch.any(terminated).item() or torch.any(truncated).item():
                    raise RuntimeError("benchmark episode ended before 350 boundaries")
                if not torch.isfinite(observation["policy"]["rgb"]).all().item():
                    raise RuntimeError("benchmark RGB became non-finite")
                if step < warmup_steps:
                    continue
                free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device("cuda:0"))
                latest = runtime.latest_update
                measurements.append({
                    "boundary_index": step - config["benchmark"]["warmup_steps"],
                    "physics_wall_s": timing["physics"],
                    "rgb_sync_wall_s": timing["rgb"],
                    "coverage_wall_s": timing["coverage"],
                    "total_boundary_wall_s": total_s,
                    "gpu_free_bytes": int(free_bytes),
                    "gpu_total_bytes": int(total_bytes),
                    "gpu_free_fraction": float(free_bytes / total_bytes),
                    "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "candidate_counts": latest.candidate_counts.detach().cpu().tolist(),
                    "ray_count": int(latest.ray_count),
                    "forced_capture": bool(runtime.rgb_sync.last_forced_capture),
                    "frame_ids": latest.frame_ids.detach().cpu().tolist(),
                })
            elapsed = sum(row["total_boundary_wall_s"] for row in measurements)
            manifest = {
                "schema": "robotarm_magnetic_lab.task009d0_throughput_manifest",
                "version": 1,
                "status": "pass",
                "branch": "feature/TASK-009D0-vectorized-training-infrastructure",
                "commit": commit,
                "config_sha256": config["config_sha256"],
                "task_id": TASK_ID,
                "num_envs": args.num_envs,
                "repeat_index": args.repeat_index,
                "smoke": bool(args.smoke),
                "device": str(env.device),
                "clocks": config["clocks"],
                "warmup_steps": warmup_steps,
                "measured_steps": measured_steps,
                "environment_transitions_per_second": args.num_envs * len(measurements) / elapsed,
                "minimum_gpu_free_fraction": min(row["gpu_free_fraction"] for row in measurements),
                "maximum_process_rss_kib": max(row["process_max_rss_kib"] for row in measurements),
                "forced_capture_count": sum(row["forced_capture"] for row in measurements),
                "faults": faults,
                "measurements": measurements,
            }
            _atomic_json(output, manifest)
            print("TASK009D0_GATE5_RUN", json.dumps({
                "status": "pass", "num_envs": args.num_envs, "repeat_index": args.repeat_index,
                "throughput": manifest["environment_transitions_per_second"],
                "minimum_gpu_free_fraction": manifest["minimum_gpu_free_fraction"], "path": str(output.resolve()),
            }, sort_keys=True))
            env.close()
    except Exception as exc:
        faults.append(f"{type(exc).__name__}: {exc}")
        _atomic_json(output, {
            "schema": "robotarm_magnetic_lab.task009d0_throughput_manifest", "version": 1,
            "status": "fail", "branch": "feature/TASK-009D0-vectorized-training-infrastructure",
            "commit": commit, "config_sha256": config["config_sha256"], "task_id": TASK_ID,
            "num_envs": args.num_envs, "repeat_index": args.repeat_index, "device": str(args.device),
            "clocks": config["clocks"], "warmup_steps": 2 if args.smoke else config["benchmark"]["warmup_steps"],
            "measured_steps": 5 if args.smoke else config["benchmark"]["measured_steps"],
            "smoke": bool(args.smoke), "faults": faults,
            "measurements": measurements,
        })
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
