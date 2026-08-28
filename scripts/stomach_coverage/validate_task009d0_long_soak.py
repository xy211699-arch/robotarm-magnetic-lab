#!/usr/bin/env python3
"""Gate 6: run two exact 120-second TASK-009D0 vector episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
TASK_ID = "Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0"
SCHEMA = "robotarm_magnetic_lab.task009d0_long_soak"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask_hash(mask) -> str:
    data = mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _threshold_times(values: list[float], thresholds: list[float]) -> dict[str, float | None]:
    result = {}
    for threshold in thresholds:
        result[str(threshold)] = next(
            (index * 0.1 for index, value in enumerate(values) if value >= threshold),
            None,
        )
    return result


def validate_soak(record: dict[str, Any]) -> dict[str, Any]:
    """Strictly validate an already generated Gate 6 manifest."""
    if record.get("schema") != SCHEMA or int(record.get("version", 0)) != 1:
        raise ValueError("long-soak schema/version mismatch")
    if record.get("clocks") != {
        "physics_hz": 240,
        "control_hz": 10,
        "physics_steps_per_action": 24,
    }:
        raise ValueError("long-soak clock mismatch")
    selected = int(record.get("selected_num_envs", 0))
    episodes = record.get("episodes", [])
    if len(episodes) != 2:
        raise ValueError("long soak requires exactly two episodes")
    prior_final_hashes = None
    for episode_index, episode in enumerate(episodes):
        if int(episode.get("episode_index", -1)) != episode_index:
            raise ValueError("episode indices are not consecutive")
        if int(episode.get("formal_steps", -1)) != 1200:
            raise ValueError("episode requires 1200 formal steps")
        if int(episode.get("formal_physics_substeps", -1)) != 28800:
            raise ValueError("episode requires 28800 formal physics substeps")
        if int(episode.get("inter_episode_hold_substeps", -1)) != 240:
            raise ValueError("episode reset requires 240 HOLD substeps")
        if episode.get("post_reset_episode_length") != [0] * selected:
            raise ValueError("episode length was not zero after reset")
        if not bool(episode.get("terminal_observation_present", False)):
            raise ValueError("terminal observation is missing")
        envs = episode.get("envs", [])
        if len(envs) != selected:
            raise ValueError("episode environment count mismatch")
        initial_hashes = []
        final_hashes = []
        for env_id, row in enumerate(envs):
            if int(row.get("env_id", -1)) != env_id:
                raise ValueError("environment rows are not ordered")
            if int(row.get("coverage_points", -1)) != 1201:
                raise ValueError("each environment requires 1201 coverage points")
            reachable = np.asarray(row.get("reachable_coverage", []), dtype=np.float64)
            raw = np.asarray(row.get("raw_coverage", []), dtype=np.float64)
            frames = np.asarray(row.get("frame_ids", []), dtype=np.int64)
            if reachable.shape != (1201,) or raw.shape != (1201,) or frames.shape != (1201,):
                raise ValueError("coverage and frame vectors must contain 1201 points")
            if not np.isfinite(reachable).all() or not np.isfinite(raw).all():
                raise ValueError("coverage contains non-finite values")
            if np.any(np.diff(reachable) < -1.0e-15) or np.any(np.diff(raw) < -1.0e-15):
                raise ValueError("cumulative coverage decreased")
            if np.any(np.diff(frames) != 1):
                raise ValueError("RGB frames are not unique consecutive boundaries")
            if not bool(row.get("state_finite")) or not bool(row.get("rgb_finite")):
                raise ValueError("episode contains non-finite state or RGB")
            initial_hashes.append(str(row.get("initial_mask_sha256", "")))
            final_hashes.append(str(row.get("final_mask_sha256", "")))
        if prior_final_hashes is not None and any(
            before == after for before, after in zip(prior_final_hashes, initial_hashes, strict=True)
        ):
            raise ValueError("second episode inherited a final coverage mask")
        prior_final_hashes = final_hashes
    if record.get("faults"):
        raise ValueError("long-soak manifest contains faults")
    return record


def main() -> None:
    from isaaclab.app import AppLauncher

    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_directory", type=Path, required=True)
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
    from robotarm_magnetic_lab.runtime.task009d0_config import load_task009d0_config

    frozen_path = ROOT / "configs/task009d0/vectorized_training_frozen_v1.json"
    config = load_task009d0_config(frozen_path, frozen=True)
    policy_config = load_random_baseline_config(
        ROOT / "configs/task009c/random_baseline_preexperiment_v1.json"
    )
    selected = int(config["selected_num_envs"])
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=selected)
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    faults: list[str] = []
    episodes: list[dict[str, Any]] = []
    manifest_path = args.output_directory / "task009d0_gate6_long_soak.json"
    try:
        with launch_simulation(app, cfg):
            env = gym.make(TASK_ID, cfg=cfg).unwrapped
            observation, reset_extras = env.reset(seed=int(config["training_seed"]))
            for episode_index in range(2):
                runtime = env._task009d0_coverage_runtime
                initial = runtime.latest_update
                if initial is None or torch.any(initial.reachable.coverage_fraction <= 0).item():
                    raise RuntimeError("episode initial C0 is missing or non-positive")
                policies = [
                    build_policy(
                        "R3",
                        int(config["training_seed"]) + episode_index * 10000 + env_id,
                        policy_config,
                    )
                    for env_id in range(selected)
                ]
                episode_pose_ids = list(env._last_pose_batch.pose_ids)
                reachable = [[float(initial.reachable.coverage_fraction[i].item())] for i in range(selected)]
                raw = [[float(initial.raw.coverage_fraction[i].item())] for i in range(selected)]
                frames = [[int(initial.frame_ids[i].item())] for i in range(selected)]
                initial_hashes = [
                    _mask_hash(runtime.reachable_accumulator.mask[i]) for i in range(selected)
                ]
                mode_counts = [
                    {name: 0 for name in ("HOLD", "MOVE_POS", "MOVE_NEG", "VIEW_POS", "VIEW_NEG", "UP")}
                    for _ in range(selected)
                ]
                state_finite = [True] * selected
                rgb_finite = [True] * selected
                terminal_audit = None
                terminal_present = False
                for formal_step in range(1, 1201):
                    actions = [policy.act() for policy in policies]
                    for env_id, action in enumerate(actions):
                        mode_counts[env_id][action.mode.name] += 1
                    action_tensor = torch.tensor(
                        [action.as_pair() for action in actions],
                        device=env.device,
                        dtype=torch.float32,
                    )
                    observation, _, terminated, truncated, extras = env.step(action_tensor)
                    if torch.any(terminated).item():
                        raise RuntimeError(f"episode {episode_index} terminated at {formal_step}")
                    if formal_step < 1200:
                        if torch.any(truncated).item():
                            raise RuntimeError(f"episode {episode_index} truncated early at {formal_step}")
                        update = runtime.latest_update
                        root_pose = env.scene["capsule"].data.root_pose_w.torch
                        root_velocity = env.scene["capsule"].data.root_com_vel_w.torch
                        rgb = observation["policy"]["rgb"]
                    else:
                        if not torch.all(truncated).item():
                            raise RuntimeError("episode did not truncate at the synchronous horizon")
                        terminal_audit = extras.get("task009d0_terminal_audit")
                        terminal_observation = extras.get("terminal_observation")
                        terminal_present = terminal_audit is not None and terminal_observation is not None
                        if not terminal_present:
                            raise RuntimeError("terminal audit or observation is missing")
                        update = None
                        root_pose = terminal_audit["root_pose"]
                        root_velocity = terminal_audit["root_velocity"]
                        rgb = terminal_observation["policy"]["rgb"]
                    finite_rows = torch.isfinite(root_pose).all(dim=1) & torch.isfinite(root_velocity).all(dim=1)
                    rgb_rows = torch.isfinite(rgb.reshape(selected, -1)).all(dim=1)
                    for env_id in range(selected):
                        state_finite[env_id] &= bool(finite_rows[env_id].item())
                        rgb_finite[env_id] &= bool(rgb_rows[env_id].item())
                        if update is None:
                            reachable[env_id].append(float(terminal_audit["reachable_coverage"][env_id].item()))
                            raw[env_id].append(float(terminal_audit["raw_coverage"][env_id].item()))
                            frames[env_id].append(int(terminal_audit["frame_ids"][env_id].item()))
                        else:
                            reachable[env_id].append(float(update.reachable.coverage_fraction[env_id].item()))
                            raw[env_id].append(float(update.raw.coverage_fraction[env_id].item()))
                            frames[env_id].append(int(update.frame_ids[env_id].item()))
                assert terminal_audit is not None
                final_hashes = [
                    _mask_hash(terminal_audit["reachable_masks"][i]) for i in range(selected)
                ]
                env_rows = []
                for env_id in range(selected):
                    values = reachable[env_id]
                    env_rows.append({
                        "env_id": env_id,
                        "pose_id": episode_pose_ids[env_id],
                        "coverage_points": len(values),
                        "reachable_coverage": values,
                        "raw_coverage": raw[env_id],
                        "frame_ids": frames[env_id],
                        "C0": values[0],
                        "C_final": values[-1],
                        "normalized_auc": float(np.trapezoid(values, dx=0.1) / 120.0),
                        "threshold_times_s": _threshold_times(values, config["episode"]["thresholds"]),
                        "initial_mask_sha256": initial_hashes[env_id],
                        "final_mask_sha256": final_hashes[env_id],
                        "mode_counts": mode_counts[env_id],
                        "mode_fractions": {
                            name: count / 1200 for name, count in mode_counts[env_id].items()
                        },
                        "state_finite": state_finite[env_id],
                        "rgb_finite": rgb_finite[env_id],
                    })
                reset_info = extras["task009d0_reset"]
                composer = env.scene["capsule"].permanent_wrench_composer
                episodes.append({
                    "episode_index": episode_index,
                    "formal_steps": 1200,
                    "formal_physics_substeps": 28800,
                    "inter_episode_hold_substeps": sum(
                        int(row["physics_substeps"]) for row in reset_info["hold_cycles"]
                    ),
                    "post_reset_episode_length": env.episode_length_buf.detach().cpu().tolist(),
                    "post_reset_previous_action_zero": bool(
                        torch.all(env.action_manager.get_term("parameterized_force").previous_action_features == 0).item()
                    ),
                    "post_reset_actor_force_zero": bool(
                        torch.all(composer.out_force_b.torch == 0).item()
                        and torch.all(composer.out_torque_b.torch == 0).item()
                    ),
                    "terminal_observation_present": terminal_present,
                    "reset_pose_ids": list(reset_info["pose_ids"]),
                    "envs": env_rows,
                })
            manifest = {
                "schema": SCHEMA,
                "version": 1,
                "status": "pass",
                "branch": "feature/TASK-009D0-vectorized-training-infrastructure",
                "commit": commit,
                "config_sha256": config["config_sha256"],
                "selected_num_envs": selected,
                "device": str(env.device),
                "clocks": config["clocks"],
                "episodes": episodes,
                "faults": faults,
            }
            validate_soak(manifest)
            _atomic_json(manifest_path, manifest)
            inventory_path = args.output_directory / "artifact_inventory.json"
            _atomic_json(inventory_path, {
                "files": [{
                    "path": str(manifest_path.resolve()),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": _sha256(manifest_path),
                }]
            })
            print("TASK009D0_GATE6", json.dumps({
                "status": "pass",
                "selected_num_envs": selected,
                "episodes": len(episodes),
                "path": str(manifest_path.resolve()),
            }, sort_keys=True))
            env.close()
    except Exception as exc:
        faults.append(f"{type(exc).__name__}: {exc}")
        _atomic_json(manifest_path, {
            "schema": SCHEMA,
            "version": 1,
            "status": "fail",
            "branch": "feature/TASK-009D0-vectorized-training-infrastructure",
            "commit": commit,
            "config_sha256": config["config_sha256"],
            "selected_num_envs": selected,
            "device": str(args.device),
            "clocks": config["clocks"],
            "episodes": episodes,
            "faults": faults,
        })
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
