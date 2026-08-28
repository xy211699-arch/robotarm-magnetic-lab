#!/usr/bin/env python3
"""Gate 3: validate two-environment trajectory and row-reset isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
TASK_ID = "Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0"


def validate_isolation_manifest(manifest: dict) -> dict:
    for row in manifest["phase_a"]:
        positions = np.asarray(row["local_positions"], dtype=np.float64)
        quaternions = np.asarray(row["local_quaternions"], dtype=np.float64)
        if np.linalg.norm(positions[0] - positions[1]) > 1.0e-6:
            raise ValueError("equal-action local trajectory diverged")
        alignment = abs(float(np.dot(quaternions[0], quaternions[1])))
        if alignment < 1.0 - 1.0e-6:
            raise ValueError("equal-action quaternion alignment failed")
    for row in manifest["phase_b_env1_replay"]:
        if float(row["position_error_m"]) > 1.0e-6:
            raise ValueError("environment 1 replay trajectory diverged")
        if float(row["quaternion_alignment"]) < 1.0 - 1.0e-6:
            raise ValueError("environment 1 replay quaternion failed")
    reset = manifest["row_reset"]
    checks = (
        ("coverage_mask_hash_before", "coverage_mask_hash_after", "coverage"),
        ("local_pose_before", "local_pose_after", "pose"),
        ("frame_before", "frame_after", "frame"),
        ("episode_index_before", "episode_index_after", "episode"),
        ("previous_action_before", "previous_action_after", "previous action"),
        ("reward_before", "reward_after", "reward"),
    )
    for before, after, label in checks:
        if reset[before] != reset[after]:
            raise ValueError(f"untouched environment {label} changed")
    return {"status": "pass", "phase_a_boundaries": len(manifest["phase_a"])}


def _mask_hash(mask) -> str:
    return hashlib.sha256(mask.detach().cpu().numpy().astype(np.bool_).tobytes()).hexdigest()


def _local_poses(env):
    poses = env.scene["capsule"].data.root_pose_w.torch.detach().clone()
    poses[:, :3] -= env.scene.env_origins
    return poses


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

    args.output_directory.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=2)
    cfg.pose_split = "validation"
    cfg.explicit_pose_ids = ("validation-0006", "validation-0006")
    try:
        with launch_simulation(app, cfg):
            env = gym.make(TASK_ID, cfg=cfg).unwrapped
            phase_a = []
            observation, extras = env.reset(seed=990009)
            initial_poses = _local_poses(env)
            print(
                "TASK009D0_GATE3_INITIAL_DIAGNOSTIC",
                json.dumps(
                    {
                        "local_poses": initial_poses.cpu().tolist(),
                        "position_difference_m": float(
                            torch.linalg.vector_norm(initial_poses[0, :3] - initial_poses[1, :3]).item()
                        ),
                        "quaternion_alignment": abs(
                            float(torch.dot(initial_poses[0, 3:], initial_poses[1, 3:]).item())
                        ),
                        "initial_coverage": extras["task009d0_reset"]["initial_coverage"],
                    },
                    sort_keys=True,
                ),
            )
            initial_position_error = float(
                torch.linalg.vector_norm(initial_poses[0, :3] - initial_poses[1, :3]).item()
            )
            initial_quaternion_alignment = abs(
                float(torch.dot(initial_poses[0, 3:], initial_poses[1, 3:]).item())
            )
            if initial_position_error > 1.0e-6 or initial_quaternion_alignment < 1.0 - 1.0e-6:
                failure = {
                    "status": "fail",
                    "gate": 3,
                    "task_id": TASK_ID,
                    "branch": "feature/TASK-009D0-vectorized-training-infrastructure",
                    "commit": subprocess.check_output(
                        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
                    ).strip(),
                    "reason": "equal-action environments differ after required 1 s HOLD stabilization",
                    "tolerance": {
                        "maximum_local_position_error_m": 1.0e-6,
                        "minimum_quaternion_alignment": 1.0 - 1.0e-6,
                    },
                    "observed": {
                        "local_poses": initial_poses.cpu().tolist(),
                        "local_position_error_m": initial_position_error,
                        "quaternion_alignment": initial_quaternion_alignment,
                        "initial_coverage": extras["task009d0_reset"]["initial_coverage"],
                        "pose_ids": extras["task009d0_reset"]["pose_ids"],
                        "hold_cycles": extras["task009d0_reset"]["hold_cycles"],
                    },
                    "downstream_gates_skipped": [4, 5, 6],
                }
                target = args.output_directory / "task009d0_gate3_two_env_isolation.json"
                target.write_text(
                    json.dumps(failure, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    "TASK009D0_GATE3",
                    json.dumps(
                        {"status": "fail", "path": str(target.resolve())},
                        sort_keys=True,
                    ),
                )
                env.close()
                return
            for boundary in range(100):
                action = torch.tensor([[3.0, 0.5], [3.0, 0.5]], device=env.device)
                observation, reward, terminated, truncated, extras = env.step(action)
                poses = _local_poses(env)
                if torch.any(~torch.isfinite(poses)).item():
                    raise RuntimeError("non-finite phase A pose")
                runtime = env._task009d0_coverage_runtime
                if not torch.equal(runtime.latest_reachable_visible_mask[0], runtime.latest_reachable_visible_mask[1]):
                    difference = runtime.latest_reachable_visible_mask[0] ^ runtime.latest_reachable_visible_mask[1]
                    camera = env.scene["capsule_camera"]
                    print(
                        "TASK009D0_GATE3_VISIBILITY_DIAGNOSTIC",
                        json.dumps(
                            {
                                "boundary": boundary + 1,
                                "different_vertices": int(difference.sum().item()),
                                "visible_counts": runtime.latest_reachable_visible_mask.sum(dim=1).cpu().tolist(),
                                "local_positions": poses[:, :3].cpu().tolist(),
                                "camera_positions_local": (
                                    camera.data.pos_w.torch - env.scene.env_origins
                                ).detach().cpu().tolist(),
                                "camera_quaternions": camera.data.quat_w_ros.torch.detach().cpu().tolist(),
                            },
                            sort_keys=True,
                        ),
                    )
                    raise RuntimeError("equal-action current visibility differs")
                if not torch.equal(runtime.reachable_accumulator.mask[0], runtime.reachable_accumulator.mask[1]):
                    raise RuntimeError("equal-action cumulative coverage differs")
                if not torch.allclose(reward[0], reward[1], atol=1e-12, rtol=0.0):
                    raise RuntimeError("equal-action reward differs")
                phase_a.append({
                    "boundary": boundary + 1,
                    "local_positions": poses[:, :3].cpu().tolist(),
                    "local_quaternions": poses[:, 3:].cpu().tolist(),
                    "frame_ids": runtime.latest_update.frame_ids.cpu().tolist(),
                    "coverage_hashes": [_mask_hash(runtime.reachable_accumulator.mask[row]) for row in range(2)],
                    "rewards": reward.cpu().tolist(),
                })
            validate_isolation_manifest({"phase_a": phase_a, "phase_b_env1_replay": [], "row_reset": {
                "coverage_mask_hash_before":"x","coverage_mask_hash_after":"x","local_pose_before":[],"local_pose_after":[],"frame_before":0,"frame_after":0,"episode_index_before":0,"episode_index_after":0,"previous_action_before":[],"previous_action_after":[],"reward_before":0,"reward_after":0,
            }})
            phase_a_env1 = [row for row in phase_a]

            observation, extras = env.reset(seed=990009)
            replay = []
            last_reward = torch.zeros(2, device=env.device)
            for boundary in range(100):
                divergent_mode = 5.0 if boundary % 2 == 0 else 1.0
                action = torch.tensor([[divergent_mode, 0.75], [3.0, 0.5]], device=env.device)
                observation, last_reward, terminated, truncated, extras = env.step(action)
                pose = _local_poses(env)[1]
                expected = phase_a_env1[boundary]
                expected_pos = torch.tensor(expected["local_positions"][1], device=env.device)
                expected_quat = torch.tensor(expected["local_quaternions"][1], device=env.device)
                replay.append({
                    "boundary": boundary + 1,
                    "position_error_m": float(torch.linalg.vector_norm(pose[:3] - expected_pos).item()),
                    "quaternion_alignment": abs(float(torch.dot(pose[3:], expected_quat).item())),
                })

            runtime = env._task009d0_coverage_runtime
            poses_before = _local_poses(env)
            previous = env.action_manager.get_term("parameterized_force").previous_action_features.clone()
            row_reset = {
                "coverage_mask_hash_before": _mask_hash(runtime.reachable_accumulator.mask[1]),
                "local_pose_before": poses_before[1].cpu().tolist(),
                "frame_before": int(runtime.rgb_sync.latest[1].item()),
                "episode_index_before": int(env._episode_indices[1].item()),
                "previous_action_before": previous[1].cpu().tolist(),
                "reward_before": float(last_reward[1].item()),
            }
            env.reset_rows_for_test(torch.tensor([0], device=env.device))
            poses_after = _local_poses(env)
            previous_after = env.action_manager.get_term("parameterized_force").previous_action_features
            row_reset.update({
                "coverage_mask_hash_after": _mask_hash(runtime.reachable_accumulator.mask[1]),
                "local_pose_after": poses_after[1].cpu().tolist(),
                "frame_after": int(runtime.rgb_sync.latest[1].item()),
                "episode_index_after": int(env._episode_indices[1].item()),
                "previous_action_after": previous_after[1].cpu().tolist(),
                "reward_after": float(last_reward[1].item()),
            })
            manifest = {
                "status": "pass",
                "task_id": TASK_ID,
                "branch": "feature/TASK-009D0-vectorized-training-infrastructure",
                "commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
                "phase_a": phase_a,
                "phase_b_env1_replay": replay,
                "row_reset": row_reset,
            }
            manifest["summary"] = validate_isolation_manifest(manifest)
            target = args.output_directory / "task009d0_gate3_two_env_isolation.json"
            target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print("TASK009D0_GATE3", json.dumps({"status":"pass","path":str(target.resolve())}, sort_keys=True))
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
