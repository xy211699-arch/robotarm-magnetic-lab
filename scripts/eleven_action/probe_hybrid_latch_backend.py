#!/usr/bin/env python3
"""TASK-006 CUDA gate for the dynamic six-DOF rigid-body latch backend."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
TASK_ID = "Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0"


def _axis_from_xyzw(quaternion_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    axis = rotation @ np.asarray([0.0, 0.0, -1.0])
    return axis / np.linalg.norm(axis)


def _pose(capsule) -> tuple[np.ndarray, np.ndarray]:
    pose = capsule.data.root_link_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    return pose[:3].copy(), _axis_from_xyzw(pose[3:7])


def _velocity(capsule) -> tuple[np.ndarray, np.ndarray]:
    value = capsule.data.root_com_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
    return value[:3].copy(), value[3:6].copy()


def _angle_deg(first, second) -> float:
    return math.degrees(math.acos(np.clip(float(np.asarray(first) @ np.asarray(second)), -1.0, 1.0)))


def _jsonable_readback(value) -> dict:
    return {
        "backend": value.backend.value,
        "latched": bool(value.latched),
        "position_world_m": value.position_world_m.tolist(),
        "quaternion_wxyz": value.quaternion_wxyz.tolist(),
        "linear_velocity_world_m_s": value.linear_velocity_world_m_s.tolist(),
        "angular_velocity_world_rad_s": value.angular_velocity_world_rad_s.tolist(),
        "locked_position_axis_mask": int(value.locked_position_axis_mask),
        "locked_rotation_axis_mask": int(value.locked_rotation_axis_mask),
        "kinematic_enabled": bool(value.kinematic_enabled),
        "simulation_disabled": bool(value.simulation_disabled),
        "reason": None if value.reason is None else value.reason.value,
    }


def _run_prefix(env, term, action_id: int, steps: int = 3) -> list[dict]:
    import torch

    request = torch.tensor([[float(action_id)]], device=env.unwrapped.device)
    idle = torch.full((1, 1), -1.0, device=env.unwrapped.device)
    rows = []
    for index in range(steps):
        env.step(request if index == 0 else idle)
        position, axis = _pose(term.capsule)
        linear, angular = _velocity(term.capsule)
        rows.append(
            {
                "environment_step": index,
                "position_world_m": position.tolist(),
                "target_axis_world": axis.tolist(),
                "linear_velocity_world_m_s": linear.tolist(),
                "angular_velocity_world_rad_s": angular.tolist(),
            }
        )
    return rows


def _restore_trial_state(env, profile, root_pose):
    """Restore an identical cold-start state for one paired branch."""
    import torch
    from calibrate_eleven_action import set_runtime_profile

    env.reset()
    term = env.unwrapped.action_manager.get_term("eleven_action")
    set_runtime_profile(term, profile)
    capsule = term.capsule
    capsule.write_root_pose_to_sim_index(root_pose=root_pose.clone())
    capsule.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=env.unwrapped.device)
    )
    return term


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument(
        "--backend",
        choices=("dynamic_lock_flags", "tensor_disable_simulation"),
        default="dynamic_lock_flags",
    )
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--output_directory",
        type=Path,
        default=ROOT / "logs/hybrid_latched_task006/backend_probe",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    import gymnasium as gym
    import torch
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from pxr import PhysxSchema, UsdPhysics
    import omni.usd
    from calibrate_eleven_action import reset_flat_trial
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
        LatchBackendName,
        LatchReason,
        LatchedContactSnapshot,
        dynamic_profile_sha256,
        latch_profile_sha256,
        load_dynamic_profile,
    )
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.eleven_action_latch import CapsuleLatchRuntime

    output = args.output_directory / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    summary = {
        "schema_version": "task006_hybrid_latch_backend_probe_v2",
        "status": "fail",
        "backend": args.backend,
        "seed": args.seed,
        "task": args.task,
        "requested_device": args.device,
        "gpu_dynamics": True,
        "ccd_warning": "GPU dynamics active; sweep CCD is disabled and was not changed by TASK-006",
        "dynamic_profile_sha256": dynamic_profile_sha256(),
        "latch_profile_sha256": latch_profile_sha256(),
        "hold_trials": 0,
        "release_pairs": 0,
        "failures": [],
    }
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.sim.render_interval = 240
    env_cfg.observations.policy.rgb = None
    env_cfg.scene.capsule_camera = None
    rng = np.random.default_rng(args.seed)
    profile = load_dynamic_profile()
    env = None
    runtime = None
    try:
        with launch_simulation(env_cfg, args):
            env = gym.make(args.task, cfg=env_cfg)
            term = env.unwrapped.action_manager.get_term("eleven_action")
            capsule = term.capsule
            prim_path = capsule.root_view.prim_paths[0]
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
            physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not physx_api:
                physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if args.backend == "tensor_disable_simulation":
                runtime = CapsuleLatchRuntime.tensor_disable_simulation(capsule, physx_api, rigid_api)
            else:
                runtime = CapsuleLatchRuntime.dynamic_lock_flags(capsule, physx_api, rigid_api)
            summary["capsule_prim_path"] = prim_path
            summary["cuda_device"] = str(env.unwrapped.device)
            idle = torch.full((1, 1), -1.0, device=env.unwrapped.device)

            for trial in range(10):
                reset_flat_trial(
                    env,
                    profile,
                    tilt_deg=float(np.linspace(0.0, 90.0, 10)[trial]),
                    azimuth_deg=float((trial * 137.5) % 360.0),
                    roll_deg=float((trial * 71.0) % 360.0),
                    settle_steps=12,
                )
                capsule.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((1, 6), device=env.unwrapped.device)
                )
                before_position, before_axis = _pose(capsule)
                before_api_position = before_position.copy()
                lock = runtime.lock_current(term.controller, LatchReason.INITIAL)
                after_api_position, after_api_axis = _pose(capsule)
                for _ in range(60):
                    env.step(idle)
                held_position, held_axis = _pose(capsule)
                linear, angular = _velocity(capsule)
                unlock = runtime.unlock_zeroed(term.controller)
                unlock_position, unlock_axis = _pose(capsule)
                row = {
                    "kind": "hold",
                    "trial": trial,
                    "lock_readback": _jsonable_readback(lock),
                    "unlock_readback": _jsonable_readback(unlock),
                    "api_lock_position_jump_m": float(np.linalg.norm(after_api_position - before_api_position)),
                    "api_lock_axis_jump_deg": _angle_deg(after_api_axis, before_axis),
                    "locked_position_drift_m": float(np.linalg.norm(held_position - after_api_position)),
                    "locked_axis_drift_deg": _angle_deg(held_axis, after_api_axis),
                    "locked_linear_speed_m_s": float(np.linalg.norm(linear)),
                    "locked_angular_speed_rad_s": float(np.linalg.norm(angular)),
                    "api_unlock_position_jump_m": float(np.linalg.norm(unlock_position - held_position)),
                    "api_unlock_axis_jump_deg": _angle_deg(unlock_axis, held_axis),
                }
                rows.append(row)
                summary["hold_trials"] += 1

            for action_id in range(1, 11):
                for trial in range(10):
                    tilt = float(rng.uniform(62.0, 88.0) if action_id >= 9 else rng.uniform(5.0, 85.0))
                    azimuth = float(rng.uniform(0.0, 360.0))
                    roll = float(rng.uniform(0.0, 360.0))
                    sampled_term = reset_flat_trial(
                        env, profile, tilt_deg=tilt, azimuth_deg=azimuth, roll_deg=roll, settle_steps=12
                    )
                    paired_root_pose = sampled_term.capsule.data.root_pose_w.torch.clone()

                    term = _restore_trial_state(env, profile, paired_root_pose)
                    capsule = term.capsule
                    direct_start_position, direct_start_axis = _pose(capsule)
                    if action_id >= 9:
                        term.controller.set_latched_contact_snapshot(
                            LatchedContactSnapshot(True, False, True, term._physics_substep)
                        )
                    direct = _run_prefix(env, term, action_id)

                    term = _restore_trial_state(env, profile, paired_root_pose)
                    capsule = term.capsule
                    latched_start_position, latched_start_axis = _pose(capsule)
                    lock = runtime.lock_current(term.controller, LatchReason.INITIAL)
                    for _ in range(60):
                        env.step(idle)
                    unlock = runtime.unlock_zeroed(term.controller)
                    if action_id >= 9:
                        term.controller.set_latched_contact_snapshot(
                            LatchedContactSnapshot(True, False, True, term._physics_substep)
                        )
                    latched = _run_prefix(env, term, action_id)
                    position_delta = max(
                        float(np.linalg.norm(
                            np.asarray(a["position_world_m"]) - np.asarray(b["position_world_m"])
                        ))
                        for a, b in zip(latched, direct, strict=True)
                    )
                    axis_delta = max(
                        _angle_deg(a["target_axis_world"], b["target_axis_world"])
                        for a, b in zip(latched, direct, strict=True)
                    )
                    rows.append(
                        {
                            "kind": "release_pair",
                            "action_id": action_id,
                            "trial": trial,
                            "tilt_deg": tilt,
                            "azimuth_deg": azimuth,
                            "roll_deg": roll,
                            "lock_readback": _jsonable_readback(lock),
                            "unlock_readback": _jsonable_readback(unlock),
                            "initial_position_delta_m": float(
                                np.linalg.norm(latched_start_position - direct_start_position)
                            ),
                            "initial_target_axis_delta_deg": _angle_deg(
                                latched_start_axis, direct_start_axis
                            ),
                            "direct_first_0p05_s": direct,
                            "latched_first_0p05_s": latched,
                            "max_position_delta_m": position_delta,
                            "max_target_axis_delta_deg": axis_delta,
                        }
                    )
                    summary["release_pairs"] += 1

            hold_rows = [row for row in rows if row["kind"] == "hold"]
            pair_rows = [row for row in rows if row["kind"] == "release_pair"]
            summary["max_api_pose_jump_m"] = max(
                max(row["api_lock_position_jump_m"], row["api_unlock_position_jump_m"])
                for row in hold_rows
            )
            summary["max_locked_position_drift_m"] = max(row["locked_position_drift_m"] for row in hold_rows)
            summary["max_locked_axis_drift_deg"] = max(row["locked_axis_drift_deg"] for row in hold_rows)
            summary["max_locked_linear_speed_m_s"] = max(row["locked_linear_speed_m_s"] for row in hold_rows)
            summary["max_locked_angular_speed_rad_s"] = max(row["locked_angular_speed_rad_s"] for row in hold_rows)
            summary["max_release_position_delta_m"] = max(row["max_position_delta_m"] for row in pair_rows)
            summary["max_release_axis_delta_deg"] = max(row["max_target_axis_delta_deg"] for row in pair_rows)
            summary["max_initial_position_delta_m"] = max(
                row["initial_position_delta_m"] for row in pair_rows
            )
            summary["max_initial_axis_delta_deg"] = max(
                row["initial_target_axis_delta_deg"] for row in pair_rows
            )
            if args.backend == "tensor_disable_simulation":
                readback_ok = all(
                    row["lock_readback"]["simulation_disabled"]
                    and not row["unlock_readback"]["simulation_disabled"]
                    for row in rows
                )
                summary["simulation_disable_readback_pass"] = readback_ok
            else:
                readback_ok = all(
                    row["lock_readback"]["locked_position_axis_mask"] == 0b111
                    and row["lock_readback"]["locked_rotation_axis_mask"] == 0b111
                    and row["unlock_readback"]["locked_position_axis_mask"] == 0
                    and row["unlock_readback"]["locked_rotation_axis_mask"] == 0
                    for row in rows
                )
                summary["mask_readback_pass"] = readback_ok
            checks = {
                "backend_readback": readback_ok,
                "api_pose_jump": summary["max_api_pose_jump_m"] <= 1.0e-9,
                "locked_position_drift": summary["max_locked_position_drift_m"] <= 1.0e-7,
                "locked_axis_drift": summary["max_locked_axis_drift_deg"] <= 1.0e-4,
                "locked_zero_linear_velocity": summary["max_locked_linear_speed_m_s"] <= 1.0e-7,
                "locked_zero_angular_velocity": summary["max_locked_angular_speed_rad_s"] <= 1.0e-7,
                "paired_identical_initial_position": summary["max_initial_position_delta_m"] <= 1.0e-9,
                "paired_identical_initial_axis": summary["max_initial_axis_delta_deg"] <= 1.0e-4,
                "paired_position": summary["max_release_position_delta_m"] <= 0.0005,
                "paired_axis": summary["max_release_axis_delta_deg"] <= 1.0,
                "trial_counts": summary["hold_trials"] >= 10 and summary["release_pairs"] >= 100,
            }
            summary["checks"] = checks
            summary["failures"] = [name for name, passed in checks.items() if not passed]
            summary["status"] = "pass" if all(checks.values()) else "fail"
    except Exception as error:
        summary["failures"].append(f"exception:{type(error).__name__}:{error}")
        summary["traceback"] = traceback.format_exc()
    finally:
        if runtime is not None:
            try:
                runtime.unlock_zeroed(None)
            except Exception:
                pass
        if env is not None:
            env.close()
        _write_rows(output / "probe_rows.jsonl", rows)
        rows_bytes = (output / "probe_rows.jsonl").read_bytes()
        summary["rows_sha256"] = hashlib.sha256(rows_bytes).hexdigest()
        summary["rows_byte_size"] = len(rows_bytes)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"HYBRID_LATCH_BACKEND_{summary['status'].upper()} "
            f"backend={args.backend} summary={output / 'summary.json'}",
            flush=True,
        )
        simulation_app.close()
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
