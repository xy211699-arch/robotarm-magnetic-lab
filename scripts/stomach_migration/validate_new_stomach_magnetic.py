#!/usr/bin/env python3
"""Exercise the reused 4x9 magnetic controller in the confirmed new stomach.

This is a scene/test adapter. It does not compute magnetic trajectories itself:
RollEscapeProbe generates roll requests and the existing ActuatorVectorAction
plans, certifies and executes the robot/Ball motion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/robotarm_magnetic_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("hold", "axes", "tip_roll", "area_scan"), default="hold")
parser.add_argument("--seconds", type=int, default=10)
parser.add_argument("--amplitude", type=float, default=0.5)
parser.add_argument("--force-mn", type=float, choices=(10.0, 15.0, 20.0, 30.0), default=10.0)
parser.add_argument("--source-height-mm", type=float, choices=(100.0, 110.0, 120.0), default=120.0)
parser.add_argument("--direction-sign", type=int, choices=(-1, 1), default=-1)
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--output", type=Path, default=ROOT / "artifacts/new_stomach_magnetic")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.seconds <= 300 or not 0 < args.amplitude <= 1:
    parser.error("seconds must be 1..300 and amplitude must be in (0,1]")
args.enable_cameras = bool(args.record_video)
launcher = AppLauncher(args)

import numpy as np
import omni.usd
import torch
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv

from robotarm_magnetic_lab.baselines.magnetic_roll_escape import RollEscapeProbe
from robotarm_magnetic_lab.coverage.visibility import WarpFirstHitRaycaster
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.batched_arm_clearance import (
    minimum_clearance_fast,
    minimum_world_clearance,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_vector_env_cfg import (
    RobotarmMagneticNewStomachVectorEnvCfg,
)
from robotarm_magnetic_lab.ui.magnetic_observer import DualObserver, configure_observers


def _array(tensor):
    return tensor.detach().cpu().numpy()


def _json_line(stream, record):
    stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    stream.flush()


def _clearance(term):
    q = _array(term.robot.data.joint_pos.torch[0, term.ids[:6]])[None, :]
    own = float(minimum_clearance_fast(term.kinematics, q)[0])
    wall = float(minimum_world_clearance(term.world_checker, q)[0])
    return {"self_clearance_m": own, "wall_clearance_m": wall}


def _separation(term, bridge):
    capsule = term.capsule
    p = _array(capsule.data.root_link_pos_w.torch[0])
    q = _array(capsule.data.root_link_quat_w.torch[0])
    offset = float(bridge.config["magnets"]["target_cylinder"]["center_offset_axis_m"])
    magnet = p + Rotation.from_quat(q).apply([0.0, 0.0, offset])
    source = term.pose(term.source)[0]
    return float(np.linalg.norm(source - magnet))


def _requested_chunk(mode, cycle, amplitude, probe):
    if probe is not None:
        return probe.chunk()
    chunk = np.zeros((4, 9), dtype=np.float64)
    if mode == "axes":
        for horizon in range(4):
            k = cycle + horizon
            if k >= 2:
                chunk[horizon, ((k - 2) // 2) % 9] = amplitude * (1 if k % 2 == 0 else -1)
    return chunk


def main():
    run_dir = args.output.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "running",
        "mode": args.mode,
        "seconds_requested": args.seconds,
        "seconds_completed": 0,
        "physics_hz": 240,
        "command_hz": 1,
        "action_shape": [1, 36],
        "controller": "reused_actuator_vector_4x9",
        "direct_capsule_force": False,
        "record_video": bool(args.record_video),
        "force_target_mn": args.force_mn if args.mode in ("tip_roll", "area_scan") else None,
        "source_height_mm": args.source_height_mm if args.mode in ("tip_roll", "area_scan") else None,
        "output": str(run_dir),
        "projected_cycles": 0,
        "rejected_cycles": 0,
        "tracking_failures": 0,
    }
    cfg = RobotarmMagneticNewStomachVectorEnvCfg()
    cfg.sim.device = args.device
    cfg.episode_length_s = args.seconds + 10.0
    cfg.observations.vision = None
    cfg.scene.capsule_camera = None
    cfg.scene.capsule_camera_preview = None
    cfg.sim.render_interval = 24 if args.record_video else 240
    if args.record_video:
        configure_observers(cfg)
    env = None
    observer = None
    old_update = None
    start = time.monotonic()
    samples = None
    boundaries = None
    physics = None
    try:
        with launch_simulation(cfg, args):
            env = ManagerBasedRLEnv(cfg=cfg)
            term = env.action_manager.get_term("magnet")
            bridge = env.event_manager.get_term_cfg("magnetic_collision_bridge").func
            env.reset(seed=914)
            with threadpool_limits(limits=1, user_api="blas"):
                env.step(torch.zeros((1, 36), device=env.device))
            geometry = term._new_stomach_runtime
            summary["geometry_sha256"] = geometry.geometry_sha256
            summary["geometry_consumers"] = geometry.consumer_hashes()
            summary["collision_mesh_path"] = geometry.collision_mesh_path
            summary["asset_usd_sha256"] = geometry.asset_usd_sha256
            summary["initial_clearance"] = _clearance(term)
            summary["initial_source_capsule_distance_m"] = _separation(term, bridge)
            summary["magnetic_simulation_config"] = bridge.config["simulation"].copy()
            (run_dir / "geometry_audit.json").write_text(
                json.dumps(geometry.audit.to_json(), indent=2, sort_keys=True) + "\n"
            )
            print("NEW_STOMACH_MAGNETIC_READY", json.dumps({
                "geometry_sha256": geometry.geometry_sha256,
                "collision_mesh_path": geometry.collision_mesh_path,
                "initial_clearance": summary["initial_clearance"],
                "action_shape": [1, 36],
            }), flush=True)
            probe = None
            if args.mode in ("tip_roll", "area_scan"):
                probe = RollEscapeProbe(
                    term, bridge, geometry.reference,
                    direction_sign=args.direction_sign,
                    tangential_n=args.force_mn / 1000.0,
                    minimum_height=args.source_height_mm / 1000.0,
                    motion_pattern="area_scan" if args.mode == "area_scan" else "straight",
                )
            samples = (run_dir / "state_10hz.jsonl").open("w")
            boundaries = (run_dir / "boundaries_1hz.jsonl").open("w")
            physics = (run_dir / "joint_state_240hz.jsonl").open("w")
            physics_steps = 0
            distance_bad_frames = 0
            if args.record_video:
                raycaster = WarpFirstHitRaycaster(geometry.reference, device=str(env.device))
                observer = DualObserver(env, run_dir / "video", raycaster)
                import warp as wp

                def capture(now):
                    nonlocal distance_bad_frames
                    observer.pose()
                    env.sim.render()
                    for camera in (observer.external, observer.internal):
                        wp.to_torch(camera._is_outdated).fill_(True)
                        camera._update_outdated_buffers(force_recompute=True)
                    observer.capture(now, coverage=None)
                    distance = _separation(term, bridge)
                    distance_bad_frames = (
                        distance_bad_frames + 1
                        if distance > 1.2 * summary["initial_source_capsule_distance_m"]
                        else 0
                    )
                    _json_line(samples, {
                        "time_s": now,
                        "video_frame_index": observer.frames - 1,
                        "external_sensor_frame": int(observer.external.frame.torch[0]),
                        "internal_sensor_frame": int(observer.internal.frame.torch[0]),
                        "external_rgb_sha256": hashlib.sha256(observer.last_images["external"].tobytes()).hexdigest(),
                        "internal_rgb_sha256": hashlib.sha256(observer.last_images["internal"].tobytes()).hexdigest(),
                        "capsule_velocity_world": _array(term.capsule.data.root_com_vel_w.torch[0]).tolist(),
                        "contact_force_world_N": _array(env.scene["capsule_contact"].data.net_forces_w.torch[0, 0]).tolist(),
                        "source_position_world_m": term.pose(term.source)[0].tolist(),
                        "capsule_pose_xyzw": _array(term.capsule.data.root_link_pose_w.torch[0]).tolist(),
                        "source_capsule_distance_m": distance,
                        "applied_wrench": _array(bridge.state["wrench"][0]).tolist(),
                    })
                    if distance_bad_frames >= 20:
                        raise RuntimeError("source-capsule separation exceeds 1.2x initial for 2s")

                capture(0.0)
                old_update = env.scene.update

                def update_scene(scene, dt, *positional, **keyword):
                    nonlocal physics_steps
                    old_update(dt, *positional, **keyword)
                    if abs(float(dt) - 1.0 / 240.0) > 1.0e-9:
                        raise RuntimeError("unexpected physics step duration")
                    physics_steps += 1
                    if physics_steps % 24 == 0:
                        capture(physics_steps / 240.0)

                from types import MethodType

                env.scene.update = MethodType(update_scene, env.scene)
            else:
                def capture_state(now):
                    nonlocal distance_bad_frames
                    distance = _separation(term, bridge)
                    distance_bad_frames = (
                        distance_bad_frames + 1
                        if distance > 1.2 * summary["initial_source_capsule_distance_m"]
                        else 0
                    )
                    _json_line(samples, {
                        "time_s": now,
                        "capsule_pose_xyzw": _array(term.capsule.data.root_link_pose_w.torch[0]).tolist(),
                        "capsule_velocity_world": _array(term.capsule.data.root_com_vel_w.torch[0]).tolist(),
                        "contact_force_world_N": _array(
                            env.scene["capsule_contact"].data.net_forces_w.torch[0, 0]
                        ).tolist(),
                        "source_capsule_distance_m": distance,
                        "source_position_world_m": term.pose(term.source)[0].tolist(),
                        "applied_wrench": _array(bridge.state["wrench"][0]).tolist(),
                    })
                    if distance_bad_frames >= 20:
                        raise RuntimeError("source-capsule separation exceeds 1.2x initial for 2s")

                capture_state(0.0)
                old_update = env.scene.update

                def update_scene(scene, dt, *positional, **keyword):
                    nonlocal physics_steps
                    old_update(dt, *positional, **keyword)
                    if abs(float(dt) - 1.0 / 240.0) > 1.0e-9:
                        raise RuntimeError("unexpected physics step duration")
                    physics_steps += 1
                    if physics_steps % 24 == 0:
                        capture_state(physics_steps / 240.0)

                from types import MethodType

                env.scene.update = MethodType(update_scene, env.scene)
            for cycle in range(args.seconds):
                chunk = _requested_chunk(args.mode, cycle, args.amplitude, probe)
                with threadpool_limits(limits=1, user_api="blas"):
                    _, _, terminated, truncated, _ = env.step(
                        torch.as_tensor(chunk.reshape(1, 36), device=env.device, dtype=torch.float32)
                    )
                record = term.boundary_result()
                record.update({
                    "cycle": cycle,
                    "time_s": cycle + 1.0,
                    "clearance_actual": _clearance(term),
                    "source_capsule_distance_m": _separation(term, bridge),
                    "applied_wrench": _array(bridge.state["wrench"][0]).tolist(),
                    "capsule_pose_xyzw": _array(term.capsule.data.root_link_pose_w.torch[0]).tolist(),
                    "capsule_contact_force_world_N": _array(env.scene["capsule_contact"].data.net_forces_w.torch[0, 0]).tolist(),
                    "termination": bool(terminated.any()),
                    "truncation": bool(truncated.any()),
                })
                if probe is not None:
                    record["roll_controller"] = probe.diagnostics
                _json_line(boundaries, record)
                for state in term.physics_records:
                    _json_line(physics, {"cycle": cycle, **state})
                summary["seconds_completed"] = cycle + 1
                summary["projected_cycles"] += int(0.0 < record["projection_scale"] < 1.0)
                summary["rejected_cycles"] += int(record["projection_scale"] == 0.0)
                summary["tracking_failures"] += int(record["tracking_acceptance"] != "passed")
                (run_dir / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
                if bool(terminated.any()) or bool(truncated.any()):
                    raise RuntimeError("environment terminated before requested duration")
            summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "error"
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if env is not None and old_update is not None:
            env.scene.update = old_update
        if observer is not None:
            observer.snapshot("final")
            observer.close()
            summary["video_frames"] = observer.frames
        for stream in (samples, boundaries, physics):
            if stream is not None:
                stream.close()
        summary["wall_s"] = time.monotonic() - start
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (run_dir / "status.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print("NEW_STOMACH_MAGNETIC_SUMMARY", json.dumps(summary, allow_nan=False), flush=True)
        if env is not None:
            env.close()
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
