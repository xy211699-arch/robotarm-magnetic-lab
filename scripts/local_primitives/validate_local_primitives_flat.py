#!/usr/bin/env python3
"""Quantitatively validate the four TASK-004 primitives on flat contact."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))

TASK_ID = "Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0"
SCHEMA_VERSION = "local_primitives_flat_validation_v1"
DEFAULT_OUTPUT = Path("/mnt/isaac-linux/robotarm_magnetic_lab/logs/local_primitives_flat")
PRIMITIVE_NAMES = (
    "side_to_upright", "upright_to_side", "upright_to_30_deg", "cone_30_deg",
)
SEQUENCES = ((0,), (0, 1), (0, 2), (0, 2, 3))


def evaluate_flat_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen quantitative acceptance gate to recorded evidence."""

    if summary.get("preflight", {}).get("gate", {}).get("status") != "pass":
        return {"status": "needs_decision", "failures": ["preflight did not pass"]}
    if summary.get("runtime_contract", {}).get("forbidden_calls"):
        return {"status": "needs_decision", "failures": ["forbidden runtime state writer"]}
    failures: list[str] = []
    records = summary.get("primitives", {})
    for name in PRIMITIVE_NAMES:
        record = records.get(name)
        if record is None:
            failures.append(f"{name}: missing result")
            continue
        if record.get("status") != "succeeded_holding":
            failures.append(f"{name}: did not enter stable holding")
        completion_time = record.get("completion_time_s")
        if completion_time is None or float(completion_time) >= 10.0:
            failures.append(f"{name}: completion time is not below 10 seconds")
        if int(record.get("nonfinite_samples", 0)):
            failures.append(f"{name}: nonfinite state or wrench")
        if float(record.get("max_force_n", math.inf)) > float(record.get("force_bound_n", -math.inf)) + 1.0e-9:
            failures.append(f"{name}: force bound exceeded")
        if float(record.get("max_torque_nm", math.inf)) > float(record.get("torque_bound_nm", -math.inf)) + 1.0e-12:
            failures.append(f"{name}: torque bound exceeded")
    rise = records.get("side_to_upright", {})
    if int(rise.get("camera_hemisphere_load_samples", 0)):
        failures.append("side_to_upright: camera hemisphere carried load")
    if not bool(rise.get("late_dominant_non_camera", False)):
        failures.append("side_to_upright: late dominant support was not non-camera")
    cone = records.get("cone_30_deg", {})
    if float(cone.get("actual_cone_coverage_rad", -math.inf)) < 2.0 * math.pi - math.radians(10.0):
        failures.append("cone_30_deg: actual azimuth coverage is incomplete")
    if float(cone.get("cone_tilt_rmse_rad", math.inf)) > math.radians(5.0):
        failures.append("cone_30_deg: tilt RMSE exceeds five degrees")
    return {"status": "pass" if not failures else "fail", "failures": failures}


def record_calibration_attempt(
    log_path: Path,
    attempt_index: int,
    summary_path: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
    command: list[str],
) -> None:
    """Append one immutable external calibration-evidence row."""

    record = {
        "attempt_index": int(attempt_index),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": os.environ.get("TASK004_IMPLEMENTATION_COMMIT", "working-tree"),
        "controller_config": config,
        "validation_command": command,
        "summary_path": str(summary_path.resolve()),
        "gate": summary.get("gate", {}),
        "primitives": summary.get("primitives", {}),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


class ReadOnlyContactPoints:
    """Collect PhysX contact points without changing controller inputs or state."""

    def __init__(self, capsule_root: str) -> None:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self.capsule_root = capsule_root
        self._convert_path = PhysicsSchemaTools.intToSdfPath
        self.points: list[tuple[list[float], float]] = []
        interface = get_physx_simulation_interface()
        self.subscription = interface.subscribe_contact_report_events(self._callback)

    def _callback(self, headers, data) -> None:
        points: list[tuple[list[float], float]] = []
        for header in headers:
            collider0 = str(self._convert_path(header.collider0))
            collider1 = str(self._convert_path(header.collider1))
            if not (collider0.startswith(self.capsule_root) or collider1.startswith(self.capsule_root)):
                continue
            for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                datum = data[index]
                try:
                    impulse = abs(float(datum.impulse))
                except (TypeError, ValueError):
                    impulse = 0.0
                points.append(([float(v) for v in datum.position], impulse))
        self.points = points

    def close(self) -> None:
        self.subscription = None


def _flat(value):
    import numpy as np

    tensor = value.torch if hasattr(value, "torch") else value
    return tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _run(args) -> tuple[dict[str, Any], Path]:
    import gymnasium as gym
    import numpy as np
    import torch
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
        PrimitiveStatus, directed_axis_from_quaternion_wxyz, make_local_primitive_controller_cfg,
    )
    from scripts.local_primitives.inspect_local_primitives_prerequisites import build_gate, scan_runtime_contract, source_report

    output = args.output_directory.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args.task, device="cpu", num_envs=1)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    stream_path = output / "samples.jsonl"
    records: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    monitor = None
    try:
        if env is not None:
            env.reset(seed=args.seed)
            term = env.unwrapped.action_manager.get_term("local_primitive")
            monitor = ReadOnlyContactPoints(term.capsule.root_view.prim_paths[0])
            preflight = source_report()
            preflight["tasks"]["registered"] = sorted(
                task_id for task_id in gym.registry if "Local-Primitives" in task_id
            )
            preflight["contact_points"]["read_only_access"] = True
            preflight["isolation"]["flat_action_terms"] = list(env.unwrapped.action_manager.active_terms)
            preflight["gate"] = build_gate(preflight)
            direction = np.array([
                math.cos(math.radians(args.direction_azimuth_deg)),
                math.sin(math.radians(args.direction_azimuth_deg)),
            ])
            with stream_path.open("w", encoding="utf-8") as stream:
                for sequence_index, sequence in enumerate(SEQUENCES):
                    env.reset(seed=args.seed + sequence_index)
                    for code in sequence:
                        name = PRIMITIVE_NAMES[code]
                        camera_load = 0
                        late_non_camera = False
                        max_force = max_torque = 0.0
                        nonfinite = 0
                        action = np.array([1.0, float(code), direction[0], direction[1]], dtype=np.float32)
                        final_telemetry = None
                        for step in range(int(math.ceil(10.0 / env.unwrapped.step_dt))):
                            tensor = torch.as_tensor(action, device=env.unwrapped.device).reshape(1, 4)
                            env.step(tensor)
                            action[0] = 0.0
                            telemetry = term.telemetry
                            if telemetry is None:
                                continue
                            final_telemetry = telemetry
                            pose = _flat(term.capsule.data.root_com_pose_w)
                            link_pose = _flat(term.capsule.data.root_link_pose_w)
                            velocity = _flat(term.capsule.data.root_com_vel_w)
                            quat_wxyz = link_pose[3:7][[3, 0, 1, 2]]
                            axis = directed_axis_from_quaternion_wxyz(quat_wxyz)
                            force = _flat(term.applied_force_world)
                            torque = _flat(term.applied_torque_world)
                            values = np.concatenate((pose, velocity, force, torque))
                            nonfinite += int(not np.isfinite(values).all())
                            max_force = max(max_force, float(np.linalg.norm(force)))
                            max_torque = max(max_torque, float(np.linalg.norm(torque)))
                            contacts = monitor.points
                            if contacts:
                                largest = max(value[1] for value in contacts)
                                for point, impulse in contacts:
                                    if largest <= 0.0 or impulse < 0.1 * largest:
                                        continue
                                    sigma = float(np.dot(np.asarray(point) - pose[:3], axis))
                                    desired_tilt = math.acos(float(np.clip(telemetry.desired_axis_world[2], -1.0, 1.0)))
                                    if sigma > 0.0065 and code == 0:
                                        camera_load += 1
                                    if desired_tilt < math.radians(45.0) and sigma < -0.0055:
                                        late_non_camera = True
                            row = {
                                "sequence": sequence_index, "primitive": name, "step": step,
                                "sim_time_s": float(telemetry.elapsed_s),
                                "status": telemetry.status.value,
                                "desired_axis_world": telemetry.desired_axis_world.tolist(),
                                "actual_axis_world": telemetry.actual_axis_world.tolist(),
                                "position_world_m": pose[:3].tolist(), "velocity_world": velocity.tolist(),
                                "force_world_n": force.tolist(), "torque_world_nm": torque.tolist(),
                                "cone_phase_rad": telemetry.cone_phase_rad,
                            }
                            stream.write(json.dumps(row, sort_keys=True) + "\n")
                            if telemetry.status in (
                                PrimitiveStatus.SUCCEEDED_HOLDING,
                                PrimitiveStatus.TIMED_OUT,
                                PrimitiveStatus.NONFINITE,
                                PrimitiveStatus.INVALID_START,
                            ):
                                break
                        if final_telemetry is None:
                            raise RuntimeError(f"no telemetry for primitive {name}")
                        record = {
                            "status": final_telemetry.status.value,
                            "completion_time_s": final_telemetry.completion_time_s,
                            "max_force_n": max_force,
                            "force_bound_n": math.hypot(term.controller.cfg.xy_force_limit_n, term.controller.cfg.downward_preload_n),
                            "max_torque_nm": max_torque,
                            "torque_bound_nm": term.controller.cfg.torque_limit_nm,
                            "nonfinite_samples": nonfinite,
                            "camera_hemisphere_load_samples": camera_load,
                            "late_dominant_non_camera": late_non_camera,
                            "actual_cone_coverage_rad": final_telemetry.cone_phase_rad,
                            "cone_tilt_rmse_rad": final_telemetry.cone_tilt_rmse_rad,
                        }
                        aggregate[name].append(record)
            for name, attempts in aggregate.items():
                records[name] = max(attempts, key=lambda item: float(item["completion_time_s"] or math.inf))
                if name == "side_to_upright":
                    records[name]["camera_hemisphere_load_samples"] = sum(item["camera_hemisphere_load_samples"] for item in attempts)
                    records[name]["late_dominant_non_camera"] = all(item["late_dominant_non_camera"] for item in attempts)
            summary = {
                "schema_version": SCHEMA_VERSION,
                "task": args.task,
                "preflight": preflight,
                "runtime_contract": scan_runtime_contract(),
                "sequences": [list(sequence) for sequence in SEQUENCES],
                "primitives": records,
                "controller_config": asdict(make_local_primitive_controller_cfg()),
            }
            summary["gate"] = evaluate_flat_summary(summary)
            summary_path = output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if args.calibration_log is not None and args.attempt_index is not None:
                record_calibration_attempt(
                    args.calibration_log,
                    args.attempt_index,
                    summary_path,
                    summary,
                    summary["controller_config"],
                    [sys.executable, *sys.argv],
                )
        return summary, summary_path
    finally:
        if monitor is not None:
            monitor.close()
        env.close()


def main() -> int:
    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direction_azimuth_deg", type=float, default=0.0)
    parser.add_argument("--output_directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration_log", type=Path)
    parser.add_argument("--attempt_index", type=int)
    parser.add_argument("--record_existing_summary", type=Path)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.task != TASK_ID:
        parser.error(f"TASK-004 flat validator accepts only {TASK_ID}")
    if (args.calibration_log is None) != (args.attempt_index is None):
        parser.error("--calibration_log and --attempt_index must be supplied together")
    if args.record_existing_summary is not None:
        if args.calibration_log is None:
            parser.error("--record_existing_summary requires calibration logging arguments")
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
            make_local_primitive_controller_cfg,
        )
        summary_path = args.record_existing_summary.expanduser().resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = asdict(make_local_primitive_controller_cfg())
        summary.setdefault("controller_config", config)
        record_calibration_attempt(
            args.calibration_log, args.attempt_index, summary_path, summary,
            config, [sys.executable, *sys.argv],
        )
        print(f"LOCAL_PRIMITIVES_CALIBRATION_RECORDED={args.calibration_log}")
        return 0
    args.device = "cpu"
    args.enable_cameras = True
    launcher = AppLauncher(args)
    try:
        summary, path = _run(args)
        print(json.dumps(summary["gate"], sort_keys=True))
        print(f"LOCAL_PRIMITIVES_FLAT_SUMMARY={path}")
        if summary["gate"]["status"] == "pass":
            print("LOCAL_PRIMITIVES_FLAT_VALIDATION_PASS")
            return 0
        print("LOCAL_PRIMITIVES_FLAT_VALIDATION_FAIL")
        return 1
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
