#!/usr/bin/env python3
"""Deterministically calibrate TASK-004 simulation-only primitive authority."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))

TASK_ID = "Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0"
DEFAULT_OUTPUT = ROOT / "logs" / "local_primitives_sim_authority"
PRIMARY_TORQUES_NM = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 5.0e-3)
PRIMARY_PIN_FORCES_N = (0.05, 0.10, 0.20, 0.50)
EXPANDED_TORQUES_NM = (0.01, 0.02)
EXPANDED_PIN_FORCES_N = (1.0, 2.0)
CALIBRATED_ENDPOINT_DAMPING_NS_PER_M = 20.0


@dataclass(frozen=True)
class AuthorityCandidate:
    pose_torque_limit_nm: float
    endpoint_pin_force_n: float


@dataclass(frozen=True)
class CalibrationRecord:
    candidate: AuthorityCandidate
    status: str
    completion_time_s: float | None
    reason: str


def candidate_grid(
    torques_nm: Iterable[float], pin_forces_n: Iterable[float],
) -> tuple[AuthorityCandidate, ...]:
    """Return deterministic torque-major, pin-force-minor candidates."""

    return tuple(
        AuthorityCandidate(float(torque), float(pin))
        for torque in torques_nm
        for pin in pin_forces_n
    )


def select_candidate(records: Iterable[CalibrationRecord]) -> AuthorityCandidate | None:
    """Select the passing candidate with lowest torque, pin force, then time."""

    passing = [record for record in records if record.status == "pass"]
    if not passing:
        return None
    winner = min(
        passing,
        key=lambda record: (
            record.candidate.pose_torque_limit_nm,
            record.candidate.endpoint_pin_force_n,
            float(record.completion_time_s or math.inf),
        ),
    )
    return winner.candidate


def _candidate_controller_cfg(base_cfg, candidate: AuthorityCandidate):
    torque = candidate.pose_torque_limit_nm
    pin = candidate.endpoint_pin_force_n
    return replace(
        base_cfg,
        axis_kp_nm_per_rad=torque,
        axis_kd_nms_per_rad=0.08 * torque,
        roll_damping_nms_per_rad=0.08 * torque,
        pose_torque_limit_nm=torque,
        endpoint_pin_force_n=pin,
        anchor_kd_ns_per_m=CALIBRATED_ENDPOINT_DAMPING_NS_PER_M,
        force_slew_limit_n_per_s=50.0,
        torque_slew_limit_nm_per_s=0.2,
        total_force_limit_n=min(5.0, max(base_cfg.total_force_limit_n, 1.25 * pin)),
        total_torque_limit_nm=min(0.02, max(base_cfg.total_torque_limit_nm, torque)),
        profile_sha256="",
    )


def _write_selected_profile(path: Path, base_profile, candidate: AuthorityCandidate) -> None:
    """Write exactly the authority exercised by the winning candidate."""

    values = asdict(base_profile)
    torque = candidate.pose_torque_limit_nm
    pin = candidate.endpoint_pin_force_n
    values.update(
        axis_kp_nm_per_rad=torque,
        axis_kd_nms_per_rad=0.08 * torque,
        roll_damping_nms_per_rad=0.08 * torque,
        pose_torque_limit_nm=torque,
        endpoint_pin_force_n=pin,
        anchor_kd_ns_per_m=CALIBRATED_ENDPOINT_DAMPING_NS_PER_M,
        force_slew_limit_n_per_s=50.0,
        torque_slew_limit_nm_per_s=0.2,
        total_force_limit_n=min(5.0, max(base_profile.total_force_limit_n, 1.25 * pin)),
        total_torque_limit_nm=min(0.02, max(base_profile.total_torque_limit_nm, torque)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_candidate(env, term, monitor, candidate, base_cfg, seed: int) -> tuple[CalibrationRecord, dict]:
    import numpy as np
    import torch
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
        LocalPrimitiveController, PrimitiveStatus,
    )

    env.reset(seed=seed)
    term.controller = LocalPrimitiveController(_candidate_controller_cfg(base_cfg, candidate))
    action = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    previous_position = None
    substep_index = 0
    max_displacement = 0.0
    camera_load_samples = 0
    nonfinite_samples = 0
    max_force = max_torque = 0.0
    final_telemetry = None
    reason = "eight_second_timeout"
    for _ in range(int(math.ceil(8.0 / env.unwrapped.step_dt))):
        env.step(torch.as_tensor(action, device=env.unwrapped.device).reshape(1, 4))
        action[0] = 0.0
        final_telemetry = term.telemetry
        positions = term.substep_positions_world_m
        for position in positions[substep_index:]:
            if previous_position is not None:
                max_displacement = max(
                    max_displacement, float(np.linalg.norm(position - previous_position)),
                )
            previous_position = position
        substep_index = len(positions)
        if max_displacement > 0.005:
            reason = "physics_step_center_displacement_exceeded"
            break
        if final_telemetry is None:
            continue
        component_values = np.concatenate((
            final_telemetry.total_force_world_n,
            final_telemetry.total_torque_world_nm,
            final_telemetry.actual_axis_world,
        ))
        nonfinite_samples += int(not np.isfinite(component_values).all())
        max_force = max(max_force, float(np.linalg.norm(final_telemetry.total_force_world_n)))
        max_torque = max(max_torque, float(np.linalg.norm(final_telemetry.total_torque_world_nm)))
        pose = term.capsule.data.root_com_pose_w.torch[0].detach().cpu().numpy()
        axis = final_telemetry.actual_axis_world
        contacts = monitor.consume()
        if contacts:
            largest = max(impulse for _, impulse in contacts)
            for point, impulse in contacts:
                if largest > 0.0 and impulse >= 0.1 * largest:
                    sigma = float(np.dot(np.asarray(point) - pose[:3], axis))
                    camera_load_samples += int(sigma > 0.0065)
        if final_telemetry.status == PrimitiveStatus.SUCCEEDED_HOLDING:
            reason = "succeeded_holding"
            break
        if final_telemetry.status in (
            PrimitiveStatus.TIMED_OUT, PrimitiveStatus.NONFINITE,
            PrimitiveStatus.INVALID_START,
        ):
            reason = final_telemetry.status.value
            break
    completion = None if final_telemetry is None else final_telemetry.completion_time_s
    final_velocity = term.capsule.data.root_com_vel_w.torch[0].detach().cpu().numpy()
    upright_error = math.inf
    status_value = "missing_telemetry"
    if final_telemetry is not None:
        upright_error = math.acos(float(np.clip(final_telemetry.actual_axis_world[2], -1.0, 1.0)))
        status_value = final_telemetry.status.value
    passed = (
        status_value == PrimitiveStatus.SUCCEEDED_HOLDING.value
        and completion is not None
        and completion <= 8.0
        and upright_error <= math.radians(3.0)
        and nonfinite_samples == 0
        and max_displacement <= 0.005
        and camera_load_samples == 0
    )
    record = CalibrationRecord(candidate, "pass" if passed else "fail", completion, reason)
    evidence = {
        "candidate": asdict(candidate),
        "status": record.status,
        "controller_status": status_value,
        "reason": reason,
        "completion_time_s": completion,
        "upright_error_rad": upright_error,
        "stable_time_s": None if final_telemetry is None else final_telemetry.stable_time_s,
        "camera_hemisphere_load_samples": camera_load_samples,
        "max_physics_step_center_displacement_m": max_displacement,
        "nonfinite_samples": nonfinite_samples,
        "max_force_n": max_force,
        "max_torque_nm": max_torque,
        "final_linear_speed_m_s": float(np.linalg.norm(final_velocity[:3])),
        "final_angular_speed_rad_s": float(np.linalg.norm(final_velocity[3:6])),
        "final_roll_speed_rad_s": None if final_telemetry is None else abs(
            float(np.dot(final_velocity[3:6], final_telemetry.actual_axis_world))
        ),
        "final_perpendicular_angular_speed_rad_s": None if final_telemetry is None else float(
            np.linalg.norm(
                final_velocity[3:6]
                - np.dot(final_velocity[3:6], final_telemetry.actual_axis_world)
                * final_telemetry.actual_axis_world
            )
        ),
        "final_axis_tracking_error_rad": None if final_telemetry is None else math.acos(
            float(np.clip(
                np.dot(final_telemetry.actual_axis_world, final_telemetry.desired_axis_world),
                -1.0, 1.0,
            ))
        ),
    }
    return record, evidence


def _run(args) -> tuple[AuthorityCandidate | None, Path]:
    import gymnasium as gym
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
        load_simulation_profile, make_local_primitive_controller_cfg, simulation_profile_sha256,
    )
    from scripts.local_primitives.validate_local_primitives_flat import ReadOnlyContactPoints

    output = args.output_directory.expanduser().resolve() / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    attempts_path = output / "attempts.jsonl"
    records: list[CalibrationRecord] = []
    base_cfg = make_local_primitive_controller_cfg()
    grids = (
        (AuthorityCandidate(*args.only_candidate),),
    ) if args.only_candidate is not None else (
        candidate_grid(PRIMARY_TORQUES_NM, PRIMARY_PIN_FORCES_N),
        candidate_grid(EXPANDED_TORQUES_NM, EXPANDED_PIN_FORCES_N),
    )
    with attempts_path.open("w", encoding="utf-8") as stream:
        for grid_index, grid in enumerate(grids, start=1):
            for candidate in grid:
                # A new scene per candidate prevents PhysX contact-cache and
                # solver-history leakage from manufacturing reset-order passes.
                cfg = parse_env_cfg(args.task, device="cpu", num_envs=1)
                cfg.seed = args.seed
                env = gym.make(args.task, cfg=cfg)
                monitor = None
                try:
                    term = env.unwrapped.action_manager.get_term("local_primitive")
                    monitor = ReadOnlyContactPoints(term.capsule.root_view.prim_paths[0])
                    record, evidence = _run_candidate(
                        env, term, monitor, candidate, base_cfg, args.seed,
                    )
                finally:
                    if monitor is not None:
                        monitor.close()
                    env.close()
                records.append(record)
                evidence["grid"] = grid_index
                evidence["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
                stream.write(json.dumps(evidence, sort_keys=True) + "\n")
                stream.flush()
                print(
                    "SIMULATION_AUTHORITY_ATTEMPT "
                    f"grid={grid_index} torque={candidate.pose_torque_limit_nm:g} "
                    f"pin={candidate.endpoint_pin_force_n:g} status={record.status} "
                    f"reason={record.reason} completion={record.completion_time_s}"
                )
            selected = select_candidate(records)
            if selected is not None:
                break
    selected = select_candidate(records)
    if selected is not None:
        profile_path = args.write_selected_profile.expanduser().resolve()
        _write_selected_profile(profile_path, load_simulation_profile(), selected)
        digest = simulation_profile_sha256(profile_path)
        selection = {
            "selected": asdict(selected), "profile_path": str(profile_path),
            "profile_sha256": digest, "attempt_count": len(records),
        }
        (output / "selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(f"SIMULATION_AUTHORITY_SELECTED={json.dumps(selection, sort_keys=True)}")
    return selected, attempts_path


def main() -> int:
    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write_selected_profile", type=Path,
        default=ROOT / "configs/local_primitives/simulation_profile.json",
    )
    parser.add_argument("--output_directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only_candidate", type=float, nargs=2, metavar=("TORQUE_NM", "PIN_FORCE_N"),
        help="diagnostic single-candidate rerun; the contractual full grids remain the default",
    )
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.task != TASK_ID:
        parser.error(f"calibrator accepts only {TASK_ID}")
    args.device = "cpu"
    args.enable_cameras = True
    launcher = AppLauncher(args)
    try:
        selected, attempts_path = _run(args)
        print(f"SIMULATION_AUTHORITY_ATTEMPTS={attempts_path}")
        if selected is None:
            print("SIMULATION_AUTHORITY_NEEDS_DECISION")
            return 2
        return 0
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
