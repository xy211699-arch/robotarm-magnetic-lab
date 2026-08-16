#!/usr/bin/env python3
"""Deterministically validate TASK-003 gravity, contact, force, and continuity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))

TASK_ID = "Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0"
DEFAULT_OUTPUT = Path(
    "/mnt/isaac-linux/robotarm_magnetic_lab/logs/dynamic_force_validation"
)
DIRECTIONS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def evaluate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate recorded evidence without assuming free-space displacement."""
    preflight = summary.get("preflight", {})
    runtime = preflight.get("runtime_contract", {})
    if runtime.get("forbidden_calls"):
        return {"status": "needs_decision", "failures": ["forbidden runtime state writer"]}
    if preflight.get("gate", {}).get("status") != "pass":
        return {"status": "needs_decision", "failures": ["preflight did not pass"]}

    failures: list[str] = []
    directions = summary.get("directions", {})
    missing = sorted(set(DIRECTIONS) - set(directions))
    extra = sorted(set(directions) - set(DIRECTIONS))
    if missing or extra:
        failures.append(f"six signed directions differ: missing={missing}, extra={extra}")
    continuity = summary.get("continuity", {})
    if int(continuity.get("nonfinite_samples", 0)):
        failures.append("nonfinite state observed")
    if float(continuity.get("max_physics_step_displacement_m", math.inf)) > float(
        continuity.get("allowed_physics_step_displacement_m", -math.inf)
    ):
        failures.append("physics-step displacement exceeded authored velocity bound")
    if int(continuity.get("sustained_clearance_decrease_events", 0)):
        failures.append("sustained surface-clearance decrease observed")
    for name, record in directions.items():
        if float(record.get("max_force_error_n", math.inf)) > 1.0e-6:
            failures.append(f"{name}: applied wrench mismatch")
        if float(record.get("max_commanded_torque_nm", math.inf)) > 1.0e-12:
            failures.append(f"{name}: nonzero commanded torque")
        if int(record.get("physics_substep_samples", 0)) <= 0:
            failures.append(f"{name}: no physics progression evidence")
    return {"status": "pass" if not failures else "fail", "failures": failures}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _quat_wxyz_matrix(quaternion) -> Any:
    import numpy as np

    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    w, x, y, z = np.asarray([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class ReadOnlyClearanceProbe:
    """Measure capsule-to-luminal-mesh clearance without changing simulation state."""

    def __init__(self) -> None:
        import numpy as np
        from scipy.spatial import cKDTree
        from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage

        reference = reference_from_stage()
        self.vertices = np.asarray(reference.vertices_world, dtype=np.float64)
        self.triangles = np.asarray(reference.triangles, dtype=np.int64)
        self.triangle_vertices = self.vertices[self.triangles]
        self.centroids = self.triangle_vertices.mean(axis=1)
        self.tree = cKDTree(self.centroids)
        self.boundary_edges = self._boundary_edges(self.triangles)
        self.candidate_count = min(512, len(self.triangles))

    @staticmethod
    def _boundary_edges(triangles):
        from collections import Counter

        counts = Counter()
        for triangle in triangles:
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                counts[tuple(sorted((int(first), int(second))))] += 1
        return frozenset(edge for edge, count in counts.items() if count == 1)

    def measure(self, center, quaternion, radius_m: float, cylinder_height_m: float):
        import numpy as np
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface.surface_mesh import (
            _closest_points_on_triangles,
        )

        center = np.asarray(center, dtype=np.float64).reshape(3)
        axis = _quat_wxyz_matrix(quaternion) @ np.asarray([0.0, 0.0, 1.0])
        samples = center + np.linspace(-0.5, 0.5, 5)[:, None] * cylinder_height_m * axis
        best_clearance = math.inf
        boundary = False
        for sample in samples:
            _, candidate_ids = self.tree.query(sample, k=self.candidate_count)
            candidate_ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
            points, barycentric = _closest_points_on_triangles(
                sample, self.triangle_vertices[candidate_ids]
            )
            distances = np.linalg.norm(points - sample, axis=1)
            selected_local = int(np.argmin(distances))
            selected = int(candidate_ids[selected_local])
            clearance = float(distances[selected_local] - radius_m)
            best_clearance = min(best_clearance, clearance)
            bary = barycentric[selected_local]
            triangle = self.triangles[selected]
            for zero_index in np.flatnonzero(bary <= 1.0e-9):
                edge = tuple(
                    sorted(int(triangle[i]) for i in range(3) if i != int(zero_index))
                )
                boundary = boundary or edge in self.boundary_edges
        return best_clearance, boundary


def _flat(value):
    import numpy as np

    tensor = value.torch if hasattr(value, "torch") else value
    return tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _state(env, term, probe, phase: str, step: int) -> dict[str, Any]:
    import numpy as np

    base = env.unwrapped
    capsule = base.scene["capsule"]
    contact = base.scene["capsule_contact"]
    pose = _flat(capsule.data.root_com_pose_w)
    velocity = _flat(capsule.data.root_com_vel_w)
    clearance, boundary = probe.measure(pose[:3], pose[3:7], 0.0065, 0.012)
    values = np.concatenate(
        (pose, velocity, _flat(contact.data.net_forces_w)[:3], _flat(term.applied_force_world))
    )
    return {
        "phase": phase,
        "step": int(step),
        "sim_time_s": float(base.episode_length_buf[0].item()) * float(base.step_dt),
        "position_world_m": pose[:3].tolist(),
        "quaternion_wxyz": pose[3:7].tolist(),
        "linear_velocity_world_m_s": velocity[:3].tolist(),
        "angular_velocity_world_rad_s": velocity[3:6].tolist(),
        "force_world_n": _flat(term.applied_force_world).tolist(),
        "torque_world_nm": _flat(term.applied_torque_world).tolist(),
        "contact_force_world_n": _flat(contact.data.net_forces_w)[:3].tolist(),
        "surface_clearance_m": float(clearance),
        "boundary_escape": bool(boundary),
        "finite": bool(np.isfinite(values).all() and np.isfinite(clearance)),
    }


def _phase(env, term, probe, stream, *, name: str, direction, steps: int, start_step: int):
    import numpy as np
    import torch

    rows = []
    action = torch.as_tensor(direction, device=env.unwrapped.device, dtype=torch.float32).reshape(1, 3)
    for offset in range(steps):
        env.step(action)
        row = _state(env, term, probe, name, start_step + offset + 1)
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        rows.append(row)
    return rows


def _max_substep_displacement(term) -> tuple[float, int]:
    import numpy as np

    values = np.asarray(term.substep_positions_world, dtype=np.float64).reshape(-1, 3)
    if len(values) < 2:
        return 0.0, len(values)
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).max()), len(values)


def _clearance_decrease_events(rows: list[dict[str, Any]]) -> int:
    """Count 6-frame monotone drops exceeding 0.5 mm; never correct them."""
    values = [float(row["surface_clearance_m"]) for row in rows]
    events = 0
    for start in range(max(0, len(values) - 6)):
        window = values[start : start + 7]
        if all(window[i + 1] < window[i] - 1.0e-6 for i in range(6)) and window[0] - window[-1] > 5.0e-4:
            events += 1
    return events


def _run_validation(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_tasks  # noqa: F401
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = args.output_directory.expanduser().resolve() / stamp
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args.task, device="cpu", num_envs=1)
    cfg.seed = args.seed
    cfg.actions.dynamic_force.force_weight_ratio = args.force_weight_ratio
    cfg.sim.device = "cpu"
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            env.reset(seed=args.seed)
            term = env.unwrapped.action_manager.get_term("dynamic_force")
            probe = ReadOnlyClearanceProbe()
            # The environment is already live, so build the equivalent report
            # directly below rather than nesting a second simulation context.
            import omni.usd
            from inspect_dynamic_force_prerequisites import (
                _inspect_capsule,
                _inspect_physics,
                _inspect_stomach,
                _scan_runtime_contract,
                build_gate,
            )
            stage = omni.usd.get_context().get_stage()
            preflight = {
                "repository": {
                    "commit": _git("rev-parse", "HEAD"),
                    "branch": _git("branch", "--show-current"),
                },
                "task": {
                    "id": args.task,
                    "num_envs": 1,
                    "action_terms": list(env.unwrapped.action_manager.active_terms),
                },
                "physics": _inspect_physics(stage, env),
                "capsule": _inspect_capsule(stage, env.unwrapped.scene["capsule"]),
                "stomach": _inspect_stomach(stage),
                "contact_sensor": {"present": "capsule_contact" in env.unwrapped.scene.keys()},
                "runtime_contract": _scan_runtime_contract(term),
                "gate": {},
            }
            preflight["gate"] = build_gate(preflight)
            if preflight["gate"]["status"] != "pass":
                summary = {
                    "preflight": preflight,
                    "settling": {},
                    "directions": {},
                    "continuity": {},
                    "contact": {},
                    "status": "needs_decision",
                }
                return summary, output

            all_rows: list[dict[str, Any]] = []
            directions: dict[str, Any] = {}
            global_max_substep = 0.0
            total_substeps = 0
            with (output / "samples.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
                env.reset(seed=args.seed)
                term = env.unwrapped.action_manager.get_term("dynamic_force")
                settle_rows = _phase(
                    env, term, probe, stream, name="zero_input_settle", direction=(0, 0, 0), steps=180, start_step=0
                )
                all_rows.extend(settle_rows)
                settling = {
                    "initial_state": settle_rows[0],
                    "final_state": settle_rows[-1],
                    "displacement_m": float(
                        np.linalg.norm(
                            np.asarray(settle_rows[-1]["position_world_m"])
                            - np.asarray(settle_rows[0]["position_world_m"])
                        )
                    ),
                    "contact_observed": any(
                        np.linalg.norm(row["contact_force_world_n"]) > 1.0e-6 for row in settle_rows
                    ),
                    "max_speed_m_s": max(np.linalg.norm(row["linear_velocity_world_m_s"]) for row in settle_rows),
                    "max_angular_speed_rad_s": max(np.linalg.norm(row["angular_velocity_world_rad_s"]) for row in settle_rows),
                    "clearance_range_m": [
                        min(row["surface_clearance_m"] for row in settle_rows),
                        max(row["surface_clearance_m"] for row in settle_rows),
                    ],
                    "gravity_enabled": bool(preflight["capsule"]["gravity_enabled"]),
                    "nonfinite_samples": sum(not row["finite"] for row in settle_rows),
                }
                step_cursor = len(settle_rows)
                for index, (name, direction) in enumerate(DIRECTIONS.items()):
                    env.reset(seed=args.seed + index + 1)
                    term = env.unwrapped.action_manager.get_term("dynamic_force")
                    before = _phase(env, term, probe, stream, name=f"{name}_settle", direction=(0, 0, 0), steps=60, start_step=step_cursor)
                    step_cursor += 60
                    active = _phase(env, term, probe, stream, name=f"{name}_active", direction=direction, steps=30, start_step=step_cursor)
                    step_cursor += 30
                    release = _phase(env, term, probe, stream, name=f"{name}_release", direction=(0, 0, 0), steps=30, start_step=step_cursor)
                    step_cursor += 30
                    rows = before + active + release
                    all_rows.extend(rows)
                    expected = np.asarray(direction, dtype=np.float64) * term.mass_kg * 9.81 * args.force_weight_ratio
                    active_forces = np.asarray([row["force_world_n"] for row in active])
                    torques = np.asarray([row["torque_world_nm"] for row in rows])
                    max_substep, substeps = _max_substep_displacement(term)
                    global_max_substep = max(global_max_substep, max_substep)
                    total_substeps += substeps
                    directions[name] = {
                        "requested_direction_world": list(direction),
                        "expected_force_world_n": expected.tolist(),
                        "applied_force_world_n": active_forces[-1].tolist(),
                        "max_force_error_n": float(np.max(np.linalg.norm(active_forces - expected, axis=1))),
                        "max_commanded_torque_nm": float(np.linalg.norm(torques, axis=1).max()),
                        "displacement_m": (
                            np.asarray(rows[-1]["position_world_m"])
                            - np.asarray(rows[0]["position_world_m"])
                        ).tolist(),
                        "velocity_change_m_s": (
                            np.asarray(rows[-1]["linear_velocity_world_m_s"])
                            - np.asarray(rows[0]["linear_velocity_world_m_s"])
                        ).tolist(),
                        "contact_force_norm_range_n": [
                            float(min(np.linalg.norm(row["contact_force_world_n"]) for row in rows)),
                            float(max(np.linalg.norm(row["contact_force_world_n"]) for row in rows)),
                        ],
                        "surface_clearance_range_m": [
                            float(min(row["surface_clearance_m"] for row in rows)),
                            float(max(row["surface_clearance_m"] for row in rows)),
                        ],
                        "max_physics_step_displacement_m": max_substep,
                        "physics_substep_samples": substeps,
                        "boundary_escape": any(row["boundary_escape"] for row in rows),
                    }
            allowed = float(preflight["capsule"]["max_linear_velocity_m_s"]) * float(
                preflight["physics"]["dt_s"]
            ) + 1.0e-5
            continuity = {
                "nonfinite_samples": sum(not row["finite"] for row in all_rows),
                "max_physics_step_displacement_m": global_max_substep,
                "allowed_physics_step_displacement_m": allowed,
                "physics_substep_samples": total_substeps,
                "sustained_clearance_decrease_events": _clearance_decrease_events(all_rows),
            }
            contact = {
                "observed": any(np.linalg.norm(row["contact_force_world_n"]) > 1.0e-6 for row in all_rows),
                "max_force_n": float(max(np.linalg.norm(row["contact_force_world_n"]) for row in all_rows)),
                "minimum_measured_surface_clearance_m": float(min(row["surface_clearance_m"] for row in all_rows)),
                "boundary_escape_observed": any(row["boundary_escape"] for row in all_rows),
                "clearance_method": "five axial centerline samples; exact closest point over 512 nearest triangle centroids; clearance minus 6.5 mm radius",
            }
            summary = {
                "schema_version": "dynamic_force_validation_v1",
                "repository": {
                    "commit": _git("rev-parse", "HEAD"),
                    "branch": _git("branch", "--show-current"),
                },
                "task": args.task,
                "seed": int(args.seed),
                "force_weight_ratio": float(args.force_weight_ratio),
                "preflight": preflight,
                "settling": settling,
                "directions": directions,
                "continuity": continuity,
                "contact": contact,
            }
            summary.update(evaluate_summary(summary))
            return summary, output
        finally:
            env.close()


def main() -> int:
    headless = "--headless" in sys.argv[1:]
    if headless:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_weight_ratio", type=float, default=0.5)
    parser.add_argument("--output_directory", type=Path, default=DEFAULT_OUTPUT)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.task != TASK_ID or args.num_envs != 1:
        parser.error(f"validator requires one environment of {TASK_ID}")
    if not 0.0 < args.force_weight_ratio <= 2.0:
        parser.error("--force_weight_ratio must be in (0, 2]")
    args.device = "cpu"
    args.enable_cameras = True
    launcher = AppLauncher(args)
    try:
        summary, output = _run_validation(args)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"DYNAMIC_FORCE_VALIDATION_{summary['status'].upper()} output={output}", flush=True)
        for failure in summary.get("failures", []):
            print(f"DYNAMIC_FORCE_VALIDATION_FAILURE {failure}", flush=True)
        return 0 if summary["status"] == "pass" else 2
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
