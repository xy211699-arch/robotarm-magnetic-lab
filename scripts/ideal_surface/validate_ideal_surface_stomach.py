#!/usr/bin/env python3
"""Validate all ideal actions and a deterministic long run in the live stomach."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--random_actions", type=int, default=1000)
parser.add_argument("--output", type=Path, default=ROOT / "logs" / "ideal_surface_validation")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli = parser.parse_args()
if args_cli.random_actions < 0:
    parser.error("--random_actions must be non-negative")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import robotarm_magnetic_lab.tasks  # noqa: F401, E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotarm_magnetic_lab.coverage.simulator_runtime import P0CoverageRuntime  # noqa: E402


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = Path(args_cli.output) / session_id
    env = evaluator = None
    records: list[dict] = []
    observed_actions: set[int] = set()
    request_ids: set[int] = set()
    coverage_frame_ids: set[int] = set()
    start_direction_tilts: list[float] = []
    with launch_simulation(env_cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
            env.reset()
            term = env.unwrapped.action_manager.get_term("ideal_surface")
            def execute(action_id: int, phase: str):
                mask = np.asarray(term.action_mask(), dtype=np.bool_).reshape(-1)
                if not 0 <= action_id < len(mask) or not bool(mask[action_id]):
                    raise AssertionError(
                        f"phase={phase} action={action_id} masked mask={mask.astype(int).tolist()}"
                    )
                before = term.controller.snapshot
                before_component = int(
                    term.controller.mesh.component_ids[before.surface_triangle_id]
                )
                action = torch.full(
                    env.action_space.shape,
                    int(action_id),
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
                _, _, terminated, truncated, _ = env.step(action)
                result = term.last_result
                if result is None:
                    raise AssertionError(f"phase={phase}: missing terminal result")
                if result.request_id in request_ids:
                    raise AssertionError(f"duplicate request result {result.request_id}")
                request_ids.add(int(result.request_id))
                if result.status.value != "DONE":
                    raise AssertionError(
                        f"phase={phase} action={action_id} unexpected {result.to_dict()}"
                    )
                if bool(terminated[0] or truncated[0]):
                    raise AssertionError(f"phase={phase} unexpectedly terminated")
                if not (
                    np.isfinite(result.final_position_world).all()
                    and np.isfinite(result.final_quaternion_for_sim).all()
                ):
                    raise AssertionError(f"phase={phase}: nonfinite capsule pose")
                hard_limit = (
                    term.controller.cfg.hard_penetration_radius_fraction
                    * term.controller.capsule.radius_m
                )
                if result.maximum_penetration_m > hard_limit + 1.0e-12:
                    raise AssertionError(f"phase={phase}: hard penetration")
                after = term.controller.snapshot
                after_component = int(
                    term.controller.mesh.component_ids[after.surface_triangle_id]
                )
                if before_component != after_component:
                    raise AssertionError(
                        f"phase={phase}: nonadjacent component jump {before_component}->{after_component}"
                    )
                # Camera frame counters restart with a manager reset.  Coverage
                # therefore starts only after the reset-heavy all-action checks
                # and then runs continuously through the deterministic sequence.
                update = None if evaluator is None else evaluator.maybe_update()
                if update is not None:
                    frame_id = int(update.frame_id)
                    if frame_id in coverage_frame_ids:
                        raise AssertionError(f"duplicate coverage frame update {frame_id}")
                    coverage_frame_ids.add(frame_id)
                record = {
                    "index": len(records) + 1,
                    "phase": phase,
                    "request_id": int(result.request_id),
                    "action_id": int(action_id),
                    "position_world": result.final_position_world.tolist(),
                    "quaternion_wxyz": result.final_quaternion_for_sim.tolist(),
                    "active_triangle": int(after.surface_triangle_id),
                    "component_id": after_component,
                    "maximum_penetration_m": float(result.maximum_penetration_m),
                    "flags": {
                        "upright": bool(after.flags.upright),
                        "side_contact": bool(after.flags.side_contact),
                        "contact_limited": bool(result.contact_limited),
                        "boundary_limited": bool(result.boundary_limited),
                        "no_effect": bool(result.no_effect),
                    },
                    "coverage_frame_id": None if update is None else int(update.frame_id),
                    "coverage_fraction": (
                        None if evaluator is None else float(evaluator.accumulator.mask.mean())
                    ),
                }
                records.append(record)
                observed_actions.add(int(action_id))
                term.acknowledge_result()
                return result

            # Exercise actions valid directly from the approved wall-contact reset.
            # Roll remains masked until the strict 0.1 s separated-barrel gate.
            for action_id in (0, 9, 10, 11, 12):
                env.reset()
                result = execute(action_id, f"direct_action_{action_id}")
                if result.no_effect:
                    raise AssertionError(f"direct action {action_id} unexpectedly had no effect")

            # Each compass direction starts from the same reset and rises to upright first.
            for action_id in range(1, 9):
                env.reset()
                for rise_index in range(7):
                    term.action_mask()
                    if term.controller.snapshot.flags.upright:
                        break
                    execute(10, f"prepare_upright_{action_id}_{rise_index}")
                if not term.controller.snapshot.flags.upright:
                    raise AssertionError(f"failed to reach upright before action {action_id}")
                result = execute(action_id, f"start_direction_{action_id}")
                tilt_deg = float(np.degrees(result.final_tilt_rad))
                tilt_tolerance = 1.0 if result.contact_limited else 0.2
                expected_phi = 45.0 * (action_id - 1)
                actual_phi = float(np.degrees(result.final_azimuth_rad))
                phi_error = abs((actual_phi - expected_phi + 180.0) % 360.0 - 180.0)
                phi_tolerance = 5.0 if result.contact_limited else 2.0
                if (
                    result.no_effect
                    or abs(tilt_deg - 15.0) > tilt_tolerance
                    or phi_error > phi_tolerance
                ):
                    raise AssertionError(
                        f"start direction {action_id} tilt={tilt_deg:.6f} "
                        f"phi={actual_phi:.6f} contact_limited={result.contact_limited}"
                    )
                start_direction_tilts.append(tilt_deg)

            # Reach a geometrically stable side contact before exercising roll.
            roll_pose_found = False
            for direction_action in range(1, 9):
                env.reset()
                for rise_index in range(7):
                    term.action_mask()
                    if term.controller.snapshot.flags.upright:
                        break
                    execute(10, f"roll_prepare_upright_{direction_action}_{rise_index}")
                if not term.controller.snapshot.flags.upright:
                    continue
                execute(direction_action, f"roll_start_{direction_action}")
                for tilt_index in range(5):
                    mask = np.asarray(term.action_mask(), dtype=np.bool_).reshape(-1)
                    if not bool(mask[9]):
                        break
                    execute(9, f"roll_tilt_{direction_action}_{tilt_index}")
                    if term.controller.snapshot.flags.side_contact:
                        roll_pose_found = True
                        break
                if roll_pose_found:
                    break
            if not roll_pose_found:
                raise AssertionError("no compass tilt reached stable separated-barrel side contact")
            for action_id in (13, 14):
                execute(action_id, f"stable_side_roll_{action_id}")

            if observed_actions != set(range(15)):
                raise AssertionError(f"missing action IDs {sorted(set(range(15)) - observed_actions)}")

            # The requested long run starts afresh and samples only valid masks.
            env.reset()
            evaluator = P0CoverageRuntime(
                env,
                output,
                task_id=args_cli.task,
                seed=args_cli.seed,
                commit=_git("rev-parse", "HEAD"),
                branch=_git("branch", "--show-current"),
                enable_view=False,
            )
            rng = np.random.default_rng(args_cli.seed)
            for index in range(args_cli.random_actions):
                mask = np.asarray(term.action_mask(), dtype=np.bool_).reshape(-1)
                valid = np.flatnonzero(mask)
                if not len(valid):
                    raise AssertionError(f"random index {index}: empty action mask")
                action_id = int(rng.choice(valid))
                execute(action_id, f"random_{index + 1:04d}")

            summary = {
                "schema": "ideal_surface_validation_v1",
                "seed": int(args_cli.seed),
                "random_action_count": int(args_cli.random_actions),
                "total_action_count": len(records),
                "observed_action_ids": sorted(observed_actions),
                "unique_result_count": len(request_ids),
                "coverage_update_count": len(coverage_frame_ids),
                "coverage_frame_ids_unique": True,
                "start_direction_final_tilt_deg": start_direction_tilts,
                "final_coverage_fraction": float(evaluator.accumulator.mask.mean()),
                "status": "PASS",
            }
            with (evaluator.partial_directory / "ideal_surface_records.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            (evaluator.partial_directory / "ideal_surface_validation.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print("IDEAL_SURFACE_STOMACH_PASS " + json.dumps(summary, sort_keys=True), flush=True)
            return 0
        finally:
            if evaluator is not None:
                final = evaluator.finalize("validation")
                print(f"IDEAL_SURFACE_VALIDATION_OUTPUT {final}", flush=True)
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
