#!/usr/bin/env python3
"""Validate TASK-009B's first gate with a live six-mode 10 Hz sequence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
HEADLESS = "--headless" in sys.argv
if HEADLESS:
    sys.argv.remove("--headless")
    os.environ["HEADLESS"] = "1"

from isaaclab.app import AppLauncher


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument(
    "--output_directory",
    type=Path,
    default=Path("/tmp/task009b-environment-integration"),
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this validator only accepts {TASK_ID}")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    CONTROL_HZ,
    PHYSICS_HZ,
    PHYSICS_STEPS_PER_CONTROL,
    ParameterizedForceMode,
)


SEQUENCE = (
    (ParameterizedForceMode.HOLD, 0.0),
    (ParameterizedForceMode.MOVE_POS, 0.5),
    (ParameterizedForceMode.MOVE_NEG, 0.5),
    (ParameterizedForceMode.VIEW_POS, 0.5),
    (ParameterizedForceMode.VIEW_NEG, 0.5),
    (ParameterizedForceMode.UP, 0.5),
)


def frame_id(camera) -> int:
    value = getattr(camera, "frame", None)
    if value is None:
        value = getattr(camera.data, "frame", None)
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy().reshape(-1)[0]
    return int(value)


def actor_keys(observation) -> list[str]:
    result = []
    for group, terms in observation.items():
        if isinstance(terms, dict):
            result.extend(f"{group}.{name}" for name in terms)
        else:
            result.append(str(group))
    return sorted(result)


def finite_tensor(value) -> bool:
    tensor = getattr(value, "torch", value)
    return bool(torch.isfinite(tensor).all().item())


def main() -> int:
    output = args_cli.output_directory / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    env = None
    rows = []
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            observation, _ = env.reset()
            base = env.unwrapped
            term = base.action_manager.get_term("parameterized_force")
            camera = base.scene["capsule_camera"]
            for cycle, (mode, alpha) in enumerate(SEQUENCE):
                start_time = float(base.common_step_counter * base.step_dt)
                start_frame = frame_id(camera)
                action = torch.tensor(
                    [[float(mode), alpha]],
                    dtype=torch.float32,
                    device=base.device,
                )
                observation, *_ = env.step(action)
                end_time = float(base.common_step_counter * base.step_dt)
                end_frame = frame_id(camera)
                trace = term.current_cycle_trace
                active_substeps = sum(
                    math.sqrt(sum(value * value for value in item.submitted_force_world)) > 0.0
                    or math.sqrt(sum(value * value for value in item.submitted_torque_world)) > 0.0
                    for item in trace
                )
                keys = actor_keys(observation)
                finite = (
                    finite_tensor(base.scene["capsule"].data.root_pose_w)
                    and finite_tensor(base.scene["capsule"].data.root_com_vel_w)
                    and finite_tensor(camera.data.output["rgb"])
                )
                row = {
                    "cycle": cycle,
                    "mode": mode.name,
                    "mode_id": int(mode),
                    "alpha": alpha,
                    "physics_substeps": len(trace),
                    "physics_step_indices": [item.physics_step_in_cycle for item in trace],
                    "active_force_substeps": int(active_substeps),
                    "start_sim_time_s": start_time,
                    "end_sim_time_s": end_time,
                    "duration_sim_time_s": end_time - start_time,
                    "start_rgb_frame_id": start_frame,
                    "end_rgb_frame_id": end_frame,
                    "rgb_frame_updates": end_frame - start_frame,
                    "actor_observation_keys": keys,
                    "finite_state_and_rgb": finite,
                    "last_telemetry": asdict(term.last_telemetry),
                }
                expected_active = 0 if mode == ParameterizedForceMode.HOLD else 24
                if len(trace) != PHYSICS_STEPS_PER_CONTROL:
                    raise RuntimeError(f"cycle {cycle}: expected 24 substeps, got {len(trace)}")
                if row["physics_step_indices"] != list(range(PHYSICS_STEPS_PER_CONTROL)):
                    raise RuntimeError(f"cycle {cycle}: invalid physics substep indices")
                if active_substeps != expected_active:
                    raise RuntimeError(
                        f"cycle {cycle}: expected {expected_active} active substeps, got {active_substeps}"
                    )
                if not math.isclose(end_time - start_time, 0.1, abs_tol=1.0e-9):
                    raise RuntimeError(f"cycle {cycle}: action boundary is not 0.1 s")
                if end_frame - start_frame != 1:
                    raise RuntimeError(f"cycle {cycle}: expected one RGB update, got {end_frame-start_frame}")
                if keys != ["policy.rgb"]:
                    raise RuntimeError(f"cycle {cycle}: Actor observation is not RGB-only: {keys}")
                if not finite:
                    raise RuntimeError(f"cycle {cycle}: non-finite state or RGB")
                rows.append(row)
                with (output / "environment_cycles.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    "TASK009B_ENV_CYCLE "
                    f"cycle={cycle} mode={mode.name} alpha={alpha:.1f} "
                    f"substeps={len(trace)} active={active_substeps} "
                    f"sim={start_time:.3f}->{end_time:.3f} "
                    f"rgb={start_frame}->{end_frame} finite={finite}",
                    flush=True,
                )
            term.capsule.permanent_wrench_composer.reset()
            summary = {
                "schema": "task009b_environment_integration_v1",
                "status": "pass",
                "task": args_cli.task,
                "physics_hz": PHYSICS_HZ,
                "control_hz": CONTROL_HZ,
                "physics_steps_per_control": PHYSICS_STEPS_PER_CONTROL,
                "cycles": len(rows),
                "output": str(output),
            }
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"TASK009B_ENV_PASS {json.dumps(summary, sort_keys=True)}", flush=True)
            return 0
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()

