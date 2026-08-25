#!/usr/bin/env python3
"""Calibrate 0.1 s MOVE displacement from stable flat-table contact."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
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


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Table-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument(
    "--ratios",
    default="0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80,0.90,1.00,1.10",
    help="Comma-separated MOVE resultant-force ratios in units of capsule mg.",
)
parser.add_argument("--samples", type=int, default=5, help="Stable side-lying yaw states per MOVE direction.")
parser.add_argument("--settle_max_cycles", type=int, default=20)
parser.add_argument(
    "--minimum_settle_cycles",
    type=int,
    default=2,
    help="Minimum 0.1 s HOLD cycles before a trial, even if contact is already stable.",
)
parser.add_argument("--stable_linear_speed", type=float, default=0.002)
parser.add_argument("--stable_angular_speed", type=float, default=0.10)
parser.add_argument("--output_directory", type=Path, default=Path("/tmp/move-stable-contact-calibration"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this calibration only accepts {TASK_ID}")
if args_cli.samples < 5:
    parser.error("--samples must be at least 5")
if args_cli.minimum_settle_cycles < 2:
    parser.error("--minimum_settle_cycles must be at least 2")
if args_cli.minimum_settle_cycles > args_cli.settle_max_cycles:
    parser.error("--minimum_settle_cycles cannot exceed --settle_max_cycles")
try:
    ratios = sorted(set(float(value) for value in args_cli.ratios.split(",")))
except ValueError as exc:
    parser.error(f"invalid --ratios: {exc}")
if not ratios or any(not 0.0 < value <= 3.0 for value in ratios):
    parser.error("all MOVE ratios must be in (0, 3]")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
import torch

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.runtime.move_displacement import corrected_move_displacement
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    GRAVITY_M_S2,
    ParameterizedForceMode,
    horizontal_lateral_direction,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_lab_env_cfg import (
    CAPSULE_START_POS,
)


def _torch(value):
    return getattr(value, "torch", value)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


class StableContactRunner:
    def __init__(self, env) -> None:
        self.env = env
        self.base = env.unwrapped
        self.capsule = self.base.scene["capsule"]
        self.contact = self.base.scene["capsule_contact"]
        self.term = self.base.action_manager.get_term("parameterized_force")
        self.device = self.base.device

    def step(self, mode: ParameterizedForceMode, alpha: float = 0.5) -> None:
        action = torch.tensor([[float(mode), float(alpha)]], device=self.device, dtype=torch.float32)
        self.env.step(action)

    def contact_count(self) -> int:
        values = _torch(self.contact.data.net_forces_w)[0].detach().cpu().numpy().reshape(-1, 3)
        return int(np.count_nonzero(np.linalg.norm(values, axis=1) >= 1.0e-4))

    def state(self) -> dict:
        pose = _torch(self.capsule.data.root_pose_w)[0].detach().cpu().numpy().astype(np.float64)
        velocity = _torch(self.capsule.data.root_com_vel_w)[0].detach().cpu().numpy().astype(np.float64)
        com, camera, other, axis, _quat = self.term._geometry()
        return {
            "pose_xyzw": pose,
            "com": com,
            "camera": camera,
            "other": other,
            "axis": axis,
            "linear_velocity": velocity[:3],
            "angular_velocity": velocity[3:],
            "contact_count": self.contact_count(),
        }

    def _stable(self, state: dict) -> bool:
        return bool(
            np.linalg.norm(state["linear_velocity"]) <= args_cli.stable_linear_speed
            and np.linalg.norm(state["angular_velocity"]) <= args_cli.stable_angular_speed
        )

    def settle(self) -> tuple[dict, int]:
        consecutive = 0
        contact_observed = False
        for cycle in range(1, args_cli.settle_max_cycles + 1):
            self.step(ParameterizedForceMode.HOLD)
            state = self.state()
            contact_observed = contact_observed or state["contact_count"] > 0
            consecutive = consecutive + 1 if self._stable(state) else 0
            # A sleeping PhysX actor can report zero instantaneous contact force even
            # though it remains supported by the table.  Require contact to have been
            # observed during this warm-up, then use kinematics at the trial boundary.
            if cycle >= args_cli.minimum_settle_cycles and consecutive >= 2 and contact_observed:
                return state, cycle
        state = self.state()
        raise RuntimeError(
            "stable contact not reached: "
            f"contact={state['contact_count']} "
            f"linear={np.linalg.norm(state['linear_velocity']):.6g} "
            f"angular={np.linalg.norm(state['angular_velocity']):.6g}"
        )

    def initialize_yaw(self, yaw_rad: float, seed: int) -> dict:
        self.env.reset(seed=seed)
        pose = _torch(self.capsule.data.root_pose_w).clone()
        pose[0, :3] = torch.tensor(CAPSULE_START_POS, device=self.device, dtype=torch.float32)
        pose[0, 2] = max(float(pose[0, 2]), 0.0067)
        rotation = Rotation.from_euler("Z", yaw_rad) * Rotation.from_euler("Y", np.pi / 2.0)
        pose[0, 3:7] = torch.as_tensor(rotation.as_quat(), device=self.device, dtype=torch.float32)
        self.capsule.write_root_pose_to_sim_index(root_pose=pose)
        self.capsule.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=self.device, dtype=torch.float32)
        )
        self.term.reset()
        state, _cycles = self.settle()
        return {"pose_xyzw": state["pose_xyzw"].copy(), "yaw_rad": yaw_rad, "seed": seed}

    def restore_and_warm(self, stable_state: dict) -> tuple[dict, int]:
        self.env.reset(seed=int(stable_state["seed"]))
        pose = torch.as_tensor(stable_state["pose_xyzw"], device=self.device, dtype=torch.float32).reshape(1, 7)
        self.capsule.write_root_pose_to_sim_index(root_pose=pose)
        self.capsule.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=self.device, dtype=torch.float32)
        )
        self.term.reset()
        return self.settle()

    def response(self, stable_state: dict, mode: ParameterizedForceMode, ratio: float | None) -> dict:
        start, warm_cycles = self.restore_and_warm(stable_state)
        if ratio is not None:
            self.term.config = replace(
                self.term.config,
                move_min_ratio=float(ratio),
                move_max_ratio=float(ratio),
            )
        self.step(mode)
        end = self.state()
        return {"start": start, "end": end, "warm_cycles": warm_cycles}


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for ratio in ratios:
        selected = [row for row in rows if row["ratio_mg"] == ratio]
        if not selected:
            continue
        values = np.asarray([row["corrected_signed_mm"] for row in selected], dtype=np.float64)
        active = np.asarray([row["active_signed_mm"] for row in selected], dtype=np.float64)
        hold = np.asarray([row["hold_signed_mm"] for row in selected], dtype=np.float64)
        result.append(
            {
                "ratio_mg": ratio,
                "samples": len(values),
                "mean_corrected_mm": float(np.mean(values)),
                "median_corrected_mm": float(np.median(values)),
                "std_corrected_mm": float(np.std(values, ddof=1)),
                "min_corrected_mm": float(np.min(values)),
                "max_corrected_mm": float(np.max(values)),
                "mean_active_mm": float(np.mean(active)),
                "mean_hold_mm": float(np.mean(hold)),
                "direction_correct_rate": float(np.mean(values > 0.0)),
            }
        )
    return result


def main() -> int:
    output = args_cli.output_directory / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    cfg.sim.render_interval = 24
    rows = []
    env = None
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            runner = StableContactRunner(env)
            mass_kg = float(_torch(runner.capsule.data.body_mass).reshape(-1)[0].item())
            stable_states = [
                runner.initialize_yaw(2.0 * np.pi * index / args_cli.samples, 825100 + index)
                for index in range(args_cli.samples)
            ]
            baselines = {}
            for index, stable_state in enumerate(stable_states):
                baseline = runner.response(stable_state, ParameterizedForceMode.HOLD, None)
                baselines[index] = baseline

            for ratio in ratios:
                for mode in (ParameterizedForceMode.MOVE_POS, ParameterizedForceMode.MOVE_NEG):
                    for index, stable_state in enumerate(stable_states):
                        active = runner.response(stable_state, mode, ratio)
                        direction = horizontal_lateral_direction(
                            active["start"]["axis"],
                            negative=mode == ParameterizedForceMode.MOVE_NEG,
                        )
                        baseline = baselines[index]
                        metric = corrected_move_displacement(
                            active["start"]["com"],
                            active["end"]["com"],
                            baseline["start"]["com"],
                            baseline["end"]["com"],
                            direction,
                        )
                        row = {
                            "ratio_mg": ratio,
                            "mode": mode.name,
                            "sample": index,
                            "yaw_deg": float(np.degrees(stable_state["yaw_rad"])),
                            "active_signed_mm": 1000.0 * metric.active_signed_m,
                            "hold_signed_mm": 1000.0 * metric.hold_signed_m,
                            "corrected_signed_mm": 1000.0 * metric.corrected_signed_m,
                            "direction_world": direction,
                            "active_start_contact_count": active["start"]["contact_count"],
                            "active_start_linear_speed_m_s": float(np.linalg.norm(active["start"]["linear_velocity"])),
                            "active_start_angular_speed_rad_s": float(np.linalg.norm(active["start"]["angular_velocity"])),
                            "active_warm_cycles": active["warm_cycles"],
                            "hold_warm_cycles": baseline["warm_cycles"],
                        }
                        rows.append(row)
                        _append_jsonl(output / "trials.jsonl", row)
                ratio_summary = summarize(rows)[-1]
                print(
                    "MOVE_CALIBRATION_RATIO "
                    f"ratio_mg={ratio:.4f} n={ratio_summary['samples']} "
                    f"mean_mm={ratio_summary['mean_corrected_mm']:.4f} "
                    f"std_mm={ratio_summary['std_corrected_mm']:.4f} "
                    f"range_mm=[{ratio_summary['min_corrected_mm']:.4f},{ratio_summary['max_corrected_mm']:.4f}]",
                    flush=True,
                )

            levels = summarize(rows)
            in_band = [row for row in levels if 1.0 <= row["mean_corrected_mm"] <= 3.0]
            summary = {
                "schema": "stable_contact_move_displacement_calibration_v1",
                "formula": "dot(active_end-active_start, command_dir) - dot(hold_end-hold_start, command_dir)",
                "physics_hz": 240,
                "control_hz": 10,
                "active_duration_s": 0.1,
                "mass_kg": mass_kg,
                "gravity_m_s2": GRAVITY_M_S2,
                "mg_n": mass_kg * GRAVITY_M_S2,
                "samples_per_direction": args_cli.samples,
                "stable_contact_required": True,
                "minimum_settle_cycles": args_cli.minimum_settle_cycles,
                "settle_max_cycles": args_cli.settle_max_cycles,
                "levels": levels,
                "tested_band": None
                if not in_band
                else {"lower_ratio_mg": in_band[0]["ratio_mg"], "upper_ratio_mg": in_band[-1]["ratio_mg"]},
            }
            _write_json(output / "summary.json", summary)
            print(
                "MOVE_CALIBRATION_COMPLETE "
                f"output={output} tested_band={summary['tested_band']}",
                flush=True,
            )
            return 0
        finally:
            if env is not None:
                env.close()
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
