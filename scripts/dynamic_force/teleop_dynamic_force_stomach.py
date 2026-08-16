#!/usr/bin/env python3
"""Continuously apply held world-frame forces to the dynamic stomach capsule."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

HEADLESS = "--headless" in sys.argv
if HEADLESS:
    sys.argv.remove("--headless")
    os.environ["HEADLESS"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402


TASK_ID = "Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0"
DEFAULT_OUTPUT = Path(
    "/mnt/isaac-linux/robotarm_magnetic_lab/logs/dynamic_force_teleop"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--force_weight_ratio", type=float, default=0.9)
parser.add_argument("--vertical_force_weight_ratio", type=float, default=1.1)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument(
    "--scripted_sequence",
    default="",
    help='Comma-separated world directions and durations, e.g. "+x:0.5,zero:0.25".',
)
parser.add_argument("--output_directory", type=Path, default=DEFAULT_OUTPUT)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this launcher accepts only {TASK_ID}")
if args_cli.num_envs != 1:
    parser.error("dynamic-force teleoperation requires --num_envs 1")
if args_cli.max_steps < 0:
    parser.error("--max_steps must be non-negative")
if not 0.0 < args_cli.force_weight_ratio <= 2.0:
    parser.error("--force_weight_ratio must be in (0, 2]")
if not 0.0 < args_cli.vertical_force_weight_ratio <= 2.0:
    parser.error("--vertical_force_weight_ratio must be in (0, 2]")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import robotarm_magnetic_lab.tasks  # noqa: F401, E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotarm_magnetic_lab.teleop import (  # noqa: E402
    DynamicForceCommandKind,
    DynamicForceKeyboard,
)
from robotarm_magnetic_lab.ui import (  # noqa: E402
    attach_capsule_camera_policy_view,
    configure_capsule_camera_view,
)


DIRECTION_VECTORS = {
    "+x": np.asarray([1.0, 0.0, 0.0]),
    "-x": np.asarray([-1.0, 0.0, 0.0]),
    "+y": np.asarray([0.0, 1.0, 0.0]),
    "-y": np.asarray([0.0, -1.0, 0.0]),
    "+z": np.asarray([0.0, 0.0, 1.0]),
    "-z": np.asarray([0.0, 0.0, -1.0]),
    "zero": np.zeros(3),
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _flat(value) -> np.ndarray:
    return value.torch.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _parse_scripted_sequence(value: str) -> deque[np.ndarray]:
    steps: deque[np.ndarray] = deque()
    if not value.strip():
        return steps
    for item in value.split(","):
        try:
            name, duration_text = item.strip().lower().split(":", 1)
            duration = float(duration_text)
        except ValueError as exc:
            raise ValueError(f"invalid scripted segment: {item!r}") from exc
        if name not in DIRECTION_VECTORS:
            raise ValueError(f"unknown scripted direction {name!r}")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("scripted durations must be finite and positive")
        count = max(1, int(round(duration * 60.0)))
        steps.extend(DIRECTION_VECTORS[name].copy() for _ in range(count))
    return steps


class KitKeyboardSource:
    def __init__(self) -> None:
        self.commands = deque()
        self.keyboard = DynamicForceKeyboard()
        self._input = carb.input.acquire_input_interface()
        self._device = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._device, self._on_event
        )

    def _on_event(self, event, *_args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            command = self.keyboard.key_event(event.input.name, True)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            command = self.keyboard.key_event(event.input.name, False)
        else:
            command = None
        if command is not None:
            self.commands.append(command)
        return True

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._device, self._subscription)
            self._subscription = None
        self.keyboard.release_all()


@dataclass
class SessionRecorder:
    output: Path
    samples_file: object
    started_wall_s: float
    samples: int = 0
    resets: int = 0
    snapshots: int = 0

    @classmethod
    def create(cls, output_root: Path):
        output = output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%fZ"
        )
        (output / "snapshots").mkdir(parents=True, exist_ok=False)
        stream = (output / "samples.jsonl").open("w", encoding="utf-8", buffering=1)
        return cls(output, stream, time.monotonic())

    def append(self, row: dict) -> None:
        self.samples_file.write(json.dumps(row, sort_keys=True) + "\n")
        self.samples += 1

    def snapshot(self, row: dict, rgb: np.ndarray | None, label: str) -> None:
        self.snapshots += 1
        stem = f"snapshot_{self.snapshots:04d}_{label}"
        (self.output / "snapshots" / f"{stem}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if rgb is not None:
            from PIL import Image

            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
                self.output / "snapshots" / f"{stem}.png"
            )

    def close(self, *, reason: str, sim_time_s: float) -> Path:
        self.samples_file.close()
        elapsed = time.monotonic() - self.started_wall_s
        session = {
            "schema_version": "dynamic_force_teleop_v1",
            "task": args_cli.task,
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "reason": reason,
            "samples": self.samples,
            "resets": self.resets,
            "snapshots": self.snapshots,
            "sim_time_s": float(sim_time_s),
            "wall_time_s": float(elapsed),
            "measured_environment_rate_wall_hz": (
                0.0 if elapsed <= 0.0 else float(self.samples / elapsed)
            ),
            "physics_rate_sim_hz": 240.0,
            "environment_rate_sim_hz": 60.0,
            "render_rate_sim_hz": 60.0,
            "capsule_camera_rate_sim_hz": 30.0,
            "force_weight_ratio": float(args_cli.force_weight_ratio),
            "vertical_force_weight_ratio": float(args_cli.vertical_force_weight_ratio),
        }
        (self.output / "session.json").write_text(
            json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.output


def _sample(env, term, direction: np.ndarray, step: int, reset_index: int) -> dict:
    base = env.unwrapped
    capsule = base.scene["capsule"]
    contact = base.scene["capsule_contact"]
    pose = _flat(capsule.data.root_com_pose_w)
    velocity = _flat(capsule.data.root_com_vel_w)
    contact_force = _flat(contact.data.net_forces_w)[0:3]
    return {
        "sim_time_s": float(base.episode_length_buf[0].item()) * float(base.step_dt),
        "step": int(step),
        "reset_index": int(reset_index),
        "position_world_m": pose[:3].tolist(),
        "quaternion_wxyz": pose[3:7].tolist(),
        "linear_velocity_world_m_s": velocity[:3].tolist(),
        "angular_velocity_world_rad_s": velocity[3:6].tolist(),
        "direction_world": np.asarray(direction, dtype=np.float64).tolist(),
        "force_world_n": term.applied_force_world[0].detach().cpu().tolist(),
        "torque_world_nm": term.applied_torque_world[0].detach().cpu().tolist(),
        "contact_force_world_n": contact_force.tolist(),
    }


def _camera_rgb(env) -> np.ndarray | None:
    image = env.unwrapped.scene["capsule_camera"].data.output.get("rgb")
    if image is None:
        return None
    return image[0, ..., :3].detach().cpu().numpy()


def main() -> int:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.actions.dynamic_force.force_weight_ratio = args_cli.force_weight_ratio
    env_cfg.actions.dynamic_force.vertical_force_weight_ratio = (
        args_cli.vertical_force_weight_ratio
    )
    env_cfg.sim.device = "cpu"
    if not HEADLESS:
        configure_capsule_camera_view(env_cfg)
    scripted = _parse_scripted_sequence(args_cli.scripted_sequence)
    recorder = SessionRecorder.create(args_cli.output_directory)
    print(f"DYNAMIC_FORCE_OUTPUT_DIRECTORY {recorder.output}", flush=True)
    env = keyboard = view_handle = None
    exit_code = 1
    reason = "initialization_failed"
    sim_time_s = 0.0
    with launch_simulation(env_cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
            env.reset()
            term = env.unwrapped.action_manager.get_term("dynamic_force")
            if not HEADLESS:
                keyboard = KitKeyboardSource()
                view_handle = attach_capsule_camera_policy_view(env)
            print(
                "DYNAMIC_FORCE_READY hold W/S=+X/-X, A/D=+Y/-Y, Q/E=+Z/-Z; "
                "Space=clear, Backspace=reset, F12=snapshot, Esc=exit",
                flush=True,
            )
            reset_index = 0
            step = 0
            exit_requested = False
            previous_direction = np.full(3, np.nan)
            while simulation_app.is_running() and not exit_requested:
                if keyboard is not None:
                    while keyboard.commands:
                        command = keyboard.commands.popleft()
                        if command.kind is DynamicForceCommandKind.EXIT:
                            exit_requested = True
                        elif command.kind is DynamicForceCommandKind.RESET:
                            env.reset()
                            keyboard.keyboard.release_all()
                            recorder.resets += 1
                            reset_index += 1
                        elif command.kind is DynamicForceCommandKind.SNAPSHOT:
                            row = _sample(env, term, keyboard.keyboard.direction, step, reset_index)
                            recorder.snapshot(row, _camera_rgb(env), "f12")
                        elif command.kind is DynamicForceCommandKind.CLEAR:
                            pass
                if exit_requested:
                    reason = "user_exit"
                    break
                if scripted:
                    direction = scripted.popleft()
                elif args_cli.scripted_sequence:
                    # A finite --max_steps is an acceptance horizon.  Hold a
                    # released (zero) command after the scripted segments so
                    # the requested rendered sample count is still produced.
                    if args_cli.max_steps and step < args_cli.max_steps:
                        direction = np.zeros(3)
                    else:
                        reason = "script_complete"
                        break
                elif keyboard is not None:
                    direction = keyboard.keyboard.direction
                else:
                    direction = np.zeros(3)
                action = torch.as_tensor(
                    direction, device=env.unwrapped.device, dtype=torch.float32
                ).reshape(1, 3)
                _, _, terminated, truncated, _ = env.step(action)
                step += 1
                row = _sample(env, term, direction, step, reset_index)
                sim_time_s = row["sim_time_s"]
                recorder.append(row)
                if not np.array_equal(direction, previous_direction):
                    print(
                        f"DYNAMIC_FORCE_COMMAND direction={row['direction_world']} "
                        f"force_N={row['force_world_n']}",
                        flush=True,
                    )
                    previous_direction = direction.copy()
                if step % 6 == 0:
                    print(
                        f"DYNAMIC_FORCE_STATE t={sim_time_s:.3f}s "
                        f"pos={np.round(row['position_world_m'], 6).tolist()} "
                        f"vel={np.round(row['linear_velocity_world_m_s'], 5).tolist()} "
                        f"contact_N={np.round(row['contact_force_world_n'], 5).tolist()}",
                        flush=True,
                    )
                if bool(terminated[0] or truncated[0]):
                    reason = "environment_termination"
                    recorder.snapshot(row, _camera_rgb(env), "termination")
                    exit_code = 2
                    break
                if args_cli.max_steps and step >= args_cli.max_steps:
                    reason = "max_steps"
                    break
            if exit_code != 2:
                exit_code = 0
        finally:
            if view_handle is not None:
                view_handle.close()
            if keyboard is not None:
                keyboard.close()
            if env is not None:
                env.close()
    output = recorder.close(reason=reason, sim_time_s=sim_time_s)
    print(f"DYNAMIC_FORCE_OUTPUT {output}", flush=True)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
