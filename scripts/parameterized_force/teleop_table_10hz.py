#!/usr/bin/env python3
"""Interactive flat-table visualization of the true 10 Hz force interface."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


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
parser.add_argument("--initial_alpha", type=float, choices=(0.0, 0.5, 1.0), default=0.5)
parser.add_argument(
    "--render_fps",
    type=int,
    choices=(30, 60, 120, 240),
    default=120,
    help="External viewport render frequency; physics remains fixed at 240 Hz.",
)
parser.add_argument("--output_directory", type=Path, default=Path("/tmp/parameterized-force-table-10hz"))
parser.add_argument("--max_cycles", type=int, default=0)
parser.add_argument(
    "--scripted_actions",
    default="",
    help="Headless smoke format: MODE:ALPHA:CYCLES, e.g. MOVE_POS:0.5:2,HOLD:0.5:1",
)
parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this launcher only accepts {TASK_ID}")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import carb
import gymnasium as gym
import numpy as np
import omni.appwindow
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
from robotarm_magnetic_lab.teleop import (
    ParameterizedForceKeyboard,
    ParameterizedKeyboardEventKind,
)
from robotarm_magnetic_lab.ui import (
    attach_capsule_camera_policy_view,
    attach_capsule_pose_view,
    configure_capsule_camera_view,
    configure_capsule_pose_view,
)


class KitHeldKeyboard:
    def __init__(self, alpha: float) -> None:
        self.state = ParameterizedForceKeyboard(alpha)
        self.events = deque()
        self._input = carb.input.acquire_input_interface()
        self._device = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._device, self._on_event)

    def _on_event(self, event, *_args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            update = self.state.key_event(event.input.name, True)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            update = self.state.key_event(event.input.name, False)
        else:
            update = None
        if update is not None:
            self.events.append(update)
        return True

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._device, self._subscription)
            self._subscription = None
        self.state.release_all()


def parse_scripted_actions(value: str) -> deque[tuple[ParameterizedForceMode, float]]:
    cycles: deque[tuple[ParameterizedForceMode, float]] = deque()
    for token in filter(None, (item.strip() for item in value.split(","))):
        try:
            name, alpha_text, count_text = token.split(":")
            mode = ParameterizedForceMode[name.strip().upper()]
            alpha = float(alpha_text)
            count = int(count_text)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid scripted action {token!r}") from exc
        if alpha not in (0.0, 0.5, 1.0) or count <= 0:
            raise ValueError(f"scripted action must use alpha 0/0.5/1 and positive cycles: {token!r}")
        cycles.extend((mode, alpha) for _ in range(count))
    return cycles


def save_rgb(path: Path, rgb) -> None:
    from PIL import Image

    value = getattr(rgb, "torch", rgb)
    array = value[0].detach().cpu().numpy()[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * (255.0 if array.max(initial=0.0) <= 1.0 else 1.0), 0, 255)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _frame_id(camera) -> int | None:
    # Isaac Lab 3 exposes the monotonic sensor counter on Camera.frame.  Keep
    # the data.frame fallback for compatibility with older sensor wrappers.
    value = getattr(camera, "frame", None)
    if value is None:
        value = getattr(camera.data, "frame", None)
    if value is None:
        return None
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy().reshape(-1)[0]
    return int(value)


def _actor_observation_keys(value, prefix="") -> list[str]:
    if isinstance(value, Mapping):
        result = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_actor_observation_keys(item, child))
        return result
    return [prefix]


def _boundary_state(base, term) -> dict:
    capsule = base.scene["capsule"]
    pose = capsule.data.root_pose_w.torch[0].detach().cpu().numpy()
    velocity = capsule.data.root_com_vel_w.torch[0].detach().cpu().numpy()
    com, _camera, _other, axis, _quaternion = term._geometry()
    return {
        "pose_world_xyzw": pose.astype(np.float64),
        "com_world": np.asarray(com, dtype=np.float64),
        "axis_world": np.asarray(axis, dtype=np.float64),
        "com_velocity_world": velocity.astype(np.float64),
    }


def cycle_record(
    term,
    mode,
    alpha,
    cycle_index: int,
    start_state: dict,
    end_state: dict,
    start_sim_time_s: float,
    end_sim_time_s: float,
    start_rgb_frame_id: int | None,
    end_rgb_frame_id: int | None,
    observation,
) -> dict:
    telemetry = None if term.last_telemetry is None else asdict(term.last_telemetry)
    trace = term.current_cycle_trace
    axis_dot = float(np.clip(np.dot(start_state["axis_world"], end_state["axis_world"]), -1.0, 1.0))
    axis_change_deg = float(np.degrees(np.arccos(abs(axis_dot))))
    displacement = end_state["com_world"] - start_state["com_world"]
    finite_values = np.concatenate(
        (
            end_state["pose_world_xyzw"],
            end_state["com_world"],
            end_state["axis_world"],
            end_state["com_velocity_world"],
        )
    )
    observation_keys = sorted(_actor_observation_keys(observation))
    forbidden = ("pose", "velocity", "contact", "normal", "coverage", "mask", "privileged")
    leaked = [key for key in observation_keys if any(token in key.lower() for token in forbidden)]
    if leaked:
        raise RuntimeError(f"Actor observation contains forbidden simulation truth: {leaked}")
    return {
        "cycle": cycle_index,
        "start_sim_time_s": start_sim_time_s,
        "end_sim_time_s": end_sim_time_s,
        "duration_sim_s": end_sim_time_s - start_sim_time_s,
        "mode": mode.name,
        "mode_id": int(mode),
        "alpha": float(alpha),
        "force_ratio_mg": None if telemetry is None else telemetry["force_ratio"],
        "target_total_force_n": None if telemetry is None else telemetry["target_total_force_n"],
        "physics_substeps": len(trace),
        "physics_step_indices": [item.physics_step_in_cycle for item in trace],
        "force_active_substeps": int(
            sum(
                np.linalg.norm(item.submitted_force_world) > 0.0
                or np.linalg.norm(item.submitted_torque_world) > 0.0
                for item in trace
            )
        ),
        "start_rgb_frame_id": start_rgb_frame_id,
        "end_rgb_frame_id": end_rgb_frame_id,
        "actor_observation_keys": observation_keys,
        "com_displacement_world_m": displacement.tolist(),
        "com_displacement_norm_m": float(np.linalg.norm(displacement)),
        "axis_change_deg_unoriented": axis_change_deg,
        "pose_world_xyzw": end_state["pose_world_xyzw"].tolist(),
        "com_velocity_world": end_state["com_velocity_world"].tolist(),
        "finite_state": bool(np.isfinite(finite_values).all()),
        "telemetry": telemetry,
    }


def main() -> int:
    scripted = parse_scripted_actions(args_cli.scripted_actions)
    output = args_cli.output_directory / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    cfg.sim.render_interval = PHYSICS_HZ // args_cli.render_fps
    if not HEADLESS:
        configure_capsule_camera_view(cfg)
        configure_capsule_pose_view(cfg)

    env = keyboard = camera_view = pose_view = None
    cycle_index = 0
    exit_reason = "initialization_failed"
    active_signature = None
    last_observation = None
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            last_observation, _ = env.reset()
            base = env.unwrapped
            term = base.action_manager.get_term("parameterized_force")
            if not HEADLESS:
                keyboard = KitHeldKeyboard(args_cli.initial_alpha)
                camera_view = attach_capsule_camera_policy_view(env)
                pose_view = attach_capsule_pose_view(env)
            print(
                "PARAM_FORCE_10HZ_READY "
                f"physics_hz={PHYSICS_HZ} control_hz={CONTROL_HZ} "
                f"physics_steps_per_control={PHYSICS_STEPS_PER_CONTROL} "
                f"render_fps={args_cli.render_fps} "
                "keys='hold A/D MOVE-/+; hold Q/E VIEW-/+; hold W UP; "
                "Z/X/C alpha=0/0.5/1; SPACE HOLD; R reset; P snapshot; ESC exit'",
                flush=True,
            )
            exit_reason = "simulation_closed"
            while simulation_app.is_running():
                reset_requested = snapshot_requested = exit_requested = False
                if keyboard is not None:
                    while keyboard.events:
                        event = keyboard.events.popleft()
                        reset_requested |= event.kind is ParameterizedKeyboardEventKind.RESET
                        snapshot_requested |= event.kind is ParameterizedKeyboardEventKind.SNAPSHOT
                        exit_requested |= event.kind is ParameterizedKeyboardEventKind.EXIT
                    mode, alpha = keyboard.state.command
                elif scripted:
                    mode, alpha = scripted.popleft()
                else:
                    exit_reason = "script_complete"
                    break

                if exit_requested:
                    exit_reason = "keyboard_exit"
                    break
                if reset_requested:
                    last_observation, _ = env.reset()
                    keyboard.state.release_all()
                    active_signature = None
                    print("PARAM_FORCE_10HZ_RESET", flush=True)
                    continue
                if snapshot_requested:
                    save_rgb(output / f"snapshot_{cycle_index:06d}.png", base.scene["capsule_camera"].data.output["rgb"])
                    print(f"PARAM_FORCE_10HZ_SNAPSHOT cycle={cycle_index}", flush=True)

                signature = (int(mode), float(alpha))
                if signature != active_signature:
                    print(
                        f"PARAM_FORCE_10HZ_CONTROL mode={mode.name} alpha={alpha:.1f} "
                        "boundary_latency_max_s=0.1",
                        flush=True,
                    )
                    active_signature = signature

                started = time.perf_counter()
                start_state = _boundary_state(base, term)
                start_sim_time_s = float(base.common_step_counter * base.step_dt)
                start_rgb_frame_id = _frame_id(base.scene["capsule_camera"])
                action = torch.tensor([[float(mode), float(alpha)]], device=base.device, dtype=torch.float32)
                result = env.step(action)
                last_observation = result[0]
                end_state = _boundary_state(base, term)
                end_sim_time_s = float(base.common_step_counter * base.step_dt)
                end_rgb_frame_id = _frame_id(base.scene["capsule_camera"])
                record = cycle_record(
                    term,
                    mode,
                    alpha,
                    cycle_index,
                    start_state,
                    end_state,
                    start_sim_time_s,
                    end_sim_time_s,
                    start_rgb_frame_id,
                    end_rgb_frame_id,
                    last_observation,
                )
                if record["physics_substeps"] != PHYSICS_STEPS_PER_CONTROL:
                    raise RuntimeError(f"Expected 24 physics substeps, got {record['physics_substeps']}")
                if not record["finite_state"]:
                    raise RuntimeError("Non-finite capsule state at the 10 Hz action boundary")
                with (output / "control_cycles.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                cycle_index += 1
                if args_cli.max_cycles and cycle_index >= args_cli.max_cycles:
                    exit_reason = "max_cycles"
                    break
                if args_cli.realtime:
                    remaining = 1.0 / CONTROL_HZ - (time.perf_counter() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)

            # Clear actor force before closing or handing control back.
            hold = torch.tensor(
                [[float(ParameterizedForceMode.HOLD), float(args_cli.initial_alpha)]],
                device=base.device,
                dtype=torch.float32,
            )
            env.step(hold)
            summary = {
                "schema": "parameterized_force_10hz_visual_session_v1_xyzw",
                "reason": exit_reason,
                "cycles": cycle_index,
                "physics_hz": PHYSICS_HZ,
                "control_hz": CONTROL_HZ,
                "physics_steps_per_control": PHYSICS_STEPS_PER_CONTROL,
                "render_fps": args_cli.render_fps,
                "output": str(output),
            }
            (output / "session_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"PARAM_FORCE_10HZ_COMPLETE {json.dumps(summary, sort_keys=True)}", flush=True)
            return 0
        finally:
            if keyboard is not None:
                keyboard.close()
            if camera_view is not None:
                camera_view.close()
            if pose_view is not None:
                pose_view.close()
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
