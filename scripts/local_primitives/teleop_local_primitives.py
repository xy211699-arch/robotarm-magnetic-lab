#!/usr/bin/env python3
"""连续运行 TASK-004 平面或胃部局部动力学原语。"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
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


FLAT_TASK_ID = "Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0"
STOMACH_TASK_ID = "Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0"
TASK_IDS = (FLAT_TASK_ID, STOMACH_TASK_ID)
DEFAULT_OUTPUT = ROOT / "logs" / "local_primitives_teleop"

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--task", choices=TASK_IDS, default=FLAT_TASK_ID, help="选择平面或胃部任务")
parser.add_argument("--num_envs", type=int, default=1, help="本任务固定为单环境")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--direction_azimuth_deg", type=float, default=0.0, help="世界 XY 平面方位角（度）")
parser.add_argument(
    "--scripted_sequence",
    default="",
    help='自动序列，例如 "0,1;reset;0,2;reset;0,2,3"',
)
parser.add_argument("--capsule_camera_view", action="store_true", help="显示胶囊相机实时窗口")
parser.add_argument("--max_steps", type=int, default=0, help="最大环境步数，0 表示不限制")
parser.add_argument("--output_directory", type=Path, default=DEFAULT_OUTPUT, help="遥测与快照目录")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    parser.error("局部动力学原语仅支持 --num_envs 1")
if args_cli.max_steps < 0:
    parser.error("--max_steps 不能为负数")
if not math.isfinite(args_cli.direction_azimuth_deg):
    parser.error("--direction_azimuth_deg 必须为有限值")
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
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (  # noqa: E402
    PrimitiveStatus,
)
from robotarm_magnetic_lab.teleop import (  # noqa: E402
    CommandKind,
    LocalPrimitiveKeyboard,
    parse_local_primitive_sequence,
)
from robotarm_magnetic_lab.ui import (  # noqa: E402
    attach_capsule_camera_policy_view,
    configure_capsule_camera_view,
)


TERMINAL_STATUSES = {
    PrimitiveStatus.SUCCEEDED_HOLDING,
    PrimitiveStatus.INVALID_START,
    PrimitiveStatus.TIMED_OUT,
    PrimitiveStatus.NONFINITE,
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _flat(value) -> np.ndarray:
    tensor = getattr(value, "torch", value)
    return tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _vector(value) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


class KitKeyboardSource:
    def __init__(self) -> None:
        self.commands = deque()
        self.keyboard = LocalPrimitiveKeyboard()
        self._input = carb.input.acquire_input_interface()
        self._device = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._device, self._on_event)

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


class SessionRecorder:
    def __init__(self, output_root: Path, task: str, profile_sha256: str) -> None:
        self.output = output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%fZ"
        )
        (self.output / "snapshots").mkdir(parents=True, exist_ok=False)
        self._stream = (self.output / "samples.jsonl").open("w", encoding="utf-8", buffering=1)
        self.task = task
        self.profile_sha256 = profile_sha256
        self.started_wall_s = time.monotonic()
        self.samples = 0
        self.resets = 0
        self.snapshots = 0
        self.outcomes: list[dict] = []

    def append(self, row: dict) -> None:
        self._stream.write(json.dumps(row, sort_keys=True) + "\n")
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

    def close(self, reason: str, sim_time_s: float) -> Path:
        self._stream.close()
        elapsed = time.monotonic() - self.started_wall_s
        summary = {
            "schema_version": "task004_local_primitives_teleop_v1",
            "task": self.task,
            "profile_sha256": self.profile_sha256,
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "reason": reason,
            "sim_time_s": float(sim_time_s),
            "wall_time_s": float(elapsed),
            "samples": self.samples,
            "resets": self.resets,
            "snapshots": self.snapshots,
            "outcomes": self.outcomes,
        }
        (self.output / "session.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.output


def _camera_rgb(env) -> np.ndarray | None:
    image = env.unwrapped.scene["capsule_camera"].data.output.get("rgb")
    if image is None:
        return None
    return image[0, ..., :3].detach().cpu().numpy()


def _sample(env, term, step: int, reset_index: int, requested_id: int | None) -> dict:
    base = env.unwrapped
    capsule = base.scene["capsule"]
    pose = _flat(capsule.data.root_link_pose_w)
    velocity = _flat(capsule.data.root_com_vel_w)
    telemetry = term.telemetry
    row = {
        "sim_time_s": float(base.episode_length_buf[0].item()) * float(base.step_dt),
        "step": int(step),
        "reset_index": int(reset_index),
        "requested_primitive_id": requested_id,
        "profile_sha256": term.profile_sha256,
        "position_world_m": pose[:3].tolist(),
        "quaternion_xyzw": pose[3:7].tolist(),
        "linear_velocity_world_m_s": velocity[:3].tolist(),
        "angular_velocity_world_rad_s": velocity[3:6].tolist(),
        "force_world_n": _flat(term.applied_force_world).tolist(),
        "torque_world_nm": _flat(term.applied_torque_world).tolist(),
        "last_request_result": term.last_request_result,
    }
    if telemetry is not None:
        row.update({
            "status": telemetry.status.value,
            "active_primitive": None if telemetry.active_primitive is None else int(telemetry.active_primitive),
            "elapsed_primitive_s": float(telemetry.elapsed_s),
            "completion_time_s": telemetry.completion_time_s,
            "actual_axis_world": _vector(telemetry.actual_axis_world),
            "desired_axis_world": _vector(telemetry.desired_axis_world),
            "pose_torque_world_nm": _vector(telemetry.pose_torque_world_nm),
            "endpoint_force_world_n": _vector(telemetry.endpoint_force_world_n),
            "endpoint_equivalent_torque_world_nm": _vector(
                telemetry.endpoint_equivalent_torque_world_nm
            ),
            "total_force_world_n": _vector(telemetry.total_force_world_n),
            "total_torque_world_nm": _vector(telemetry.total_torque_world_nm),
            "force_saturated": bool(telemetry.force_saturated),
            "torque_saturated": bool(telemetry.torque_saturated),
            "force_slew_limited": bool(telemetry.force_slew_limited),
            "torque_slew_limited": bool(telemetry.torque_slew_limited),
        })
    return row


def main() -> int:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    try:
        scripted = deque(parse_local_primitive_sequence(args_cli.scripted_sequence))
    except ValueError as exc:
        parser.error(str(exc))
    direction_rad = math.radians(args_cli.direction_azimuth_deg)
    direction_xy = np.asarray([math.cos(direction_rad), math.sin(direction_rad)], dtype=np.float32)
    env_cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = "cpu"
    if args_cli.capsule_camera_view:
        configure_capsule_camera_view(env_cfg)

    env = keyboard = view_handle = recorder = None
    reason = "initialization_failed"
    exit_code = 1
    sim_time_s = 0.0
    with launch_simulation(env_cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
            env.reset()
            term = env.unwrapped.action_manager.get_term("local_primitive")
            recorder = SessionRecorder(args_cli.output_directory, args_cli.task, term.profile_sha256)
            keyboard = None if HEADLESS else KitKeyboardSource()
            if args_cli.capsule_camera_view:
                view_handle = attach_capsule_camera_policy_view(env)
            print(f"LOCAL_PRIMITIVES_OUTPUT_DIRECTORY {recorder.output}", flush=True)
            print(f"LOCAL_PRIMITIVES_PROFILE_SHA256 {term.profile_sha256}", flush=True)
            print(
                "LOCAL_PRIMITIVES_READY 1=侧躺到直立 2=直立到侧躺 "
                "3=直立到30度 4=30度圆锥一周；Backspace=复位 F12=快照 Esc=退出",
                flush=True,
            )
            reset_index = 0
            step = 0
            requested_id: int | None = None
            pulse_id: int | None = None
            awaiting_terminal = False
            scripted_mode = bool(args_cli.scripted_sequence)
            exit_requested = False
            while simulation_app.is_running() and not exit_requested:
                if keyboard is not None:
                    while keyboard.commands:
                        command = keyboard.commands.popleft()
                        if command.kind is CommandKind.EXIT:
                            exit_requested = True
                        elif command.kind is CommandKind.RESET:
                            env.reset()
                            keyboard.keyboard.release_all()
                            recorder.resets += 1
                            reset_index += 1
                            requested_id = pulse_id = None
                            awaiting_terminal = False
                            print("LOCAL_PRIMITIVE_RESET", flush=True)
                        elif command.kind is CommandKind.SNAPSHOT:
                            row = _sample(env, term, step, reset_index, requested_id)
                            recorder.snapshot(row, _camera_rgb(env), "f12")
                        elif command.kind is CommandKind.ACTION and not scripted_mode:
                            pulse_id = int(command.action_id)
                            requested_id = pulse_id
                if exit_requested:
                    reason = "user_exit"
                    break

                if scripted_mode and pulse_id is None and not awaiting_terminal:
                    if scripted:
                        event = scripted.popleft()
                        if event is None:
                            env.reset()
                            recorder.resets += 1
                            reset_index += 1
                            requested_id = None
                            print("LOCAL_PRIMITIVE_SCRIPT_RESET", flush=True)
                            continue
                        pulse_id = int(event)
                        requested_id = pulse_id
                    else:
                        reason = "script_complete"
                        break

                action_np = np.zeros(4, dtype=np.float32)
                sent_pulse = pulse_id is not None
                if sent_pulse:
                    action_np[:] = [1.0, float(pulse_id), direction_xy[0], direction_xy[1]]
                    awaiting_terminal = True
                action = torch.as_tensor(action_np, device=env.unwrapped.device).reshape(1, 4)
                _, _, terminated, truncated, _ = env.step(action)
                step += 1
                row = _sample(env, term, step, reset_index, requested_id)
                sim_time_s = row["sim_time_s"]
                recorder.append(row)
                if sent_pulse:
                    print(
                        f"LOCAL_PRIMITIVE_REQUEST id={pulse_id} result={term.last_request_result} "
                        f"azimuth_deg={args_cli.direction_azimuth_deg:.3f}",
                        flush=True,
                    )
                    pulse_id = None
                telemetry = term.telemetry
                if awaiting_terminal and telemetry is not None and telemetry.status in TERMINAL_STATUSES:
                    outcome = {
                        "primitive_id": requested_id,
                        "status": telemetry.status.value,
                        "completion_time_s": telemetry.completion_time_s,
                        "sim_time_s": sim_time_s,
                    }
                    recorder.outcomes.append(outcome)
                    print(f"LOCAL_PRIMITIVE_OUTCOME {json.dumps(outcome, sort_keys=True)}", flush=True)
                    recorder.snapshot(row, _camera_rgb(env), telemetry.status.value)
                    awaiting_terminal = False
                    requested_id = None
                if step % 30 == 0:
                    print(
                        f"LOCAL_PRIMITIVE_STATE t={sim_time_s:.3f}s status={row.get('status', 'none')} "
                        f"pos={np.round(row['position_world_m'], 5).tolist()} "
                        f"force={np.round(row['total_force_world_n'], 5).tolist() if 'total_force_world_n' in row else []} "
                        f"torque={np.round(row['total_torque_world_nm'], 6).tolist() if 'total_torque_world_nm' in row else []}",
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
    if recorder is not None:
        output = recorder.close(reason, sim_time_s)
        print(f"LOCAL_PRIMITIVES_SESSION {output / 'session.json'}", flush=True)
    simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
