#!/usr/bin/env python3
"""连续运行 TASK-005 十一个动态动作；终端仅输出事件。"""

from __future__ import annotations

import argparse
from collections import deque
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

from isaaclab.app import AppLauncher  # noqa: E402


TASK_ID = "Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--render_fps", type=int, choices=(60, 120, 240), default=120)
parser.add_argument("--capsule_camera_view", action="store_true")
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--output_directory", type=Path, default=ROOT / "logs/eleven_action_teleop")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    parser.error("TASK-005 仅支持 --num_envs 1")
args_cli.enable_cameras = True
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import torch  # noqa: E402

import robotarm_magnetic_lab.tasks  # noqa: F401, E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotarm_magnetic_lab.teleop import CommandKind, ElevenActionKeyboard  # noqa: E402
from robotarm_magnetic_lab.ui import attach_capsule_camera_policy_view, configure_capsule_camera_view  # noqa: E402


class KitKeyboardSource:
    def __init__(self) -> None:
        self.commands = deque()
        self.keyboard = ElevenActionKeyboard()
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


def _camera_rgb(env):
    image = env.unwrapped.scene["capsule_camera"].data.output.get("rgb")
    return None if image is None else image[0, ..., :3].detach().cpu().numpy()


def _snapshot(env, output: Path, index: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    capsule = env.unwrapped.scene["capsule"]
    pose = capsule.data.root_link_pose_w.torch[0].detach().cpu().numpy().tolist()
    path = output / f"snapshot_{index:04d}.json"
    path.write_text(json.dumps({"root_link_pose_xyzw": pose}, indent=2) + "\n", encoding="utf-8")
    image = _camera_rgb(env)
    if image is not None:
        from PIL import Image

        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path.with_suffix(".png"))
    return path


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.render_interval = 240 // args_cli.render_fps
    if args_cli.capsule_camera_view:
        configure_capsule_camera_view(env_cfg)
    output = args_cli.output_directory / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    events = (output / "events.jsonl").open("w", encoding="utf-8", buffering=1)
    env = keyboard = camera_view = None
    exit_reason = "initialization_failed"
    exit_code = 1
    with launch_simulation(env_cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
            env.reset(seed=args_cli.seed)
            term = env.unwrapped.action_manager.get_term("eleven_action")
            keyboard = None if HEADLESS else KitKeyboardSource()
            if args_cli.capsule_camera_view:
                camera_view = attach_capsule_camera_policy_view(env)
            print(f"ELEVEN_ACTION_READY profile={term.profile_sha256} device={env.unwrapped.device}", flush=True)
            step = snapshot_index = 0
            pending_id = None
            last_request_id = 0
            exit_requested = False
            while simulation_app.is_running() and not exit_requested:
                if keyboard is not None:
                    while keyboard.commands:
                        command = keyboard.commands.popleft()
                        if command.kind is CommandKind.EXIT:
                            exit_requested = True
                        elif command.kind is CommandKind.RESET:
                            env.reset(seed=args_cli.seed)
                            pending_id = None
                            keyboard.keyboard.release_all()
                            print("ELEVEN_ACTION_RESET", flush=True)
                        elif command.kind is CommandKind.SNAPSHOT:
                            snapshot_index += 1
                            path = _snapshot(env, output, snapshot_index)
                            print(f"ELEVEN_ACTION_SNAPSHOT path={path}", flush=True)
                        elif command.kind is CommandKind.ACTION:
                            pending_id = int(command.action_id)
                if exit_requested:
                    exit_reason = "user_exit"
                    break
                request = -1.0 if pending_id is None else float(pending_id)
                action = torch.tensor([[request]], device=env.unwrapped.device, dtype=torch.float32)
                _, _, terminated, truncated, _ = env.step(action)
                step += 1
                if pending_id is not None:
                    print(f"ELEVEN_ACTION_REQUEST id={pending_id} ready={term.ready}", flush=True)
                    pending_id = None
                telemetry = term.telemetry
                if telemetry is not None and telemetry.request_id != last_request_id:
                    last_request_id = telemetry.request_id
                if telemetry is not None and telemetry.result is not None:
                    row = {
                        "request_id": telemetry.request_id,
                        "action_id": int(telemetry.action_id),
                        "result": telemetry.result.value,
                        "substeps": telemetry.substep_index,
                        "constrained": telemetry.constrained,
                        "start_axis_world": telemetry.start_axis_world.tolist(),
                        "end_axis_world": telemetry.end_axis_world.tolist(),
                        "support_drift_m": telemetry.support_drift_m,
                        "move_signed_displacement_m": telemetry.move_signed_displacement_m,
                        "camera_contact": telemetry.camera_contact,
                        "sidewall_contact": telemetry.sidewall_contact,
                        "profile_sha256": telemetry.profile_sha256,
                    }
                    events.write(json.dumps(row, sort_keys=True) + "\n")
                    print(f"ELEVEN_ACTION_RESULT {json.dumps(row, sort_keys=True)}", flush=True)
                    print("ELEVEN_ACTION_READY", flush=True)
                if telemetry is not None and telemetry.result is not None and telemetry.result.value == "fault":
                    print(f"ELEVEN_ACTION_FAULT message={telemetry.message}", flush=True)
                    exit_reason, exit_code = "controller_fault", 2
                    break
                if bool(terminated[0] or truncated[0]):
                    exit_reason, exit_code = "environment_termination", 2
                    break
                if args_cli.max_steps and step >= args_cli.max_steps:
                    exit_reason = "max_steps"
                    break
            if exit_code != 2:
                exit_code = 0
        finally:
            if camera_view is not None:
                camera_view.close()
            if keyboard is not None:
                keyboard.close()
            if env is not None:
                env.close()
    events.close()
    session = output / "session.json"
    session.write_text(
        json.dumps({"reason": exit_reason, "steps": step, "render_fps": args_cli.render_fps}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ELEVEN_ACTION_SESSION path={session}", flush=True)
    simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
