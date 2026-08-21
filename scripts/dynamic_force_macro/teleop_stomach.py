#!/usr/bin/env python3
"""TASK-008 胃部场景：一次按键只执行一个同步 1 秒动态力动作。"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
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


TASK_ID = "Template-Robotarm-Magnetic-Dynamic-Force-Macro-Stomach-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--profile", type=Path, default=Path("/tmp/task008-dynamic-force-calibration/selected_profile.json"))
parser.add_argument("--output_directory", type=Path, default=Path("/tmp/task008-stomach-inspection"))
parser.add_argument("--scripted_actions", default="", help="逗号分隔的动作名或 0..5；用于无键盘启动检查。")
parser.add_argument("--max_actions", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.task != TASK_ID:
    parser.error(f"本启动器仅接受 {TASK_ID}")

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import carb
import gymnasium as gym
import numpy as np
import omni.appwindow
import omni.ui
import torch

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.coverage.simulator_runtime import P0CoverageRuntime
from robotarm_magnetic_lab.runtime import SynchronousMacroRunner
from robotarm_magnetic_lab.teleop import CommandKind, DynamicForceMacroKeyboard
from robotarm_magnetic_lab.ui import attach_capsule_camera_policy_view, configure_capsule_camera_view


ACTION_NAMES = {0: "HOLD", 1: "MOVE_POS", 2: "MOVE_NEG", 3: "VIEW_POS", 4: "VIEW_NEG", 5: "UP"}
NAME_TO_ACTION = {name: action for action, name in ACTION_NAMES.items()}


def load_profile(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    required = ("move_force_ratio", "view_force_ratio", "up_force_ratio")
    if payload.get("schema") != "task008_force_profile_v1" or any(payload.get(key) is None for key in required):
        raise ValueError(f"无效 TASK-008 标定配置：{path}")
    return payload, hashlib.sha256(path.expanduser().read_bytes()).hexdigest()


def parse_actions(value: str) -> deque[int]:
    result: deque[int] = deque()
    for item in filter(None, (part.strip().upper() for part in value.split(","))):
        action = int(item) if item.isdigit() else NAME_TO_ACTION.get(item, -1)
        if action not in ACTION_NAMES:
            raise ValueError(f"未知动作 {item!r}")
        result.append(action)
    return result


class KitKeyboardSource:
    def __init__(self) -> None:
        self.commands = deque()
        self.keyboard = DynamicForceMacroKeyboard()
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


class StatusPanel:
    def __init__(self) -> None:
        self.window = omni.ui.Window("TASK-008 动态力动作状态", width=430, height=150)
        with self.window.frame:
            with omni.ui.VStack(spacing=5):
                self.action = omni.ui.Label("动作：IDLE")
                self.phase = omni.ui.Label("阶段：PAUSED")
                self.time = omni.ui.Label("仿真时间：0.000 s")
                self.coverage = omni.ui.Label("累计覆盖率：0.000%")

    def update(self, action: str, phase: str, sim_time: float, fraction: float) -> None:
        self.action.text = f"动作：{action}"
        self.phase.text = f"阶段：{phase}"
        self.time.text = f"仿真时间：{sim_time:.3f} s"
        self.coverage.text = f"累计覆盖率：{100.0 * fraction:.3f}%"

    def close(self) -> None:
        self.window.visible = False


def save_boundary_rgb(output: Path, index: int, rgb) -> Path:
    from PIL import Image
    array = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    array = np.asarray(array)[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * (255.0 if array.max(initial=0.0) <= 1.0 else 1.0), 0, 255)
    path = output / f"boundary_rgb_{index:04d}.png"
    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


def main() -> int:
    profile, profile_sha = load_profile(args_cli.profile)
    scripted = parse_actions(args_cli.scripted_actions)
    output = args_cli.output_directory / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    cfg.actions.dynamic_force_macro.move_force_ratio = float(profile["move_force_ratio"])
    cfg.actions.dynamic_force_macro.view_force_ratio = float(profile["view_force_ratio"])
    cfg.actions.dynamic_force_macro.up_force_ratio = float(profile["up_force_ratio"])
    if not HEADLESS:
        configure_capsule_camera_view(cfg)
    env = keyboard = camera_view = panel = coverage = None
    records = []
    reason = "initialization_failed"
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            coverage = P0CoverageRuntime(
                env,
                output,
                task_id=args_cli.task,
                seed=0,
                commit="TASK-008",
                branch="feature/TASK-008-six-action-dynamic-force-controller",
                enable_view=not HEADLESS,
                require_camera_facing_normal=True,
                raycast_device="cuda:0",
            )
            runner = SynchronousMacroRunner(env, coverage)
            if not HEADLESS:
                keyboard = KitKeyboardSource()
                camera_view = attach_capsule_camera_policy_view(env)
                panel = StatusPanel()
            print(
                "TASK008_STOMACH_READY Space=HOLD D/A=MOVE+/- E/Q=VIEW+/- W=UP "
                "Backspace=重置 F12=快照 Esc=退出",
                flush=True,
            )
            action_count = 0
            reason = "exit"
            while simulation_app.is_running():
                command = None
                if scripted:
                    command = type("Command", (), {"kind": CommandKind.ACTION, "action_id": scripted.popleft()})()
                elif keyboard is not None and keyboard.commands:
                    command = keyboard.commands.popleft()
                if command is None:
                    if HEADLESS:
                        break
                    simulation_app.update()
                    if panel is not None:
                        fraction = float(coverage.accumulator.mask.mean())
                        panel.update("IDLE", "PAUSED", coverage.total_sim_time_s, fraction)
                    continue
                if command.kind is CommandKind.EXIT:
                    reason = "keyboard_exit"
                    break
                if command.kind is CommandKind.RESET:
                    coverage.reset()
                    env.reset()
                    print("TASK008_RESET_COMPLETE", flush=True)
                    continue
                if command.kind is CommandKind.SNAPSHOT:
                    metadata = coverage.snapshot("manual")
                    np.save(coverage.partial_directory / "coverage_mask_manual.npy", coverage.accumulator.mask)
                    print(f"TASK008_SNAPSHOT {json.dumps(metadata, sort_keys=True)}", flush=True)
                    continue
                action = int(command.action_id)
                print(f"TASK008_ACTION_START id={action} name={ACTION_NAMES[action]}", flush=True)
                if panel is not None:
                    panel.update(ACTION_NAMES[action], "RUNNING", coverage.total_sim_time_s, float(coverage.accumulator.mask.mean()))
                transition = runner.step(action)
                action_count += 1
                rgb_path = save_boundary_rgb(coverage.partial_directory, action_count, transition.boundary_rgb)
                row = {
                    "index": action_count,
                    "action_id": action,
                    "action_name": ACTION_NAMES[action],
                    "start_frame": transition.start_rgb_frame_id,
                    "boundary_frame": transition.boundary_rgb_frame_id,
                    "duration_s": transition.simulated_duration_s,
                    "trace_sha256": transition.trace_digest,
                    "boundary_rgb": str(rgb_path),
                    "coverage_fraction": float(coverage.accumulator.mask.mean()),
                }
                records.append(row)
                with (coverage.partial_directory / "macro_actions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                coverage.update_view()
                print(f"TASK008_ACTION_COMPLETE {json.dumps(row, sort_keys=True)}", flush=True)
                if args_cli.max_actions and action_count >= args_cli.max_actions:
                    reason = "max_actions"
                    break
            summary = {"reason": reason, "profile": str(args_cli.profile), "profile_sha256": profile_sha, "actions": len(records), "output": str(output)}
            (coverage.partial_directory / "task008_stomach_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            final_dir = coverage.finalize(reason)
            print(f"TASK008_STOMACH_COMPLETE {json.dumps(summary, sort_keys=True)}", flush=True)
            return 0
        finally:
            if keyboard is not None:
                keyboard.close()
            if camera_view is not None:
                camera_view.close()
            if panel is not None:
                panel.close()
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
