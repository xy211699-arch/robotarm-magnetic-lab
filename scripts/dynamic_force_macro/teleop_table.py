#!/usr/bin/env python3
"""TASK-008 桌面诊断：可视化执行单个 1 秒动态力宏动作并输出验收指标。"""

from __future__ import annotations

import argparse
from collections import deque
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


TASK_ID = "Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0"


def force_ratio(value: str) -> float:
    """Parse one positive, finite force-to-weight ratio within the TASK-008 limit."""
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 3.0:
        raise argparse.ArgumentTypeError("力度倍率必须是 (0, 3.0] 内的有限数")
    return parsed


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument(
    "--move_force_ratio", type=force_ratio, default=0.40,
    help="MOVE低档两端点合力相对于胶囊自重mg的倍率；两端各承担一半。",
)
parser.add_argument(
    "--move_force_ratio_medium", type=force_ratio, default=0.50,
    help="MOVE中档合力相对于胶囊自重mg的倍率。",
)
parser.add_argument(
    "--move_force_ratio_high", type=force_ratio, default=0.60,
    help="MOVE高档合力相对于胶囊自重mg的倍率。",
)
parser.add_argument(
    "--view_force_ratio", type=force_ratio, default=0.25,
    help="VIEW低档施加在相机侧端点的力相对于胶囊自重mg的倍率。",
)
parser.add_argument(
    "--view_force_ratio_medium", type=force_ratio, default=0.35,
    help="VIEW中档施加在相机侧端点的力相对于胶囊自重mg的倍率。",
)
parser.add_argument(
    "--view_force_ratio_high", type=force_ratio, default=0.45,
    help="VIEW高档施加在相机侧端点的力相对于胶囊自重mg的倍率。",
)
parser.add_argument(
    "--up_force_ratio", type=force_ratio, default=0.85,
    help="UP施加在实际相机端半球球心的世界向上力相对于胶囊自重mg的倍率。",
)
parser.add_argument("--output_directory", type=Path, default=Path("/tmp/task008-table-visual-inspection"))
parser.add_argument("--scripted_actions", default="", help="逗号分隔的动作名或 0..13。")
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

import robotarm_magnetic_lab.tasks  # noqa: F401
from common import evaluate_trace
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.runtime import SynchronousMacroRunner
from robotarm_magnetic_lab.teleop import CommandKind, DynamicForceMacroKeyboard
from robotarm_magnetic_lab.ui import attach_capsule_camera_policy_view, configure_capsule_camera_view
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force_macro import (
    DynamicForceMacroActionId,
    resolved_force_levels_n,
)


ACTION_NAMES = {int(action): action.name for action in DynamicForceMacroActionId}
NAME_TO_ACTION = {name: action for action, name in ACTION_NAMES.items()}


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
    def __init__(self, forces: dict[str, float]) -> None:
        self.window = omni.ui.Window("TASK-008 桌面动作诊断", width=520, height=170)
        with self.window.frame:
            with omni.ui.VStack(spacing=5):
                self.action = omni.ui.Label("动作：IDLE")
                self.phase = omni.ui.Label("阶段：等待按键")
                self.ratios = omni.ui.Label(
                    f"倍率：MOVE={forces['move_force_ratio']:g}/{forces['move_force_ratio_medium']:g}/"
                    f"{forces['move_force_ratio_high']:g}  VIEW={forces['view_force_ratio']:g}/"
                    f"{forces['view_force_ratio_medium']:g}/{forces['view_force_ratio_high']:g}  "
                    f"UP={forces['up_force_ratio']:g}"
                )
                self.newtons = omni.ui.Label(
                    f"合力：MOVE={forces['move_total_force_n']:.6f} N "
                    f"(每端{forces['move_force_per_endpoint_n']:.6f} N)  "
                    f"VIEW={forces['view_camera_endpoint_force_n']:.6f} N  "
                    f"UP={forces['up_camera_endpoint_force_n']:.6f} N"
                )
                self.result = omni.ui.Label("结果：尚未执行")

    def running(self, name: str) -> None:
        self.action.text = f"动作：{name}"
        self.phase.text = "阶段：运行 1.000 s / 240 个物理子步"
        self.result.text = "结果：计算中"

    def complete(self, passed: bool, metrics: dict) -> None:
        self.phase.text = "阶段：边界采样完成，外力已清除"
        self.result.text = f"结果：{'PASS' if passed else 'FAIL'}  {json.dumps(metrics, ensure_ascii=False)}"

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


def evaluate(action: int, trace) -> tuple[bool, dict]:
    if action == 0:
        force_norm = max((float(np.linalg.norm(item.applied_force_world)) for item in trace), default=float("inf"))
        torque_norm = max((float(np.linalg.norm(item.applied_torque_world)) for item in trace), default=float("inf"))
        passed = len(trace) == 240 and force_norm <= 1.0e-9 and torque_norm <= 1.0e-9
        return passed, {"substeps": len(trace), "max_force_n": force_norm, "max_torque_nm": torque_norm}
    return evaluate_trace(action, trace)


def main() -> int:
    scripted = parse_actions(args_cli.scripted_actions)
    output = args_cli.output_directory / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    cfg.actions.dynamic_force_macro.move_force_ratio = args_cli.move_force_ratio
    cfg.actions.dynamic_force_macro.move_force_ratio_medium = args_cli.move_force_ratio_medium
    cfg.actions.dynamic_force_macro.move_force_ratio_high = args_cli.move_force_ratio_high
    cfg.actions.dynamic_force_macro.view_force_ratio = args_cli.view_force_ratio
    cfg.actions.dynamic_force_macro.view_force_ratio_medium = args_cli.view_force_ratio_medium
    cfg.actions.dynamic_force_macro.view_force_ratio_high = args_cli.view_force_ratio_high
    cfg.actions.dynamic_force_macro.up_force_ratio = args_cli.up_force_ratio
    if not HEADLESS:
        configure_capsule_camera_view(cfg)

    env = keyboard = camera_view = panel = None
    records = []
    reason = "initialization_failed"
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            runner = SynchronousMacroRunner(env)
            term = env.unwrapped.action_manager.get_term("dynamic_force_macro")
            forces = resolved_force_levels_n(term.mass_kg, term.config)
            (output / "force_configuration.json").write_text(
                json.dumps(forces, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if not HEADLESS:
                keyboard = KitKeyboardSource()
                camera_view = attach_capsule_camera_policy_view(env)
                panel = StatusPanel(forces)
            print(f"TASK008_TABLE_FORCE_CONFIG {json.dumps(forces, sort_keys=True)}", flush=True)
            print(
                f"TASK008_TABLE_READY Space=HOLD MOVE[{args_cli.move_force_ratio:g}:D/A "
                f"{args_cli.move_force_ratio_medium:g}:L/J {args_cli.move_force_ratio_high:g}:O/U] "
                f"VIEW[{args_cli.view_force_ratio:g}:E/Q {args_cli.view_force_ratio_medium:g}:K/H "
                f"{args_cli.view_force_ratio_high:g}:I/Y] W=UP "
                "Backspace=重置 Esc=退出",
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
                    continue
                if command.kind is CommandKind.EXIT:
                    reason = "keyboard_exit"
                    break
                if command.kind is CommandKind.RESET:
                    env.reset()
                    print("TASK008_TABLE_RESET_COMPLETE", flush=True)
                    continue
                if command.kind is not CommandKind.ACTION:
                    continue

                action = int(command.action_id)
                name = ACTION_NAMES[action]
                if panel is not None:
                    panel.running(name)
                print(f"TASK008_TABLE_ACTION_START id={action} name={name}", flush=True)
                transition = runner.step(action)
                passed, metrics = evaluate(action, term.trace)
                action_count += 1
                rgb_path = save_boundary_rgb(output, action_count, transition.boundary_rgb)
                row = {
                    "index": action_count,
                    "action_id": action,
                    "action_name": name,
                    "pass": bool(passed),
                    "metrics": metrics,
                    "duration_s": transition.simulated_duration_s,
                    "trace_sha256": transition.trace_digest,
                    "boundary_rgb": str(rgb_path),
                    "force_configuration": forces,
                }
                records.append(row)
                with (output / "macro_actions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                if panel is not None:
                    panel.complete(passed, metrics)
                print(f"TASK008_TABLE_ACTION_COMPLETE {json.dumps(row, sort_keys=True)}", flush=True)
                if args_cli.max_actions and action_count >= args_cli.max_actions:
                    reason = "max_actions"
                    break

            summary = {
                "reason": reason,
                "actions": len(records),
                "force_configuration": forces,
                "output": str(output),
            }
            (output / "task008_table_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"TASK008_TABLE_COMPLETE {json.dumps(summary, sort_keys=True)}", flush=True)
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
