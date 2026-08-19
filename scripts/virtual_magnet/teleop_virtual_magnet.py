#!/usr/bin/env python3
"""TASK-007 单键单动作可视化：0..9 与减号映射 11 个动作。"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

from common import ACTION_NAMES, key_name_to_action


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", choices=("flat", "stomach"), default="flat")
parser.add_argument("--render_fps", type=int, choices=(60, 120, 240), default=120)
parser.add_argument("--capsule_camera_view", action="store_true")
parser.add_argument("--max_steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
if getattr(args_cli, "headless", False):
    parser.error("键盘可视化不能使用 headless")
args_cli.enable_cameras = True
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import carb
import gymnasium as gym
import omni.appwindow
import torch

import robotarm_magnetic_lab.tasks  # noqa: F401,E402
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.ui import attach_capsule_camera_policy_view, configure_capsule_camera_view


TASKS = {
    "flat": "Template-Robotarm-Magnetic-Virtual-Magnet-Flat-Lab-v0",
    "stomach": "Template-Robotarm-Magnetic-Virtual-Magnet-Stomach-Lab-v0",
}


class Keyboard:
    def __init__(self):
        self.queue = deque()
        self.down = set()
        self.executing = False
        self.input = carb.input.acquire_input_interface()
        self.device = omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription = self.input.subscribe_to_keyboard_events(self.device, self._event)

    def _event(self, event, *_args):
        key = str(event.input.name).upper()
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.down.discard(key)
            return True
        if event.type != carb.input.KeyboardEventType.KEY_PRESS or key in self.down:
            return True
        self.down.add(key)
        if key in ("ESCAPE", "KEY_ESCAPE"):
            self.queue.append("exit")
        elif key in ("R", "KEY_R") and not self.executing:
            self.queue.append("reset")
        elif not self.executing:
            action_id = key_name_to_action(key)
            if action_id is not None:
                self.queue.append(action_id)
        return True

    def close(self):
        self.input.unsubscribe_to_keyboard_events(self.device, self.subscription)


def main() -> None:
    task = TASKS[args_cli.scene]
    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1, use_fabric=True)
    cfg.sim.render_interval = 240 // args_cli.render_fps
    if args_cli.capsule_camera_view:
        configure_capsule_camera_view(cfg)
    with launch_simulation(cfg, args_cli):
        env = gym.make(task, cfg=cfg)
        env.reset()
        camera_view = attach_capsule_camera_policy_view(env) if args_cli.capsule_camera_view else None
        keyboard = Keyboard()
        print(
            "VIRTUAL_MAGNET_TELEOP_READY keys=0..9,- actions="
            + json.dumps(dict(enumerate(ACTION_NAMES)), ensure_ascii=False)
            + " reset=R exit=Esc",
            flush=True,
        )
        steps = 0
        last_time = time.perf_counter()
        try:
            while simulation_app.is_running():
                if keyboard.queue:
                    request = keyboard.queue.popleft()
                    if request == "exit":
                        break
                    if request == "reset":
                        env.reset()
                        print("VIRTUAL_MAGNET_RESET", flush=True)
                        continue
                    keyboard.executing = True
                    print(f"VIRTUAL_MAGNET_REQUEST id={request} name={ACTION_NAMES[request]}", flush=True)
                    env.step(torch.tensor([[float(request)]], device=env.unwrapped.device))
                    audit = env.unwrapped._virtual_magnet_bridge.audit
                    elapsed = max(time.perf_counter() - last_time, 1.0e-9)
                    last_time = time.perf_counter()
                    terminal = {
                        key: (value.tolist() if hasattr(value, "tolist") else value)
                        for key, value in audit.items()
                        if key not in ("desired_wrench", "model_raw_wrench")
                    }
                    terminal["measured_environment_steps_hz"] = 1.0 / elapsed
                    print("VIRTUAL_MAGNET_RESULT " + json.dumps(terminal, sort_keys=True), flush=True)
                    keyboard.executing = False
                else:
                    # Internal -1 advances physics/rendering without accepting
                    # or queueing a public action.
                    env.step(torch.tensor([[-1.0]], device=env.unwrapped.device))
                steps += 1
                if args_cli.max_steps and steps >= args_cli.max_steps:
                    break
        finally:
            keyboard.close()
            if camera_view is not None:
                camera_view.close()
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

