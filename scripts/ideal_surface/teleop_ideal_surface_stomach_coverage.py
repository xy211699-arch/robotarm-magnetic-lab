#!/usr/bin/env python3
"""Teleoperate the fifteen-action ideal-surface task with P0 coverage evidence."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))


def _default_output_directory() -> Path:
    """Resolve logs to the primary clone even when launched from a linked worktree."""
    try:
        common_git = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        common_git_path = Path(common_git).resolve()
        if common_git_path.name == ".git":
            return common_git_path.parent / "logs" / "ideal_surface_coverage_teleop"
    except (OSError, subprocess.CalledProcessError):
        pass
    return ROOT / "logs" / "ideal_surface_coverage_teleop"

HEADLESS = "--headless" in sys.argv
if HEADLESS:
    sys.argv.remove("--headless")
    os.environ["HEADLESS"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output_directory",
    type=Path,
    default=_default_output_directory(),
)
parser.add_argument("--scripted_actions", default="")
parser.add_argument("--max_idle_updates", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    parser.error("ideal-surface stomach teleoperation requires --num_envs 1")
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
from robotarm_magnetic_lab.coverage.simulator_runtime import P0CoverageRuntime  # noqa: E402
from robotarm_magnetic_lab.teleop import (  # noqa: E402
    CommandKind,
    IdealSurfaceKeyboard,
    KeyCommand,
    RequestOutcome,
    SessionController,
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


class KitKeyboardSource:
    def __init__(self) -> None:
        self.commands = deque()
        self.keyboard = IdealSurfaceKeyboard()
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


def _scripted(action_id: int) -> KeyCommand:
    return KeyCommand(CommandKind.ACTION, f"SCRIPTED_{action_id}", int(action_id))


def main() -> int:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    # The controller already latches its last safe pose on HARD_FAILURE.  Keep
    # the environment alive for one diagnostic read so the terminal action
    # result and final monotonic camera frame can be recorded before exit.
    # Manager-driven auto-reset would reset the camera frame counter inside
    # env.step(), before the coverage evaluator can observe termination.
    env_cfg.terminations.ideal_surface_hard_failure = None
    output = args_cli.output_directory / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%fZ"
    )
    print(f"IDEAL_SURFACE_OUTPUT_DIRECTORY {output.resolve()}", flush=True)
    env = evaluator = keyboard = None
    exit_code = 1
    with launch_simulation(env_cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
            env.reset()
            term = env.unwrapped.action_manager.get_term("ideal_surface")
            session = SessionController()
            evaluator = P0CoverageRuntime(
                env,
                output,
                task_id=args_cli.task,
                seed=args_cli.seed,
                commit=_git("rev-parse", "HEAD"),
                branch=_git("branch", "--show-current"),
                enable_view=not HEADLESS and not bool(args_cli.scripted_actions),
            )
            scripted = deque(
                int(item.strip())
                for item in args_cli.scripted_actions.split(",")
                if item.strip()
            )
            if not args_cli.scripted_actions:
                keyboard = KitKeyboardSource()
            print(
                "IDEAL_SURFACE_READY R/T/Y/F/H/V/B/N or Numpad 7/8/9/4/6/1/2/3=start tilt; "
                "W/S=tilt/rise; D/A=precess +/-; E/Q=roll +/-; Space=hold; "
                "Backspace=reset; F12=snapshot; Esc=exit",
                flush=True,
            )
            idle_updates = 0
            exit_requested = False
            display_last = 0.0
            while simulation_app.is_running() and not exit_requested:
                now_wall = time.monotonic()
                if evaluator.view is not None and now_wall - display_last >= 1.0 / 30.0:
                    evaluator.update_view()
                    display_last = now_wall
                commands = []
                if keyboard is not None:
                    while keyboard.commands:
                        commands.append(keyboard.commands.popleft())
                if not session.busy and scripted:
                    commands.append(_scripted(scripted.popleft()))
                if not commands:
                    if args_cli.scripted_actions and not scripted and not session.busy:
                        break
                    simulation_app.update()
                    idle_updates += 1
                    if args_cli.max_idle_updates and idle_updates >= args_cli.max_idle_updates:
                        break
                    continue
                idle_updates = 0
                for command in commands:
                    now = evaluator.sim_time_s
                    if command.kind is CommandKind.EXIT:
                        exit_requested = True
                        continue
                    if command.kind is CommandKind.SNAPSHOT:
                        evaluator.snapshot("f12")
                        continue
                    if command.kind is CommandKind.RESET:
                        record = session.request_reset(now)
                        evaluator.append_action_event(record, "request")
                        if record.outcome is RequestOutcome.RESET_ACCEPTED:
                            evaluator.reset()
                            env.reset()
                            # Force the exactly separated reset snapshot into
                            # the kinematic capsule before the next command.
                            term.action_mask()
                        continue
                    record = session.request_action(command.action_id, term.action_mask(), now)
                    evaluator.append_action_event(record, "request")
                    print(
                        f"IDEAL_REQUEST id={record.request_id} action={command.action_id} "
                        f"outcome={record.outcome.value}",
                        flush=True,
                    )
                    if record.outcome is not RequestOutcome.ACCEPTED:
                        continue
                    action = torch.full(
                        env.action_space.shape,
                        int(command.action_id),
                        device=env.unwrapped.device,
                        dtype=torch.float32,
                    )
                    _, _, terminated, truncated, _ = env.step(action)
                    if bool(terminated[0] or truncated[0]):
                        active_terminations = [
                            name
                            for name, values in env.unwrapped.termination_manager.get_active_iterable_terms(0)
                            if bool(values[0])
                        ]
                        evaluator.append_action_event(
                            record,
                            "environment_termination",
                            termination_terms=active_terminations or ["unknown"],
                        )
                        print(
                            "IDEAL_SURFACE_ERROR environment terminated terms="
                            f"{active_terminations or ['unknown']}",
                            flush=True,
                        )
                        evaluator.snapshot("environment_termination")
                        exit_code = 2
                        exit_requested = True
                        break
                    evaluator.maybe_update()
                    result = term.last_result
                    if result is None:
                        raise RuntimeError("ideal action reached its boundary without one result")
                    completion = session.acknowledge(result.status.value, evaluator.sim_time_s)
                    assert completion is not None
                    evaluator.append_action_event(
                        completion,
                        "result",
                        ideal_surface_result=result.to_dict(),
                        schema_version="ideal_surface_v2",
                    )
                    print(
                        f"IDEAL_RESULT request={completion.request_id} action={completion.action_id} "
                        f"status={completion.device_result}",
                        flush=True,
                    )
                    if result.status.value == "HARD_FAILURE":
                        evaluator.snapshot("hard_failure")
                        exit_code = 2
                        exit_requested = True
                        break
                    term.acknowledge_result()
            if exit_code != 2:
                exit_code = 0
        finally:
            if keyboard is not None:
                keyboard.close()
            if evaluator is not None:
                final = evaluator.finalize("exit")
                print(f"IDEAL_SURFACE_OUTPUT {final}", flush=True)
            if env is not None:
                env.close()
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
