#!/usr/bin/env python3
"""Run boundary-safe atomic stomach teleoperation with privileged P0 coverage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))

# Isaac Lab 3.0's current AppLauncher reserves but does not expose --headless.
# Preserve the public CLI required by the contract through its documented env flag.
HEADLESS = "--headless" in sys.argv
if HEADLESS:
    sys.argv.remove("--headless")
    os.environ["HEADLESS"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task", default="Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0"
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output_directory",
    type=Path,
    default=Path("/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop"),
)
parser.add_argument(
    "--scripted_actions",
    default="",
    help="Comma-separated action IDs. Empty uses the interactive Kit keyboard.",
)
parser.add_argument("--minimum_coverage_samples", type=int, default=0)
parser.add_argument("--max_steps_per_action", type=int, default=240)
parser.add_argument("--max_idle_updates", type=int, default=0)
parser.add_argument(
    "--target_coverage_percent",
    type=float,
    default=0.0,
    help="Automatically submit boundary-safe actions until this coverage percentage is reached.",
)
parser.add_argument(
    "--max_action_calls",
    type=int,
    default=200,
    help="Safety budget for --target_coverage_percent mode.",
)
parser.add_argument(
    "--max_run_wall_time_s",
    type=float,
    default=600.0,
    help="Wall-clock campaign budget for target mode; checked only at action boundaries.",
)
parser.add_argument(
    "--auto_action_cycle",
    default="3,3,1,7,4,4,2,8,5,6,9,10",
    help="Comma-separated action priority cycle used by automatic coverage validation.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    parser.error("P0 stomach teleoperation supports exactly --num_envs 1")
if args_cli.scripted_actions and args_cli.target_coverage_percent > 0.0:
    parser.error("--scripted_actions and --target_coverage_percent are mutually exclusive")
if not 0.0 <= args_cli.target_coverage_percent <= 100.0:
    parser.error("--target_coverage_percent must be in [0, 100]")
if args_cli.max_action_calls <= 0:
    parser.error("--max_action_calls must be positive")
if args_cli.max_run_wall_time_s <= 0.0:
    parser.error("--max_run_wall_time_s must be positive")
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
    AtomicKeyboard,
    CommandKind,
    RequestOutcome,
    SessionController,
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class KitKeyboardSource:
    """Translate Kit events through the same pure AtomicKeyboard as tests."""

    def __init__(self) -> None:
        self.commands = deque()
        self.keyboard = AtomicKeyboard()
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
        else:  # KEY_REPEAT, CHAR, and unknown events never submit a request.
            command = None
        if command is not None:
            self.commands.append(command)
        return True

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._device, self._subscription)
            self._subscription = None


def _handle_command(command, session, term, evaluator, env) -> tuple[float | None, bool]:
    """Apply one boundary command; return accepted action ID and exit flag."""
    now = evaluator.sim_time_s
    if command.kind is CommandKind.ACTION:
        record = session.request_action(command.action_id, term.action_mask(), now)
        evaluator.append_action_event(record, "request")
        print(
            f"P0_REQUEST id={record.request_id} action={command.action_id} outcome={record.outcome.value}",
            flush=True,
        )
        return (
            float(command.action_id) if record.outcome is RequestOutcome.ACCEPTED else None,
            False,
        )
    if command.kind is CommandKind.RESET:
        record = session.request_reset(now)
        evaluator.append_action_event(record, "request")
        print(f"P0_RESET id={record.request_id} outcome={record.outcome.value}", flush=True)
        if record.outcome is RequestOutcome.RESET_ACCEPTED:
            evaluator.reset()
            env.reset()
        return None, False
    if command.kind is CommandKind.SNAPSHOT:
        evaluator.snapshot("f12")
        print("P0_SNAPSHOT saved", flush=True)
        return None, False
    return None, command.kind is CommandKind.EXIT


def _scripted_command(action_id: int):
    from robotarm_magnetic_lab.teleop.atomic_keyboard import KeyCommand

    return KeyCommand(CommandKind.ACTION, f"SCRIPTED_{action_id}", int(action_id))


def main() -> int:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    env_cfg.seed = args_cli.seed
    if args_cli.target_coverage_percent > 0.0:
        # The executor itself latches a safe target on HARD_FAILURE. Disable
        # manager auto-reset for this validation mode only, so the outer loop
        # can persist the exact failure code before exiting. Physical collision
        # and timeout termination terms remain active.
        env_cfg.terminations.atomic_hard_failure = None
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output_directory = args_cli.output_directory / session_id
    keyboard_source = None
    evaluator = None
    env = None
    exit_code = 1
    run_started_wall = time.monotonic()
    termination_reason = "initialization_failed"
    action_attempts = 0
    action_calls = 0
    action_successes = 0
    action_failures = 0
    action_rejections = 0
    action_histogram: Counter[int] = Counter()
    failure_codes: Counter[str] = Counter()
    with launch_simulation(env_cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
            env.reset()
            term = env.unwrapped.action_manager.get_term("atomic")
            session = SessionController()
            evaluator = P0CoverageRuntime(
                env,
                output_directory,
                task_id=args_cli.task,
                seed=args_cli.seed,
                commit=_git("rev-parse", "HEAD"),
                branch=_git("branch", "--show-current"),
                enable_view=not HEADLESS and not bool(args_cli.scripted_actions),
            )
            campaign_started_wall = time.monotonic()
            scripted = deque(
                int(item.strip())
                for item in args_cli.scripted_actions.split(",")
                if item.strip()
            )
            auto_mode = args_cli.target_coverage_percent > 0.0
            auto_cycle = [
                int(item.strip())
                for item in args_cli.auto_action_cycle.split(",")
                if item.strip()
            ]
            if auto_mode and (not auto_cycle or any(value < 0 or value > 10 for value in auto_cycle)):
                raise ValueError("--auto_action_cycle must contain action IDs in [0, 10]")
            auto_cycle_index = 0
            submitted_scripted = []
            results = []
            if not args_cli.scripted_actions and not auto_mode:
                keyboard_source = KitKeyboardSource()
                print(
                    "P0_TELEOP_READY W/S tilt, D/A azimuth, E/Q roll, C/Z turn, "
                    "R/F approach/retreat, Space hold, Backspace reset, F12 snapshot, Esc exit",
                    flush=True,
                )
            current_action = None
            coverage_dwell = False
            steps_on_action = 0
            idle_updates = 0
            exit_requested = False
            display_last_wall = 0.0
            termination_reason = "simulation_stopped"
            while simulation_app.is_running() and not exit_requested:
                # Engineering display is wall-clock throttled to 30 Hz and only
                # consumes the last completed 1 Hz mask.
                wall_now = time.monotonic()
                if evaluator.view is not None and wall_now - display_last_wall >= 1.0 / 30.0:
                    evaluator.update_view()
                    display_last_wall = wall_now

                pending = []
                if keyboard_source is not None:
                    while keyboard_source.commands:
                        pending.append(keyboard_source.commands.popleft())
                if current_action is None and scripted:
                    next_action = scripted.popleft()
                    pending.append(_scripted_command(next_action))
                    submitted_scripted.append(next_action)
                if current_action is None and auto_mode:
                    achieved_percent = 100.0 * float(evaluator.accumulator.mask.mean())
                    if achieved_percent >= args_cli.target_coverage_percent:
                        termination_reason = "target_coverage_reached"
                        break
                    if action_calls >= args_cli.max_action_calls:
                        termination_reason = "max_action_calls_reached"
                        break
                    if time.monotonic() - campaign_started_wall >= args_cli.max_run_wall_time_s:
                        termination_reason = "max_run_wall_time_reached"
                        break
                    action_mask = np.asarray(term.action_mask(), dtype=np.bool_).reshape(-1)
                    selected = None
                    # Skip actions masked by the deployment boundary. This is
                    # an evaluator-side validation scheduler, not a safety bypass.
                    for _ in range(len(auto_cycle)):
                        candidate = auto_cycle[auto_cycle_index % len(auto_cycle)]
                        auto_cycle_index += 1
                        if candidate < len(action_mask) and bool(action_mask[candidate]):
                            selected = candidate
                            break
                    if selected is None:
                        termination_reason = "all_cycle_actions_masked"
                        break
                    pending.append(_scripted_command(selected))

                for command in pending:
                    if auto_mode and command.kind is CommandKind.ACTION:
                        action_attempts += 1
                    accepted, requested_exit = _handle_command(
                        command, session, term, evaluator, env
                    )
                    exit_requested = exit_requested or requested_exit
                    if accepted is not None:
                        current_action = accepted
                        steps_on_action = 0
                        if auto_mode:
                            action_calls += 1
                            action_histogram[int(accepted)] += 1
                    elif auto_mode and command.kind is CommandKind.ACTION:
                        action_rejections += 1

                if current_action is None:
                    if args_cli.scripted_actions:
                        if not scripted and len(evaluator.timings_s) >= args_cli.minimum_coverage_samples:
                            break
                    else:
                        simulation_app.update()
                        idle_updates += 1
                        if args_cli.max_idle_updates and idle_updates >= args_cli.max_idle_updates:
                            break
                    continue

                command_tensor = torch.full(
                    env.action_space.shape,
                    current_action,
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
                _, _, terminated, truncated, _ = env.step(command_tensor)
                steps_on_action += 1
                if bool(terminated[0] or truncated[0]):
                    active_terminations = [
                        name
                        for name, values in env.unwrapped.termination_manager.get_active_iterable_terms(0)
                        if bool(values[0])
                    ]
                    if auto_mode:
                        action_failures += 1
                        for name in active_terminations or ["unknown_environment_termination"]:
                            failure_codes[name] += 1
                    termination_reason = "environment_terminated:" + (
                        ",".join(active_terminations) if active_terminations else "unknown"
                    )
                    print(
                        "P0_ERROR environment terminated terms="
                        f"{active_terminations or ['unknown']}",
                        flush=True,
                    )
                    exit_code = 3
                    break
                evaluator.maybe_update()
                result = term.last_result
                if result is not None and session.busy:
                    completion = session.acknowledge(result.status.value, evaluator.sim_time_s)
                    assert completion is not None
                    evaluator.append_action_event(
                        completion, "result", device_result_payload=result.to_dict()
                    )
                    results.append(result.to_dict())
                    print(
                        f"P0_RESULT request={completion.request_id} action={completion.action_id} "
                        f"status={completion.device_result}",
                        flush=True,
                    )
                    if result.status.value == "DONE":
                        if auto_mode:
                            action_successes += 1
                        scripted_complete = bool(args_cli.scripted_actions) and not scripted and len(
                            results
                        ) >= len(
                            [value for value in args_cli.scripted_actions.split(",") if value]
                        )
                        if scripted_complete and len(evaluator.timings_s) < args_cli.minimum_coverage_samples:
                            # Keep the final executor target latched while the
                            # recorded-frame clock accumulates performance
                            # samples. Do not acknowledge/re-submit HOLD: doing
                            # so would turn small tracking sag into a new target
                            # once per second and create artificial drift.
                            coverage_dwell = True
                        else:
                            term.acknowledge_result()
                    else:
                        if auto_mode:
                            action_failures += 1
                            failure_code = getattr(
                                result.hard_failure_code, "value", result.hard_failure_code
                            )
                            failure_codes[str(failure_code)] += 1
                            termination_reason = "hard_failure"
                        evaluator.snapshot("hard_failure")
                        exit_code = 2
                        break
                    if not coverage_dwell:
                        current_action = None
                        steps_on_action = 0
                if coverage_dwell and len(evaluator.timings_s) >= args_cli.minimum_coverage_samples:
                    break
                if not coverage_dwell and steps_on_action >= args_cli.max_steps_per_action:
                    termination_reason = "action_step_budget_exceeded"
                    print("P0_ERROR action exceeded max_steps_per_action", flush=True)
                    exit_code = 4
                    break
            else:
                exit_requested = True

            if exit_code not in (2, 3, 4):
                if auto_mode:
                    target_reached = (
                        100.0 * float(evaluator.accumulator.mask.mean())
                        >= args_cli.target_coverage_percent
                    )
                    exit_code = 0 if target_reached else 6
                    print(
                        "P0_TARGET_RUN "
                        f"target_percent={args_cli.target_coverage_percent:.3f} "
                        f"achieved_percent={100.0 * float(evaluator.accumulator.mask.mean()):.3f} "
                        f"calls={action_calls} successes={action_successes} "
                        f"failures={action_failures} status={'PASS' if target_reached else 'FAIL'}",
                        flush=True,
                    )
                elif args_cli.scripted_actions:
                    expected = [int(value) for value in args_cli.scripted_actions.split(",") if value]
                    all_done = len(results) >= len(expected) and all(
                        item["status"] == "DONE" for item in results[: len(expected)]
                    )
                    no_duplicate_submission = submitted_scripted == expected
                    enough_samples = len(evaluator.timings_s) >= args_cli.minimum_coverage_samples
                    exit_code = 0 if all_done and no_duplicate_submission and enough_samples else 5
                    print(
                        f"P0_VALIDATION actions={len(expected)} done={sum(r['status'] == 'DONE' for r in results[:len(expected)])} "
                        f"coverage_updates={len(evaluator.timings_s)} unique_submissions={no_duplicate_submission} "
                        f"status={'PASS' if exit_code == 0 else 'FAIL'}",
                        flush=True,
                    )
                else:
                    termination_reason = "interactive_exit"
                    exit_code = 0
        finally:
            if keyboard_source is not None:
                keyboard_source.close()
            if evaluator is not None:
                if args_cli.target_coverage_percent > 0.0:
                    achieved_fraction = float(evaluator.accumulator.mask.mean())
                    report = {
                        "schema": "robotarm_magnetic_coverage_target_run",
                        "version": "1.0.0",
                        "target_coverage_fraction": args_cli.target_coverage_percent / 100.0,
                        "achieved_coverage_fraction": achieved_fraction,
                        "target_reached": achieved_fraction * 100.0 >= args_cli.target_coverage_percent,
                        "termination_reason": termination_reason,
                        "wall_time_s": time.monotonic() - run_started_wall,
                        "campaign_wall_time_s": time.monotonic() - campaign_started_wall,
                        "simulation_time_s": evaluator.total_sim_time_s,
                        "action_request_attempts": action_attempts,
                        "action_calls_accepted": action_calls,
                        "action_successes": action_successes,
                        "action_failures": action_failures,
                        "action_rejections": action_rejections,
                        "action_histogram": {
                            str(key): value for key, value in sorted(action_histogram.items())
                        },
                        "failure_codes": dict(sorted(failure_codes.items())),
                        "max_action_calls": args_cli.max_action_calls,
                        "max_run_wall_time_s": args_cli.max_run_wall_time_s,
                        "auto_action_cycle": [
                            int(item.strip())
                            for item in args_cli.auto_action_cycle.split(",")
                            if item.strip()
                        ],
                    }
                    (evaluator.partial_directory / "coverage_target_run.json").write_text(
                        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                final_directory = evaluator.finalize("exit")
                print(f"P0_OUTPUT {final_directory}", flush=True)
            if env is not None:
                env.close()
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
