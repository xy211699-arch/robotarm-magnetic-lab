#!/usr/bin/env python3
"""TASK-009B Gate-5 three-view teleoperation at exact 10 Hz boundaries."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
sys.path.insert(0, str(ROOT / "scripts"))
HEADLESS = "--headless" in sys.argv
if HEADLESS:
    sys.argv.remove("--headless")
    os.environ["HEADLESS"] = "1"

from isaaclab.app import AppLauncher

from _artifact_paths import artifact_root


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
DEFAULT_ARTIFACT_ROOT = artifact_root(ROOT) / "task009b_three_view"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--initial_alpha", type=float, choices=(0.0, 0.5, 1.0), default=0.5)
parser.add_argument("--render_fps", type=int, choices=(60,), default=60)
parser.add_argument(
    "--pose_manifest",
    type=Path,
    default=ROOT / "configs/task009b/pose_library_manifest_v1.json",
)
parser.add_argument(
    "--pose_id",
    default="",
    help="Stored pose ID; empty selects the first frozen train live-reload pose.",
)
parser.add_argument("--output_directory", type=Path, default=DEFAULT_ARTIFACT_ROOT)
parser.add_argument("--max_cycles", type=int, default=0)
parser.add_argument(
    "--scripted_actions",
    default="",
    help="Headless smoke format MODE:ALPHA:CYCLES, for example HOLD:0.5:3.",
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
from robotarm_magnetic_lab.coverage.entry_pose_library import (
    POSE_LIBRARY_MANIFEST_SCHEMA,
    file_sha256,
    manifest_hash,
    read_jsonl,
    stable_record_is_valid,
)
from robotarm_magnetic_lab.coverage.simulator_runtime import P0CoverageRuntime
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
    attach_capsule_recorded_camera_view,
    configure_capsule_recorded_camera_view,
)


class KitHeldKeyboard:
    """Forward Kit key transitions into the approved held-key resolver."""

    def __init__(self, alpha: float) -> None:
        self.state = ParameterizedForceKeyboard(alpha)
        self.events = deque()
        self.acceptance = None
        self._input = carb.input.acquire_input_interface()
        self._device = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._device, self._on_event
        )

    def _on_event(self, event, *_args):
        key = event.input.name if hasattr(event.input, "name") else str(event.input)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            normalized = key.upper()
            if normalized == "Y":
                self.acceptance = "confirmed"
                return True
            if normalized == "N":
                self.acceptance = "rejected"
                return True
            update = self.state.key_event(key, True)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            update = self.state.key_event(key, False)
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


def _parse_scripted_actions(value: str) -> deque[tuple[ParameterizedForceMode, float]]:
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
            raise ValueError(f"invalid alpha/count in scripted action {token!r}")
        cycles.extend((mode, alpha) for _ in range(count))
    return cycles


def _load_pose(manifest_path: Path, requested_pose_id: str) -> tuple[dict, str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != POSE_LIBRARY_MANIFEST_SCHEMA:
        raise RuntimeError("pose-library manifest schema mismatch")
    expected = manifest.get("config_sha256")
    payload = {key: value for key, value in manifest.items() if key != "config_sha256"}
    if manifest_hash(payload) != expected:
        raise RuntimeError("pose-library manifest hash mismatch")
    data_path = Path(manifest["data_path"])
    if not data_path.is_file() or file_sha256(data_path) != manifest["data_sha256"]:
        raise RuntimeError("external pose-library data is missing or has the wrong hash")
    records = {record["pose_id"]: record for record in read_jsonl(data_path)}
    pose_id = requested_pose_id or manifest["fixed_live_reload_pose_ids"]["train"][0]
    if pose_id not in records or not stable_record_is_valid(records[pose_id]):
        raise RuntimeError(f"pose {pose_id!r} is not a valid frozen Gate-3 pose")
    return records[pose_id], str(manifest["config_sha256"])


def _write_pose(base, pose_record: dict) -> None:
    capsule = base.scene["capsule"]
    pose = torch.as_tensor(
        pose_record["pose_world_xyzw"], device=base.device, dtype=torch.float32
    ).reshape(1, 7)
    capsule.write_root_pose_to_sim_index(root_pose=pose)
    capsule.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=base.device, dtype=torch.float32)
    )
    term = base.action_manager.get_term("parameterized_force")
    term.reset()
    capsule.permanent_wrench_composer.reset()
    base.sim.forward()
    base.scene.update(0.0)


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args), cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    pose_record, pose_manifest_sha = _load_pose(args_cli.pose_manifest, args_cli.pose_id)
    scripted = _parse_scripted_actions(args_cli.scripted_actions)
    if HEADLESS and not scripted:
        scripted.extend((ParameterizedForceMode.HOLD, args_cli.initial_alpha) for _ in range(3))
    output = args_cli.output_directory / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%fZ"
    )

    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    cfg.sim.render_interval = PHYSICS_HZ // args_cli.render_fps
    if not HEADLESS:
        configure_capsule_recorded_camera_view(cfg)

    env = keyboard = camera_view = evaluator = None
    cycle = 0
    reason = "initialization_failed"
    active_signature = None
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            base = env.unwrapped
            _write_pose(base, pose_record)
            hold_after_pose = torch.tensor(
                [[float(ParameterizedForceMode.HOLD), float(args_cli.initial_alpha)]],
                device=base.device,
                dtype=torch.float32,
            )
            env.step(hold_after_pose)
            evaluator = P0CoverageRuntime(
                env,
                output,
                task_id=TASK_ID,
                seed=int(pose_record["candidate_seed"]),
                commit=_git_value("rev-parse", "HEAD"),
                branch=_git_value("branch", "--show-current"),
                enable_view=not HEADLESS,
                require_camera_facing_normal=True,
                camera_facing_normal_sign=-1,
                raycast_device=str(args_cli.device),
            )
            if not HEADLESS:
                keyboard = KitHeldKeyboard(args_cli.initial_alpha)
                camera_view = attach_capsule_recorded_camera_view(env)

            initial_sync = dict(base._task009b_policy_rgb_sync_latest)
            initial_update = evaluator.maybe_update(
                expected_camera_frame=int(initial_sync["frame"])
            )
            evaluator.update_view()
            if initial_update is None:
                raise RuntimeError("initial 10 Hz coverage boundary was not captured")
            print(
                "TASK009B_THREE_VIEW_READY "
                f"external_hz={args_cli.render_fps} capsule_rgb_hz={CONTROL_HZ} "
                f"coverage_hz={CONTROL_HZ} physics_hz={PHYSICS_HZ} "
                f"substeps={PHYSICS_STEPS_PER_CONTROL} pose_id={pose_record['pose_id']} "
                f"pose_manifest_sha256={pose_manifest_sha} "
                "views='main viewport; Capsule Camera Recorded 10 Hz; P0 Stomach Coverage' "
                "keys='hold A/D MOVE-/+; hold Q/E VIEW-/+; hold W UP; "
                "Z/X/C alpha=0/0.5/1; SPACE HOLD; R reset; P snapshot; "
                "Y confirm; N reject; ESC exit'",
                flush=True,
            )
            reason = "simulation_closed"
            while simulation_app.is_running():
                reset_requested = snapshot_requested = exit_requested = False
                if keyboard is not None:
                    while keyboard.events:
                        event = keyboard.events.popleft()
                        reset_requested |= event.kind is ParameterizedKeyboardEventKind.RESET
                        snapshot_requested |= event.kind is ParameterizedKeyboardEventKind.SNAPSHOT
                        exit_requested |= event.kind is ParameterizedKeyboardEventKind.EXIT
                    if keyboard.acceptance is not None:
                        reason = f"human_{keyboard.acceptance}"
                        print(
                            "TASK009B_THREE_VIEW_ACCEPTANCE " + keyboard.acceptance,
                            flush=True,
                        )
                        break
                    mode, alpha = keyboard.state.command
                elif scripted:
                    mode, alpha = scripted.popleft()
                else:
                    reason = "script_complete"
                    break

                if exit_requested:
                    reason = "keyboard_exit"
                    break
                if reset_requested:
                    evaluator.reset()
                    env.reset()
                    _write_pose(base, pose_record)
                    env.step(hold_after_pose)
                    keyboard.state.release_all()
                    active_signature = None
                    reset_sync = dict(base._task009b_policy_rgb_sync_latest)
                    evaluator.maybe_update(expected_camera_frame=int(reset_sync["frame"]))
                    evaluator.update_view()
                    print("TASK009B_THREE_VIEW_RESET", flush=True)
                    continue
                if snapshot_requested:
                    metadata = evaluator.snapshot("manual")
                    print(
                        "TASK009B_THREE_VIEW_SNAPSHOT " + json.dumps(metadata, sort_keys=True),
                        flush=True,
                    )

                signature = (int(mode), float(alpha))
                if signature != active_signature:
                    print(
                        f"TASK009B_THREE_VIEW_CONTROL mode={mode.name} alpha={alpha:.1f}",
                        flush=True,
                    )
                    active_signature = signature

                started = time.perf_counter()
                action = torch.tensor(
                    [[float(mode), float(alpha)]], device=base.device, dtype=torch.float32
                )
                observation, *_ = env.step(action)
                trace = base.action_manager.get_term("parameterized_force").current_cycle_trace
                if len(trace) != PHYSICS_STEPS_PER_CONTROL:
                    raise RuntimeError(f"expected 24 physics substeps, got {len(trace)}")
                pose = base.scene["capsule"].data.root_pose_w.torch[0].detach().cpu().numpy()
                if not np.isfinite(pose).all():
                    raise RuntimeError("non-finite capsule state at 10 Hz boundary")
                sync = dict(base._task009b_policy_rgb_sync_latest)
                update = evaluator.maybe_update(expected_camera_frame=int(sync["frame"]))
                if update is None:
                    raise RuntimeError("coverage did not update at a 10 Hz action boundary")
                evaluator.update_view()
                cycle += 1
                print(
                    "TASK009B_THREE_VIEW_BOUNDARY "
                    f"cycle={cycle} mode={mode.name} alpha={alpha:.1f} "
                    f"rgb_frame={evaluator.latest_record['frame_id']} "
                    f"area_coverage_percent={100.0 * update.coverage_fraction:.3f}",
                    flush=True,
                )
                if args_cli.max_cycles and cycle >= args_cli.max_cycles:
                    reason = "max_cycles"
                    break
                if args_cli.realtime:
                    remaining = 1.0 / CONTROL_HZ - (time.perf_counter() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)

            summary = {
                "schema": "task009b_three_view_session_v1",
                "reason": reason,
                "cycles": cycle,
                "physics_hz": PHYSICS_HZ,
                "control_hz": CONTROL_HZ,
                "external_view_hz": args_cli.render_fps,
                "capsule_rgb_hz": CONTROL_HZ,
                "coverage_view_hz": CONTROL_HZ,
                "pose_id": pose_record["pose_id"],
                "human_acceptance": (
                    reason.removeprefix("human_") if reason.startswith("human_") else "needs_input"
                ),
            }
            final = evaluator.finalize(reason)
            summary["artifact_directory"] = str(final)
            print("TASK009B_THREE_VIEW_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
            evaluator = None
            return 0
        finally:
            if keyboard is not None:
                keyboard.close()
            if camera_view is not None:
                camera_view.close()
            if evaluator is not None:
                evaluator.finalize("exception")
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
