"""Deterministic nine-axis validation for the robotarm magnetic environment.

This is a permanent acceptance test, not a motion demo. It excites one joint
at a time around the configured reset pose and records joint tracking, magnetic
field anchoring, capsule motion, and collision clearance in one JSONL log.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from isaaclab.app import AppLauncher


JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6", "ballxj", "ballyj", "ballzj")
LOG_PATH = Path("/mnt/isaac-linux/robotarm_magnetic_lab/logs/interface_validation.jsonl")

parser = argparse.ArgumentParser(description="Validate all nine robot/ball action channels.")
parser.add_argument("--task", type=str, default="Template-Robotarm-Magnetic-Lab-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--phase_steps", type=int, default=60, help="Steps per joint excitation.")
parser.add_argument("--settle_steps", type=int, default=40)
parser.add_argument("--arm_amplitude", type=float, default=0.5, help="Normalized arm action amplitude.")
parser.add_argument("--ball_amplitude", type=float, default=1.0, help="Normalized ball action amplitude.")
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument(
    "--capsule_camera_view",
    action="store_true",
    default=False,
    help="Open a second Kit window showing circular 720p policy RGB at the task camera's configured rate.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.capsule_camera_view:
    if getattr(args_cli, "headless", False):
        parser.error("--capsule_camera_view cannot be combined with --headless")
    args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: E402, F401
import robotarm_magnetic_lab.tasks  # noqa: E402, F401

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotarm_magnetic_lab.ui import (  # noqa: E402
    attach_capsule_camera_policy_view,
    configure_capsule_camera_view,
)


def _tolist(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().reshape(-1).tolist()


def main() -> None:
    torch.manual_seed(42)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.episode_length_s = 180.0
    if args_cli.capsule_camera_view:
        configure_capsule_camera_view(env_cfg)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as log_stream:
        session = {
            "type": "session",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "joint_order": JOINT_NAMES,
            "phase_steps": args_cli.phase_steps,
            "settle_steps": args_cli.settle_steps,
        }
        log_stream.write(json.dumps(session) + "\n")

        with launch_simulation(env_cfg, args_cli):
            env = gym.make(args_cli.task, cfg=env_cfg)
            env.reset()
            camera_view = (
                attach_capsule_camera_policy_view(env) if args_cli.capsule_camera_view else None
            )
            base_env = env.unwrapped
            robot = base_env.scene["robot"]
            capsule = base_env.scene["capsule"]
            joint_indices = [robot.data.joint_names.index(name) for name in JOINT_NAMES]
            magnet_index = robot.data.body_names.index("magl")
            initial_joint_pos = robot.data.joint_pos.torch[0, joint_indices].detach().clone()
            initial_capsule_pos = capsule.data.root_pos_w.torch[0].detach().clone()
            max_joint_excursion = torch.zeros(len(JOINT_NAMES), device=base_env.device)
            min_clearance = float("inf")
            max_field_anchor_error = 0.0
            collision_seen = False
            terminated_early = False
            global_step = 0

            total_steps = args_cli.settle_steps + len(JOINT_NAMES) * args_cli.phase_steps
            sim = base_env.sim
            for step in range(total_steps):
                if sim.visualizers and not any(
                    visualizer.is_running() and not visualizer.is_closed for visualizer in sim.visualizers
                ):
                    terminated_early = True
                    break

                action = torch.zeros(env.action_space.shape, device=base_env.device)
                if step < args_cli.settle_steps:
                    phase_name = "settle"
                    active_index = None
                    phase_step = step
                else:
                    motion_step = step - args_cli.settle_steps
                    active_index = motion_step // args_cli.phase_steps
                    phase_step = motion_step % args_cli.phase_steps
                    phase_name = JOINT_NAMES[active_index]
                    amplitude = (
                        args_cli.arm_amplitude if active_index < 6 else args_cli.ball_amplitude
                    )
                    # One smooth 0 -> +peak -> 0 -> -peak -> 0 cycle.
                    action[:, active_index] = amplitude * np.sin(
                        2.0 * np.pi * phase_step / args_cli.phase_steps
                    )

                with torch.inference_mode():
                    _, _, terminated, truncated, _ = env.step(action)
                global_step += 1

                current_joint_pos = robot.data.joint_pos.torch[0, joint_indices]
                excursion = torch.abs(current_joint_pos - initial_joint_pos)
                max_joint_excursion = torch.maximum(max_joint_excursion, excursion)
                bridge_state = getattr(base_env, "_legacy_bridge_state", {})
                clearance_tensor = bridge_state.get("asm_clearance")
                collision_tensor = bridge_state.get("collision")
                field_anchor_tensor = bridge_state.get("field_anchor")
                clearance = (
                    float(clearance_tensor[0, 0].item())
                    if clearance_tensor is not None
                    else float("nan")
                )
                collision = (
                    bool(collision_tensor[0, 0].item())
                    if collision_tensor is not None
                    else False
                )
                magnet_pos = robot.data.body_pos_w.torch[0, magnet_index]
                if field_anchor_tensor is not None and torch.isfinite(field_anchor_tensor[0]).all():
                    field_error = float(torch.linalg.norm(field_anchor_tensor[0] - magnet_pos).item())
                    max_field_anchor_error = max(max_field_anchor_error, field_error)
                else:
                    field_error = float("nan")

                if np.isfinite(clearance):
                    min_clearance = min(min_clearance, clearance)
                collision_seen = collision_seen or collision

                should_log = (
                    step % args_cli.log_every == 0
                    or phase_step == args_cli.phase_steps - 1
                    or bool(torch.any(terminated))
                    or bool(torch.any(truncated))
                )
                if should_log:
                    record = {
                        "type": "step",
                        "step": global_step,
                        "phase": phase_name,
                        "active_joint": None if active_index is None else JOINT_NAMES[active_index],
                        "action": _tolist(action[0]),
                        "joint_pos": _tolist(current_joint_pos),
                        "joint_vel": _tolist(robot.data.joint_vel.torch[0, joint_indices]),
                        "magl_pos": _tolist(magnet_pos),
                        "field_anchor_error_m": field_error,
                        "capsule_pos": _tolist(capsule.data.root_pos_w.torch[0]),
                        "capsule_lin_vel": _tolist(capsule.data.root_lin_vel_w.torch[0]),
                        "asm_clearance_m": clearance,
                        "collision": collision,
                    }
                    line = json.dumps(record)
                    log_stream.write(line + "\n")
                    log_stream.flush()
                    print(
                        f"[INTERFACE_TEST] step={global_step} phase={phase_name} "
                        f"clearance={clearance:.6f} field_error={field_error:.6f} "
                        f"collision={collision}",
                        flush=True,
                    )

                if bool(torch.any(terminated)):
                    terminated_early = True
                    break

            capsule_displacement = float(
                torch.linalg.norm(capsule.data.root_pos_w.torch[0] - initial_capsule_pos).item()
            )
            summary = {
                "type": "summary",
                "steps_completed": global_step,
                "expected_steps": total_steps,
                "terminated_early": terminated_early,
                "max_joint_excursion_rad": {
                    name: value
                    for name, value in zip(JOINT_NAMES, _tolist(max_joint_excursion), strict=True)
                },
                "minimum_asm_clearance_m": min_clearance,
                "collision_seen": collision_seen,
                "maximum_field_anchor_error_m": max_field_anchor_error,
                "capsule_displacement_m": capsule_displacement,
            }
            log_stream.write(json.dumps(summary) + "\n")
            log_stream.flush()
            print("[INTERFACE_TEST_SUMMARY] " + json.dumps(summary), flush=True)
            if camera_view is not None:
                camera_view.close()
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
