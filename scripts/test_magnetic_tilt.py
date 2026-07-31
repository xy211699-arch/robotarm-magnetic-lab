"""Ball-joint magnetic tilt and capsule response acceptance test.

The arm remains at its validated reset pose. The three ball joints first tilt
the main magnetization axis at least 45 degrees above the ground plane, then
precess it while the capsule position, magnetization-axis angle, velocity,
magnetic wrench, and ASM clearance are recorded.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


LOG_PATH = Path("/mnt/isaac-linux/robotarm_magnetic_lab/logs/magnetic_tilt_test.jsonl")
IMAGE_DIR = Path("/mnt/isaac-linux/robotarm_magnetic_lab/logs/magnetic_tilt_images")
BALL_ACTION_INDICES = (6, 7, 8)
MAIN_MAGNET_BODY = "magl"
CAPSULE_RADIUS_M = 0.0065
CAPSULE_TOTAL_LENGTH_M = 0.025

parser = argparse.ArgumentParser(description="Test tilted rotating magnetic actuation.")
parser.add_argument("--task", default="Template-Robotarm-Magnetic-Lab-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--log_every", type=int, default=5)
parser.add_argument(
    "--max_test_steps",
    type=int,
    default=None,
    help="Optional early stop for contact/magnetic tuning.",
)
parser.add_argument(
    "--capsule_camera_view",
    action="store_true",
    default=False,
    help="Open a second Kit window showing circular 720p policy RGB at the task camera's configured rate.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.capsule_camera_view:
    if getattr(args_cli, "headless", False):
        parser.error("--capsule_camera_view cannot be combined with --headless")
    args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: E402, F401
import robotarm_magnetic_lab.tasks  # noqa: E402, F401

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.sensors import save_images_to_file  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotarm_magnetic_lab.ui import (  # noqa: E402
    attach_capsule_camera_policy_view,
    configure_capsule_camera_view,
)


def _local_z_axis_world(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    """Return the world direction of a body's local +Z axis."""
    x, y, z, w = quaternion_xyzw.unbind(-1)
    return torch.stack(
        (
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    )


def _plane_angle_deg(axis_world: torch.Tensor) -> torch.Tensor:
    """Unsigned angle between an axis and the XY ground plane."""
    return torch.rad2deg(torch.asin(torch.clamp(torch.abs(axis_world[..., 2]), 0.0, 1.0)))


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _profile(step: int) -> tuple[str, tuple[float, float, float]]:
    """Return phase name and normalized ball actions for one 20 Hz step."""
    settle_steps = 40
    tilt_steps = 100
    hold_steps = 80
    precess_steps = 160
    recover_steps = 80

    if step < settle_steps:
        return "settle", (0.0, 0.0, 0.0)
    step -= settle_steps
    if step < tilt_steps:
        fraction = _smoothstep(step / (tilt_steps - 1))
        # ballx: reset 90 deg -> 22.5 deg, giving about 67.5 deg of
        # main-magnet polarization-axis elevation above the ground plane.
        return "tilt", (-0.75 * fraction, 0.0, 0.0)
    step -= tilt_steps
    if step < hold_steps:
        return "hold_tilt", (-0.75, 0.0, 0.0)
    step -= hold_steps
    if step < precess_steps:
        phase = 2.0 * math.pi * step / (precess_steps - 1)
        return "precess", (
            -0.75,
            0.15 * math.sin(phase),
            0.75 * step / (precess_steps - 1),
        )
    step -= precess_steps
    fraction = _smoothstep(step / (recover_steps - 1))
    return "recover", (-0.75 * (1.0 - fraction), 0.0, 0.75 * (1.0 - fraction))


def _save_camera_images(observations, label: str) -> None:
    """Save the exact circular RGB/depth tensors exposed to the policy."""
    vision = observations["vision"]
    rgb = vision["rgb"][..., :3].float().clamp(0.0, 1.0)
    depth = torch.nan_to_num(
        vision["depth"], nan=0.0, posinf=0.30, neginf=0.0
    ).clamp(0.0, 0.30)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    save_images_to_file(rgb, str(IMAGE_DIR / f"{label}_rgb.png"))
    save_images_to_file(depth / 0.30, str(IMAGE_DIR / f"{label}_depth.png"))
    print(
        f"[MAGNETIC_TILT_IMAGE] label={label} "
        f"rgb_mean={float(rgb.mean().item()):.4f} "
        f"depth_nonzero={int(torch.count_nonzero(depth).item())}",
        flush=True,
    )


def _capsule_ground_gap(capsule) -> float:
    axis = _local_z_axis_world(capsule.data.root_quat_w.torch[0])
    support_height = CAPSULE_RADIUS_M + (
        0.5 * CAPSULE_TOTAL_LENGTH_M - CAPSULE_RADIUS_M
    ) * float(torch.abs(axis[2]).item())
    return float(capsule.data.root_pos_w.torch[0, 2].item()) - support_height


def _run_gravity_audit(env, base_env, capsule) -> tuple[dict, object]:
    """Drop the passive capsule with magnetic forces disabled, then reset."""
    bridge = base_env.event_manager.get_term_cfg("magnetic_collision_bridge").func
    apply_forces_original = bool(bridge.config["simulation"]["apply_forces"])
    bridge.config["simulation"]["apply_forces"] = False
    bridge._filtered_wrench.zero_()
    bridge.robot.permanent_wrench_composer.reset()
    bridge.capsule.permanent_wrench_composer.reset()

    pose = capsule.data.root_pose_w.torch.clone()
    pose[:, 2] = 0.050
    capsule.write_root_pose_to_sim_index(root_pose=pose)
    capsule.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=base_env.device)
    )

    action = torch.zeros(env.action_space.shape, device=base_env.device)
    samples = []
    observations = None
    for step in range(30):
        with torch.inference_mode():
            observations, _, _, _, _ = env.step(action)
        samples.append(
            {
                "time_s": (step + 1) * float(base_env.step_dt),
                "z_m": float(capsule.data.root_pos_w.torch[0, 2].item()),
                "vz_mps": float(capsule.data.root_lin_vel_w.torch[0, 2].item()),
                "ground_gap_m": _capsule_ground_gap(capsule),
            }
        )

    first = samples[0]
    contact_index = next(
        (index for index, sample in enumerate(samples) if sample["ground_gap_m"] <= 0.0),
        None,
    )
    audit = {
        "configured_gravity_mps2": list(base_env.sim.cfg.gravity),
        "first_50ms_vz_mps": first["vz_mps"],
        "first_50ms_effective_acceleration_mps2": first["vz_mps"]
        / float(base_env.step_dt),
        "contact_time_s": (
            samples[contact_index]["time_s"] if contact_index is not None else None
        ),
        "minimum_ground_gap_m": min(sample["ground_gap_m"] for sample in samples),
        "maximum_ground_gap_after_contact_m": (
            max(sample["ground_gap_m"] for sample in samples[contact_index:])
            if contact_index is not None
            else None
        ),
        "final_speed_mps": abs(samples[-1]["vz_mps"]),
    }
    print("[GRAVITY_AUDIT] " + json.dumps(audit), flush=True)

    bridge.config["simulation"]["apply_forces"] = apply_forces_original
    observations, _ = env.reset()
    return audit, observations


def main() -> None:
    torch.manual_seed(42)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.episode_length_s = 60.0
    if args_cli.capsule_camera_view:
        configure_capsule_camera_view(env_cfg)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot_steps = {
        39: "01_settled",
        105: "02_crossing_45deg",
        300: "03_three_axis_precession",
        459: "04_recovered",
    }
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        observations, _ = env.reset()
        camera_view = (
            attach_capsule_camera_policy_view(env) if args_cli.capsule_camera_view else None
        )
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        capsule = base_env.scene["capsule"]
        gravity_audit, observations = _run_gravity_audit(
            env, base_env, capsule
        )
        magnet_index = robot.data.body_names.index(MAIN_MAGNET_BODY)
        ball_indices = [robot.data.joint_names.index(name) for name in ("ballxj", "ballyj", "ballzj")]
        initial_capsule_position = capsule.data.root_pos_w.torch[0].detach().clone()
        full_test_steps = 40 + 100 + 80 + 160 + 80
        total_steps = (
            full_test_steps
            if args_cli.max_test_steps is None
            else min(max(args_cli.max_test_steps, 1), full_test_steps)
        )

        maximum_main_angle = 0.0
        maximum_capsule_angle = 0.0
        maximum_capsule_speed = 0.0
        maximum_capsule_angular_speed = 0.0
        maximum_ball_joint_speed = 0.0
        maximum_planar_displacement = 0.0
        maximum_ground_gap = -math.inf
        minimum_ground_gap = math.inf
        minimum_clearance = math.inf
        collision_seen = False
        completed_steps = 0

        with LOG_PATH.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "type": "session",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "total_steps": total_steps,
                        "policy_hz": 1.0 / float(base_env.step_dt),
                        "ball_action_scale_rad": math.pi / 2.0,
                        "gravity_audit": gravity_audit,
                    }
                )
                + "\n"
            )

            for step in range(total_steps):
                if base_env.sim.visualizers and not any(
                    visualizer.is_running() and not visualizer.is_closed
                    for visualizer in base_env.sim.visualizers
                ):
                    break

                phase, ball_action = _profile(step)
                action = torch.zeros(env.action_space.shape, device=base_env.device)
                action[0, list(BALL_ACTION_INDICES)] = torch.tensor(
                    ball_action, device=base_env.device
                )
                with torch.inference_mode():
                    observations, _, terminated, truncated, _ = env.step(action)
                completed_steps += 1

                magnet_axis = _local_z_axis_world(
                    robot.data.body_quat_w.torch[0, magnet_index]
                )
                capsule_axis = _local_z_axis_world(capsule.data.root_quat_w.torch[0])
                main_angle = float(_plane_angle_deg(magnet_axis).item())
                capsule_angle = float(_plane_angle_deg(capsule_axis).item())
                capsule_velocity = capsule.data.root_lin_vel_w.torch[0]
                capsule_speed = float(torch.linalg.norm(capsule_velocity).item())
                capsule_angular_speed = float(
                    torch.linalg.norm(capsule.data.root_ang_vel_w.torch[0]).item()
                )
                ball_joint_velocity = robot.data.joint_vel.torch[0, ball_indices]
                ball_joint_speed = float(torch.max(torch.abs(ball_joint_velocity)).item())
                planar_displacement = float(
                    torch.linalg.norm(
                        capsule.data.root_pos_w.torch[0, :2] - initial_capsule_position[:2]
                    ).item()
                )
                # For a capsule whose total tip-to-tip length is 25 mm, the
                # support distance along world Z varies from radius (lying)
                # to half-length (upright). Positive gap means truly airborne;
                # a small negative value is compliant-contact compression.
                ground_gap = _capsule_ground_gap(capsule)
                bridge = getattr(base_env, "_legacy_bridge_state", {})
                clearance = float(bridge["asm_clearance"][0, 0].item())
                collision = bool(bridge["collision"][0, 0].item())

                maximum_main_angle = max(maximum_main_angle, main_angle)
                maximum_capsule_angle = max(maximum_capsule_angle, capsule_angle)
                maximum_capsule_speed = max(maximum_capsule_speed, capsule_speed)
                maximum_capsule_angular_speed = max(
                    maximum_capsule_angular_speed, capsule_angular_speed
                )
                maximum_ball_joint_speed = max(maximum_ball_joint_speed, ball_joint_speed)
                maximum_planar_displacement = max(maximum_planar_displacement, planar_displacement)
                maximum_ground_gap = max(maximum_ground_gap, ground_gap)
                minimum_ground_gap = min(minimum_ground_gap, ground_gap)
                minimum_clearance = min(minimum_clearance, clearance)
                collision_seen = collision_seen or collision

                if step % args_cli.log_every == 0 or step == total_steps - 1:
                    record = {
                        "type": "step",
                        "step": completed_steps,
                        "phase": phase,
                        "ball_action": list(ball_action),
                        "ball_joint_pos_rad": robot.data.joint_pos.torch[
                            0, ball_indices
                        ].detach().cpu().tolist(),
                        "ball_joint_vel_rad_s": ball_joint_velocity.detach().cpu().tolist(),
                        "main_axis_world": magnet_axis.detach().cpu().tolist(),
                        "main_axis_plane_angle_deg": main_angle,
                        "capsule_axis_world": capsule_axis.detach().cpu().tolist(),
                        "capsule_axis_plane_angle_deg": capsule_angle,
                        "capsule_pos_m": capsule.data.root_pos_w.torch[
                            0
                        ].detach().cpu().tolist(),
                        "capsule_speed_mps": capsule_speed,
                        "capsule_angular_speed_rad_s": capsule_angular_speed,
                        "capsule_planar_displacement_m": planar_displacement,
                        "capsule_ground_gap_m": ground_gap,
                        "asm_clearance_m": clearance,
                        "collision": collision,
                    }
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    print(
                        f"[MAGNETIC_TILT] step={completed_steps} phase={phase} "
                        f"main_angle={main_angle:.2f}deg capsule_angle={capsule_angle:.2f}deg "
                        f"capsule_xy={planar_displacement:.4f}m speed={capsule_speed:.4f}m/s "
                        f"clearance={clearance:.4f}m collision={collision}",
                        flush=True,
                    )

                if step in snapshot_steps:
                    _save_camera_images(observations, snapshot_steps[step])

                if bool(torch.any(terminated)) or bool(torch.any(truncated)):
                    break

            summary = {
                "type": "summary",
                "steps_completed": completed_steps,
                "expected_steps": total_steps,
                "maximum_main_axis_plane_angle_deg": maximum_main_angle,
                "maximum_capsule_axis_plane_angle_deg": maximum_capsule_angle,
                "maximum_capsule_speed_mps": maximum_capsule_speed,
                "maximum_capsule_angular_speed_rad_s": maximum_capsule_angular_speed,
                "maximum_ball_joint_speed_rad_s": maximum_ball_joint_speed,
                "ball_joint_speed_below_1_rad_s": maximum_ball_joint_speed < 1.0,
                "maximum_capsule_planar_displacement_m": maximum_planar_displacement,
                "maximum_capsule_ground_gap_m": maximum_ground_gap,
                "minimum_capsule_ground_gap_m": minimum_ground_gap,
                "capsule_airborne_over_1mm": maximum_ground_gap > 0.001,
                "minimum_asm_clearance_m": minimum_clearance,
                "collision_seen": collision_seen,
                "main_axis_reached_45_deg": maximum_main_angle >= 45.0,
                "capsule_axis_reached_45_deg": maximum_capsule_angle >= 45.0,
                "gravity_audit": gravity_audit,
            }
            stream.write(json.dumps(summary) + "\n")
            print("[MAGNETIC_TILT_SUMMARY] " + json.dumps(summary), flush=True)

        if camera_view is not None:
            camera_view.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
