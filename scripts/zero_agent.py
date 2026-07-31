# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many environment steps, including when a visualizer is open.",
)
parser.add_argument(
    "--save_camera_diagnostics",
    action="store_true",
    default=False,
    help="Read and save the 720p policy camera tensors at reset and shutdown.",
)
parser.add_argument(
    "--capsule_side_pose",
    action="store_true",
    default=False,
    help="Place the capsule at the stomach-motion side-lying start pose before camera diagnostics.",
)
parser.add_argument(
    "--capsule_camera_view",
    action="store_true",
    default=False,
    help="Open a second Kit window showing circular 720p policy RGB at the task camera's configured rate.",
)
parser.add_argument(
    "--capsule_pose_view",
    action="store_true",
    default=False,
    help="Open a 30 Hz world-up external follow view of the capsule pose.",
)
# Start Kit before importing the task package. The project configuration
# transitively imports USD; loading it before Kit causes two incompatible USD
# library namespaces to enter the process and abort in the dynamic linker.
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

# This task always owns an RTX capsule camera. Camera extensions (including
# omni.replicator) must be loaded by the bootstrap Kit experience; discovering
# the CameraCfg later during environment creation is too late.
args_cli.enable_cameras = True
if args_cli.capsule_camera_view or args_cli.capsule_pose_view:
    if getattr(args_cli, "headless", False):
        parser.error("Capsule views cannot be combined with --headless")
    args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: E402, F401
import robotarm_magnetic_lab.tasks  # noqa: E402, F401

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotarm_magnetic_lab.ui import (  # noqa: E402
    attach_capsule_camera_policy_view,
    attach_capsule_pose_view,
    configure_capsule_camera_view,
    configure_capsule_pose_view,
)

MAX_STEPS = 100


def print_scene_diagnostics(env, label: str, observations=None) -> None:
    """Print the live poses needed to verify the migrated scene frame."""
    import omni.usd
    from pxr import UsdGeom

    scene = env.unwrapped.scene
    robot = scene["robot"]
    capsule = scene["capsule"]
    camera = scene["capsule_camera"]
    body_names = list(robot.data.body_names)

    def body_position(name: str):
        index = body_names.index(name)
        return robot.data.body_pos_w.torch[0, index].detach().cpu().tolist()

    def body_quaternion(name: str):
        index = body_names.index(name)
        return robot.data.body_quat_w.torch[0, index].detach().cpu().tolist()

    def usd_world_position(path: str):
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
        if not prim.IsValid():
            return f"INVALID:{path}"
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        translation = matrix.ExtractTranslation()
        return [float(translation[0]), float(translation[1]), float(translation[2])]

    magl_path = "/World/envs/env_0/Scene/asm/Geometry/base_link/ballxl/ballyl/ballzl/magl"
    l6_path = "/World/envs/env_0/Scene/robotarm/Geometry/world/base_link/l1/l2/l3/l4/l5/l6"
    # Save the exact tensors exposed to the learning policy. Falling back to
    # the raw renderer buffers keeps this diagnostic usable for older configs.
    if observations is not None and "vision" in observations:
        rgb = observations["vision"]["rgb"]
        depth = observations["vision"]["depth"]
        image_source = "policy_capsule_optics"
    else:
        rgb = camera.data.output["rgb"].torch.float() / 255.0
        depth = camera.data.output["distance_to_camera"].torch
        image_source = "raw_renderer"
    finite_depth = depth[torch.isfinite(depth)]
    depth_range = (
        [float(finite_depth.min().item()), float(finite_depth.max().item())]
        if finite_depth.numel() > 0
        else []
    )
    from isaaclab.sensors import save_images_to_file

    image_dir = Path("/mnt/isaac-linux/robotarm_magnetic_lab/logs/camera")
    image_dir.mkdir(parents=True, exist_ok=True)
    save_images_to_file(
        rgb[..., :3].float().clamp(0.0, 1.0),
        str(image_dir / f"{label}_rgb.png"),
    )
    depth_preview = torch.nan_to_num(depth, nan=0.0, posinf=0.30, neginf=0.0).clamp(0.0, 0.30) / 0.30
    save_images_to_file(
        depth_preview,
        str(image_dir / f"{label}_depth.png"),
    )
    pose_view_summary = ""
    try:
        pose_camera = scene["capsule_pose_camera"]
    except KeyError:
        pose_camera = None
    if pose_camera is not None:
        pose_rgb = (
            pose_camera.data.output["rgb"].torch[..., :3]
            .float()
            .div(255.0)
            .clamp(0.0, 1.0)
        )
        save_images_to_file(
            pose_rgb,
            str(image_dir / f"{label}_external_pose_rgb.png"),
        )
        pose_view_summary = (
            f" external_pose_rgb_shape={list(pose_rgb.shape)}"
            f" external_pose_rgb_mean={float(pose_rgb.mean().item()):.3f}"
        )
    print(
        f"[SCENE_DIAG] {label} "
        f"root={robot.data.root_pos_w.torch[0].detach().cpu().tolist()} "
        f"base_link={body_position('base_link')} "
        f"l6={body_position('l6')} "
        f"l6_authored_usd={usd_world_position(l6_path)} "
        f"l6_quat_xyzw={body_quaternion('l6')} "
        f"magl={body_position('magl')} "
        f"magl_authored_usd={usd_world_position(magl_path)} "
        f"magl_quat_xyzw={body_quaternion('magl')} "
        f"capsule={capsule.data.root_pos_w.torch[0].detach().cpu().tolist()} "
        f"camera_pos={camera.data.pos_w.torch[0].detach().cpu().tolist()} "
        f"camera_quat_ros_xyzw={camera.data.quat_w_ros.torch[0].detach().cpu().tolist()} "
        f"camera_led_positions={[usd_world_position('/World/envs/env_0/Scene/MagneticDemo/target_magnet/' + name) for name in ('capsule_led_top', 'capsule_led_bottom', 'capsule_led_left', 'capsule_led_right')]} "
        f"image_source={image_source} "
        f"rgb_shape={list(rgb.shape)} rgb_mean={float(rgb.float().mean().item()):.3f} "
        f"depth_shape={list(depth.shape)} depth_finite_range={depth_range}"
        f"{pose_view_summary}",
        flush=True,
    )


def main():
    """Zero actions agent with Isaac Lab environment."""

    torch.manual_seed(42)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.capsule_camera_view:
        configure_capsule_camera_view(env_cfg)
    if args_cli.capsule_pose_view:
        configure_capsule_pose_view(env_cfg)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}", flush=True)
        print(f"[INFO]: Gym action space: {env.action_space}", flush=True)
        observations, _ = env.reset()
        if args_cli.capsule_side_pose:
            capsule = env.unwrapped.scene["capsule"]
            pose = capsule.data.root_pose_w.torch.clone()
            pose[:, :3] = torch.tensor(
                (1.0608155, 0.1145374, 0.00675),
                device=env.unwrapped.device,
            )
            # xyzw: +90 degrees about local X maps capsule +Z to world -Y.
            pose[:, 3:7] = torch.tensor(
                (0.70710678, 0.0, 0.0, 0.70710678),
                device=env.unwrapped.device,
            )
            capsule.write_root_pose_to_sim_index(root_pose=pose)
            capsule.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=env.unwrapped.device)
            )
            print("[INFO]: Applied stomach-motion capsule side pose.", flush=True)
        camera_view = (
            attach_capsule_camera_policy_view(env) if args_cli.capsule_camera_view else None
        )
        pose_view = (
            attach_capsule_pose_view(env) if args_cli.capsule_pose_view else None
        )
        if args_cli.save_camera_diagnostics:
            print_scene_diagnostics(env, "after_reset", observations)

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        sim = env.unwrapped.sim
        step_count = 0
        while True:
            if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
                break
            if sim.visualizers:
                if not any(
                    visualizer.is_running() and not visualizer.is_closed for visualizer in sim.visualizers
                ):
                    break
            elif args_cli.max_steps is None and step_count >= MAX_STEPS:
                break
            with torch.inference_mode():
                observations, _, _, _, _ = env.step(actions)
            step_count += 1

        if args_cli.save_camera_diagnostics:
            print_scene_diagnostics(env, f"after_{step_count}_steps", observations)
        print(f"[INFO]: Zero-agent smoke test completed: {step_count} steps.", flush=True)
        if camera_view is not None:
            camera_view.close()
        if pose_view is not None:
            pose_view.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
