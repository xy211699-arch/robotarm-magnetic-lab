"""Live TASK-007 runtime inspection for the flat virtual-magnet task."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect the TASK-007 runtime contract.")
parser.add_argument(
    "--task",
    default="Template-Robotarm-Magnetic-Virtual-Magnet-Flat-Lab-v0",
)
parser.add_argument("--action_id", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.scene.capsule_camera = None
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        unwrapped = env.unwrapped
        capsule = unwrapped.scene["capsule"]
        robot = unwrapped.scene["robot"]
        start_capsule = capsule.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
        start_joints = robot.data.joint_pos.torch[0].detach().cpu().numpy().copy()
        start_joint_targets = robot.data.joint_pos_target.torch[0].detach().cpu().numpy().copy()
        actions = torch.tensor([[float(args_cli.action_id)]], device=unwrapped.device)
        env.step(actions)
        bridge = unwrapped._virtual_magnet_bridge
        audit = {key: _jsonable(value) for key, value in bridge.audit.items()}
        end_capsule = capsule.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
        end_joints = robot.data.joint_pos.torch[0].detach().cpu().numpy().copy()
        end_joint_targets = robot.data.joint_pos_target.torch[0].detach().cpu().numpy().copy()

        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        capsule_prim = stage.GetPrimAtPath(
            "/World/envs/env_0/Scene/MagneticDemo/target_magnet"
        )
        debug_prim = stage.GetPrimAtPath(
            "/World/envs/env_0/Scene/MagneticDemo/virtual_external_magnet"
        )
        rigid_api = UsdPhysics.RigidBodyAPI(capsule_prim)
        kinematic = rigid_api.GetKinematicEnabledAttr().Get()
        debug_has_rigid = debug_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        debug_has_collision = debug_prim.HasAPI(UsdPhysics.CollisionAPI)
        summary = {
            "action_id": args_cli.action_id,
            "physics_substeps": audit["physics_substeps"],
            "feedback_updates": audit["feedback_updates"],
            "lifecycle": audit["lifecycle"],
            "result": audit["result"],
            "capsule_dynamic": bool(not kinematic),
            "capsule_pose_delta_norm": float(np.linalg.norm(end_capsule - start_capsule)),
            "robot_joint_delta_norm": float(np.linalg.norm(end_joints - start_joints)),
            "robot_joint_target_delta_norm": float(
                np.linalg.norm(end_joint_targets - start_joint_targets)
            ),
            "action_terms": list(unwrapped.action_manager.active_terms),
            "debug_magnet_has_rigid_api": bool(debug_has_rigid),
            "debug_magnet_has_collision_api": bool(debug_has_collision),
            "finite_audit": bool(
                all(
                    np.isfinite(np.asarray(audit[key])).all()
                    for key in (
                        "desired_wrench",
                        "model_raw_wrench",
                        "model_filtered_wrench",
                        "applied_wrench",
                        "virtual_magnet_position",
                        "virtual_magnet_quaternion_xyzw",
                    )
                )
            ),
            "applied_equals_filtered": bool(
                np.allclose(audit["applied_wrench"], audit["model_filtered_wrench"])
            ),
            "audit": audit,
        }
        print("VIRTUAL_MAGNET_RUNTIME_CONTRACT " + json.dumps(summary, sort_keys=True), flush=True)
        assert summary["physics_substeps"] == 240
        assert summary["feedback_updates"] == 60
        assert summary["capsule_dynamic"]
        assert not summary["debug_magnet_has_rigid_api"]
        assert not summary["debug_magnet_has_collision_api"]
        assert summary["finite_audit"]
        assert summary["applied_equals_filtered"]
        assert summary["robot_joint_target_delta_norm"] == 0.0
        assert summary["action_terms"] == ["request", "magnetic_physics"]
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
