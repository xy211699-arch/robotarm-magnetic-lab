"""Collect synchronized RGB-D/state/action episodes from the Isaac Lab task."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path("/mnt/isaac-linux/robotarm_magnetic_lab")
DEFAULT_INTERFACE = PROJECT_ROOT / "configs/interfaces/robotarm_magnetic_v2.json"
DEFAULT_DATASET = PROJECT_ROOT / "datasets/robotarm_magnetic_v2_bringup"

parser = argparse.ArgumentParser(description="Collect model fine-tuning episodes.")
parser.add_argument("--task", default="Template-Robotarm-Magnetic-Stomach-Lab-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET)
parser.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
parser.add_argument(
    "--policy",
    choices=("zero", "scripted_tilt"),
    default="scripted_tilt",
    help="Data source. scripted_tilt is a deterministic bring-up teacher, not a trained policy.",
)
parser.add_argument(
    "--instruction",
    default="Rotate the capsule magnetic axis smoothly while maintaining safe arm clearance.",
)
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

if args_cli.num_envs != 1:
    raise ValueError("The v1 recorder intentionally supports exactly one environment")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: E402, F401
import robotarm_magnetic_lab.tasks  # noqa: E402, F401
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotarm_magnetic_lab.io import EpisodeWriter, JOINT_NAMES, load_interface_spec  # noqa: E402
from robotarm_magnetic_lab.ui import (  # noqa: E402
    attach_capsule_camera_policy_view,
    configure_capsule_camera_view,
)


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _profile(step: int, total_steps: int, device: str) -> torch.Tensor:
    """Return a safe deterministic action used only to validate recording."""
    action = torch.zeros((1, 9), device=device)
    if args_cli.policy == "zero":
        return action
    # Smoothly tilt ballx to -0.75, add a small bally sweep and return.
    phase = step / max(total_steps - 1, 1)
    envelope = math.sin(math.pi * phase) ** 2
    action[0, 6] = -0.75 * envelope
    action[0, 7] = 0.12 * math.sin(2.0 * math.pi * phase) * envelope
    return action


def _teacher_state(base_env, robot, capsule, joint_indices, magnet_index) -> dict:
    bridge = getattr(base_env, "_legacy_bridge_state", {})

    def bridge_value(name: str, default):
        value = bridge.get(name)
        if value is None:
            return default
        array = _numpy(value[0]).reshape(-1)
        return array.tolist() if array.size != 1 else float(array[0])

    capsule_pose = torch.cat(
        (capsule.data.root_pos_w.torch[0], capsule.data.root_quat_w.torch[0])
    )
    capsule_velocity = torch.cat(
        (capsule.data.root_lin_vel_w.torch[0], capsule.data.root_ang_vel_w.torch[0])
    )
    magnet_pose = torch.cat(
        (
            robot.data.body_pos_w.torch[0, magnet_index],
            robot.data.body_quat_w.torch[0, magnet_index],
        )
    )
    return {
        "joint_pos_rad": _numpy(robot.data.joint_pos.torch[0, joint_indices]).tolist(),
        "joint_vel_rad_s": _numpy(robot.data.joint_vel.torch[0, joint_indices]).tolist(),
        "capsule_pose_world_xyzw": _numpy(capsule_pose).tolist(),
        "capsule_velocity_world": _numpy(capsule_velocity).tolist(),
        "main_magnet_pose_world_xyzw": _numpy(magnet_pose).tolist(),
        "magnetic_wrench": bridge_value("wrench", [0.0] * 12),
        "asm_clearance_m": bridge_value("asm_clearance", float("nan")),
        "collision": bool(bridge_value("collision", 0.0)),
    }


def _ensure_dataset_manifest(root: Path, spec: dict, digest: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "dataset.json"
    manifest = {
        "dataset_name": root.name,
        "dataset_format_version": "1.0.0",
        "interface_schema_version": spec["schema_version"],
        "interface_sha256": digest,
        "task_id": args_cli.task,
        "created_by": "scripts/collect_finetune_dataset.py",
        "rates_hz": spec["rates_hz"],
        "interface": spec,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("interface_sha256") != digest:
            raise RuntimeError("Dataset already exists with a different interface schema")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    spec, digest = load_interface_spec(args_cli.interface)
    _ensure_dataset_manifest(args_cli.dataset_root, spec, digest)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.episode_length_s = max(30.0, args_cli.steps / 20.0 + 1.0)
    if args_cli.capsule_camera_view:
        configure_capsule_camera_view(env_cfg)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        capsule = base_env.scene["capsule"]
        joint_indices = [robot.data.joint_names.index(name) for name in JOINT_NAMES]
        magnet_index = robot.data.body_names.index("magl")
        camera_view = (
            attach_capsule_camera_policy_view(env) if args_cli.capsule_camera_view else None
        )
        camera = base_env.scene["capsule_camera"]
        control_hz = int(spec["rates_hz"]["control"])
        camera_hz = int(spec["rates_hz"]["camera"])
        if control_hz % camera_hz != 0:
            raise RuntimeError("Camera rate must divide the control rate exactly")
        expected_control_rows_per_frame = control_hz // camera_hz

        for episode_number in range(args_cli.episodes):
            observations, _ = env.reset(seed=args_cli.seed + episode_number)
            episode_id = (
                datetime.now().strftime("%Y%m%d_%H%M%S")
                + f"_e{episode_number:04d}"
            )
            writer = EpisodeWriter(
                args_cli.dataset_root,
                episode_id,
                spec,
                digest,
                metadata={
                    "task_id": args_cli.task,
                    "seed": args_cli.seed + episode_number,
                    "policy_source": args_cli.policy,
                    "language_instruction": args_cli.instruction,
                    "policy_inference_rate_hz": spec["rates_hz"]["policy"],
                    "control_rate_hz": spec["rates_hz"]["control"],
                    "camera_rate_hz": spec["rates_hz"]["camera"],
                },
            )
            success = False
            termination_reason = "step_limit"
            max_capsule_speed = 0.0
            source_camera_frame = None
            dataset_camera_frame = -1
            camera_timestamp_s = 0.0
            try:
                for step in range(args_cli.steps):
                    action = _profile(step, args_cli.steps, base_env.device)
                    rgb = _numpy(observations["vision"]["rgb"][0])
                    depth = _numpy(observations["vision"]["depth"][0, ..., 0])
                    policy_state = _numpy(observations["policy"][0])
                    teacher = _teacher_state(
                        base_env, robot, capsule, joint_indices, magnet_index
                    )
                    current_source_frame = int(camera.frame.torch[0].item())
                    camera_is_new = current_source_frame != source_camera_frame
                    if camera_is_new:
                        source_camera_frame = current_source_frame
                        dataset_camera_frame += 1
                        camera_timestamp_s = step / float(control_hz)
                    max_capsule_speed = max(
                        max_capsule_speed,
                        float(np.linalg.norm(teacher["capsule_velocity_world"][:3])),
                    )

                    with torch.inference_mode():
                        next_observations, reward, terminated, truncated, _ = env.step(action)
                    targets = robot.data.joint_pos_target.torch[0, joint_indices]
                    is_terminated = bool(terminated[0].item())
                    is_truncated = bool(truncated[0].item())
                    writer.append(
                        step=step,
                        control_time_s=step / float(control_hz),
                        camera_frame_id=dataset_camera_frame,
                        camera_timestamp_s=camera_timestamp_s,
                        camera_is_new=camera_is_new,
                        rgb=rgb,
                        depth_m=depth,
                        policy_state=policy_state,
                        action_command=_numpy(action[0]),
                        action_applied_joint_target_rad=_numpy(targets),
                        teacher=teacher,
                        reward=float(reward[0].item()),
                        terminated=is_terminated,
                        truncated=is_truncated,
                    )
                    observations = next_observations
                    if is_terminated or is_truncated:
                        termination_reason = "collision" if is_terminated else "time_limit"
                        break
                success = termination_reason == "step_limit"
                path = writer.close(
                    success=success,
                    termination_reason=termination_reason,
                    extra_summary={
                        "maximum_capsule_speed_mps": max_capsule_speed,
                        "expected_control_rows_per_camera_frame": (
                            expected_control_rows_per_frame
                        ),
                    },
                )
                print(f"[DATASET] committed {path}", flush=True)
            except Exception:
                writer.abort()
                raise
        if camera_view is not None:
            camera_view.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
