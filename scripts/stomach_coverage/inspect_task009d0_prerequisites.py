#!/usr/bin/env python3
"""Read-only live prerequisite inspection for TASK-009D0 Gate 1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, visualizer=[])
args_cli = parser.parse_args()
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import torch
import warp as wp

import robotarm_magnetic_lab.tasks  # noqa: F401, E402
from isaaclab.app import launch_simulation
from isaaclab.sensors import Camera
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.coverage.entry_pose_library import file_sha256
from robotarm_magnetic_lab.runtime.task009d0_config import (
    TASK009D0_CONFIG_PATH,
    load_task009d0_config,
    validate_task009d0_repository_inputs,
)


REFERENCE_TASK_ID = (
    "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
)


def _metadata_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return None


def _git_head(path: Path) -> str | None:
    return _command(["git", "-C", str(path), "rev-parse", "HEAD"])


def _tensor(value):
    return getattr(value, "torch", value)


def main() -> int:
    config = load_task009d0_config(TASK009D0_CONFIG_PATH)
    inputs = validate_task009d0_repository_inputs(config, repository_root=ROOT)
    output_dir = args_cli.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "task009d0_prerequisites.json"
    env = None
    failures: list[str] = []
    requested_device = str(args_cli.device)
    cfg = parse_env_cfg(
        REFERENCE_TASK_ID,
        device=requested_device,
        num_envs=1,
        use_fabric=True,
    )
    if str(cfg.sim.device) != requested_device:
        failures.append(
            f"parsed PhysX device {cfg.sim.device!s} differs from {requested_device}"
        )
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(REFERENCE_TASK_ID, cfg=cfg)
            observation, _ = env.reset(seed=990009)
            base = env.unwrapped
            camera = base.scene["capsule_camera"]
            rgb = _tensor(camera.data.output["rgb"])
            capsule_pose = _tensor(base.scene["capsule"].data.root_pose_w)
            if not torch.cuda.is_available():
                failures.append("torch CUDA is unavailable")
            if rgb.device.type != "cuda":
                failures.append(f"RTX camera tensor is on {rgb.device}, not CUDA")
            if capsule_pose.device.type != "cuda":
                failures.append(f"PhysX capsule state is on {capsule_pose.device}, not CUDA")
            if not torch.isfinite(rgb).all().item():
                failures.append("RTX camera returned non-finite RGB")
            if tuple(rgb.shape[-3:-1]) != (720, 1280):
                failures.append(f"camera shape is {tuple(rgb.shape)}, expected 720x1280")
            warp_device = str(wp.get_device(requested_device))
            if "cuda" not in warp_device.lower():
                failures.append(f"Warp device is not CUDA: {warp_device}")
            try:
                import omni.kit.app

                kit_build = omni.kit.app.get_app().get_build_version()
            except Exception as exc:  # pragma: no cover - host compatibility evidence
                kit_build = f"unavailable: {type(exc).__name__}: {exc}"
            memory = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
            nvidia_csv = _command(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ]
            )
            record = {
                "schema": "robotarm_magnetic_lab.task009d0_prerequisites",
                "version": 1,
                "status": "pass" if not failures else "partial",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "repository_root": str(ROOT),
                "repository_head": _git_head(ROOT),
                "task009d0_config_sha256": config["config_sha256"],
                "reference_task_id": REFERENCE_TASK_ID,
                "requested_device": requested_device,
                "environment_device": str(base.device),
                "physx_config_device": str(cfg.sim.device),
                "physx_state_device": str(capsule_pose.device),
                "camera_tensor_device": str(rgb.device),
                "camera_rgb_shape": list(rgb.shape),
                "camera_rgb_finite": bool(torch.isfinite(rgb).all().item()),
                "raycast_device": warp_device,
                "camera_private_update_signature": str(
                    inspect.signature(Camera._update_buffers_impl)
                ),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "warp_version": getattr(wp, "__version__", _metadata_version("warp-lang")),
                "rsl_rl_version": _metadata_version("rsl-rl-lib", "rsl-rl"),
                "isaac_lab_version": _metadata_version("isaaclab"),
                "isaac_lab_head": _git_head(Path("/mnt/isaac-linux/IsaacLab")),
                "isaac_sim_version": _metadata_version("isaacsim"),
                "kit_build_version": kit_build,
                "nvidia_smi_csv": nvidia_csv,
                "torch_cuda_memory_free_bytes": int(memory[0]),
                "torch_cuda_memory_total_bytes": int(memory[1]),
                "input_paths": {key: str(value) for key, value in inputs.items()},
                "input_sha256": {key: file_sha256(value) for key, value in inputs.items()},
                "actor_observation_groups": sorted(observation),
                "failures": failures,
            }
            output_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                "TASK009D0_PREREQUISITES "
                + json.dumps(
                    {
                        "status": record["status"],
                        "path": str(output_path),
                        "bytes": output_path.stat().st_size,
                        "sha256": file_sha256(output_path),
                        "physx_device": record["physx_state_device"],
                        "camera_device": record["camera_tensor_device"],
                        "raycast_device": record["raycast_device"],
                        "failures": failures,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0 if not failures else 1
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
