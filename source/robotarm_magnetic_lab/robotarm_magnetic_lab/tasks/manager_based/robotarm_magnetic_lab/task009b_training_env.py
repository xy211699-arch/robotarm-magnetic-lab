"""Formal TASK-009B environment semantics for training and acceptance tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from .controllers.parameterized_force import ParameterizedForceMode
from robotarm_magnetic_lab.baselines.random_policies import load_random_baseline_config
from robotarm_magnetic_lab.coverage.entry_pose_library import (
    POSE_LIBRARY_MANIFEST_SCHEMA,
    file_sha256,
    manifest_hash,
    read_jsonl,
    stable_record_is_valid,
)


RESET_HOLD_CYCLES = 10
TASK009C_OPTION_KEY = "task009c_initial_pose"
TASK009C_CONFIG_PATH = (
    Path(__file__).resolve().parents[6]
    / "configs/task009c/random_baseline_preexperiment_v1.json"
)


def _stable_rgb_digest(rgb: torch.Tensor) -> str:
    """Return a deterministic lightweight content digest without copying a full frame."""
    flat = rgb.detach().reshape(-1)
    if flat.numel() == 0:
        raise RuntimeError("policy RGB tensor is empty")
    count = min(1024, flat.numel())
    indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
    sample = flat.index_select(0, indices).float().cpu().numpy()
    metadata = np.asarray(
        [float(flat.min().item()), float(flat.max().item()), float(flat.mean().item())],
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(rgb.shape)).encode("ascii"))
    digest.update(str(rgb.dtype).encode("ascii"))
    digest.update(sample.tobytes())
    digest.update(metadata.tobytes())
    return digest.hexdigest()


def _load_task009c_pose_records(config_path: Path = TASK009C_CONFIG_PATH):
    """Load the frozen manifest and the validation poses authorized by the config."""
    config = load_random_baseline_config(config_path)
    root = Path(config_path).resolve().parents[2]
    pose_config = config["pose_library"]
    manifest_path = root / pose_config["manifest_path"]
    if file_sha256(manifest_path) != pose_config["manifest_file_sha256"]:
        raise RuntimeError("TASK-009C pose manifest file hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != POSE_LIBRARY_MANIFEST_SCHEMA:
        raise RuntimeError("TASK-009C pose manifest schema mismatch")
    payload = {key: value for key, value in manifest.items() if key != "config_sha256"}
    if manifest_hash(payload) != manifest.get("config_sha256"):
        raise RuntimeError("TASK-009C pose manifest deterministic hash mismatch")
    if manifest["config_sha256"] != pose_config["manifest_config_sha256"]:
        raise RuntimeError("TASK-009C pose manifest config hash mismatch")
    data_path = Path(pose_config["data_path"])
    if not data_path.is_file() or file_sha256(data_path) != pose_config["data_sha256"]:
        raise RuntimeError("TASK-009C external pose library is missing or has the wrong hash")
    by_id = {record["pose_id"]: record for record in read_jsonl(data_path)}
    allowed = {}
    for pose_id in config["validation_pose_ids"]:
        record = by_id.get(pose_id)
        if record is None or record.get("split") != "validation" or not stable_record_is_valid(record):
            raise RuntimeError(f"TASK-009C frozen validation pose is invalid: {pose_id}")
        allowed[pose_id] = record
    return config, manifest, allowed


def _validate_task009c_pose_record(
    requested: dict,
    *,
    config: dict,
    manifest: dict,
    allowed: dict[str, dict],
) -> tuple[str, np.ndarray]:
    """Validate an options pose without trusting caller-provided coordinates."""
    if not isinstance(requested, dict):
        raise ValueError("task009c_initial_pose must be a dictionary")
    required = ("pose_id", "split", "pose_world_xyzw")
    missing = [name for name in required if name not in requested]
    if missing:
        raise ValueError(f"task009c_initial_pose missing fields: {missing}")
    pose_id = str(requested["pose_id"])
    if requested["split"] != "validation" or pose_id not in config["validation_pose_ids"]:
        raise ValueError(f"TASK-009C pose is not in the frozen validation set: {pose_id}")
    if pose_id not in allowed:
        raise ValueError(f"TASK-009C pose ID is unknown: {pose_id}")
    supplied_manifest_hash = requested.get(
        "pose_library_manifest_config_sha256", manifest["config_sha256"]
    )
    if supplied_manifest_hash != manifest["config_sha256"]:
        raise ValueError("TASK-009C requested pose manifest hash mismatch")
    pose = np.asarray(requested["pose_world_xyzw"], dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("TASK-009C requested pose must be a finite seven-vector")
    frozen = np.asarray(allowed[pose_id]["pose_world_xyzw"], dtype=np.float64)
    if not np.allclose(pose, frozen, atol=1.0e-12, rtol=0.0):
        raise ValueError("TASK-009C requested coordinates differ from the frozen pose library")
    return pose_id, pose


class Task009BTrainingEnv(ManagerBasedRLEnv):
    """Add the contracted one-second HOLD stabilization to every explicit reset.

    The ten 10 Hz HOLD cycles advance physics and acquire ten fresh strategy
    images, but their episode-length cost is cleared before the first Actor
    action.  ``common_step_counter`` remains monotonic for frame auditing.
    """

    def reset(self, seed=None, env_ids=None, options=None):
        # Camera.reset() restarts its episode-local frame counter.  Discard the
        # prior boundary association before the base reset computes its first
        # observation.
        if hasattr(self, "_task009b_policy_rgb_sync"):
            del self._task009b_policy_rgb_sync
        if hasattr(self, "_task009b_policy_rgb_sync_latest"):
            del self._task009b_policy_rgb_sync_latest
        task009c_request = None if options is None else options.get(TASK009C_OPTION_KEY)
        observation, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        task009c_info = None
        if task009c_request is not None:
            if env_ids is not None:
                raise ValueError("TASK-009C single-environment pose reset does not accept env_ids")
            requested_config_path = Path(
                task009c_request.get("config_path", TASK009C_CONFIG_PATH)
            ).resolve()
            if TASK009C_CONFIG_PATH.parent.resolve() not in requested_config_path.parents:
                raise ValueError("TASK-009C requested configuration must be under configs/task009c")
            config, manifest, allowed = _load_task009c_pose_records(requested_config_path)
            supplied_config_hash = task009c_request.get("config_sha256")
            if supplied_config_hash is not None and supplied_config_hash != config["config_sha256"]:
                raise ValueError("TASK-009C requested configuration hash mismatch")
            pose_id, requested_pose = _validate_task009c_pose_record(
                task009c_request,
                config=config,
                manifest=manifest,
                allowed=allowed,
            )
            capsule = self.scene["capsule"]
            action_term = self.action_manager.get_term("parameterized_force")
            action_term.reset()
            capsule.permanent_wrench_composer.reset()
            pose_tensor = torch.as_tensor(
                requested_pose, device=self.device, dtype=torch.float32
            ).reshape(1, 7)
            capsule.write_root_pose_to_sim_index(root_pose=pose_tensor)
            capsule.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=self.device, dtype=torch.float32)
            )
            self.sim.forward()
            self.scene.update(0.0)
            restored_pose = capsule.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
            position_error = float(np.linalg.norm(restored_pose[:3] - requested_pose[:3]))
            quaternion_alignment = abs(float(np.dot(restored_pose[3:], requested_pose[3:])))
            if position_error > 1.0e-5 or quaternion_alignment < 1.0 - 1.0e-5:
                raise RuntimeError(
                    "TASK-009C pose write verification failed: "
                    f"position_error={position_error}, alignment={quaternion_alignment}"
                )
            task009c_info = {
                "pose_id": pose_id,
                "split": "validation",
                "pose_library_manifest_config_sha256": manifest["config_sha256"],
                "requested_pose_world_xyzw": requested_pose.tolist(),
                "write_position_error_m": position_error,
                "write_quaternion_absolute_alignment": quaternion_alignment,
            }
        hold = torch.full(
            (self.num_envs, 2),
            0.5,
            device=self.device,
            dtype=torch.float32,
        )
        hold[:, 0] = float(ParameterizedForceMode.HOLD)
        trace = []
        for cycle in range(RESET_HOLD_CYCLES):
            start_time = float(self.common_step_counter) * float(self.step_dt)
            observation, _, terminated, truncated, extras = super().step(hold)
            if bool(torch.any(terminated).item()) or bool(torch.any(truncated).item()):
                raise RuntimeError("formal reset HOLD stabilization unexpectedly terminated")
            rgb = observation["policy"]["rgb"]
            sync = dict(getattr(self, "_task009b_policy_rgb_sync_latest", {}))
            trace.append(
                {
                    "cycle": cycle,
                    "physics_substeps": int(self.cfg.decimation),
                    "start_sim_time_s": start_time,
                    "end_sim_time_s": float(self.common_step_counter) * float(self.step_dt),
                    "actor_rgb_frame": int(sync.get("frame", -1)),
                    "rgb_content_sha256": _stable_rgb_digest(rgb),
                    "rgb_finite": bool(torch.isfinite(rgb).all().item()),
                    "forced_capture": bool(sync.get("forced_capture", False)),
                }
            )
        self.episode_length_buf.zero_()
        extras["task009b_reset_stabilization"] = trace
        self._task009b_last_reset_stabilization = trace
        if task009c_info is not None:
            capsule = self.scene["capsule"]
            stable_pose = capsule.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
            stable_velocity = (
                capsule.data.root_com_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
            )
            task009c_info.update(
                {
                    "hold_cycles": trace,
                    "stable_pose_world_xyzw": stable_pose.tolist(),
                    "stable_velocity_world": stable_velocity.tolist(),
                    "final_rgb_content_sha256": _stable_rgb_digest(observation["policy"]["rgb"]),
                    "episode_length_buf": int(self.episode_length_buf[0].item()),
                }
            )
            extras[TASK009C_OPTION_KEY] = task009c_info
        return observation, extras
