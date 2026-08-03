"""Atomic episode writer for behavior-cloning and VLA fine-tuning data."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from .schema import ACTION_DIM, DEPTH_SHAPE, POLICY_STATE_DIM, RGB_SHAPE


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


class EpisodeWriter:
    """Write one episode into a temporary directory, then commit atomically."""

    def __init__(
        self,
        dataset_root: str | Path,
        episode_id: str,
        interface_spec: dict[str, Any],
        interface_sha256: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.episode_id = episode_id
        self.final_dir = self.dataset_root / "episodes" / episode_id
        self.temp_dir = self.dataset_root / "episodes" / f".{episode_id}.incomplete"
        if self.final_dir.exists() or self.temp_dir.exists():
            raise FileExistsError(f"Episode already exists: {episode_id}")
        (self.temp_dir / "rgb").mkdir(parents=True)
        (self.temp_dir / "depth").mkdir()
        self._steps_path = self.temp_dir / "steps.jsonl"
        self._steps = self._steps_path.open("w", encoding="utf-8")
        self._count = 0
        self._camera_frame_count = 0
        self._last_camera_frame_id: int | None = None
        self._last_rgb_rel: Path | None = None
        self._last_depth_rel: Path | None = None
        self._closed = False
        self.interface_spec = interface_spec
        self.interface_sha256 = interface_sha256
        self.metadata = metadata or {}

    @staticmethod
    def _rgb_uint8(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.shape != RGB_SHAPE:
            raise ValueError(f"Expected RGB {RGB_SHAPE}, got {rgb.shape}")
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0)
        return rgb.astype(np.uint8, copy=False)

    def _depth_uint16(self, depth_m: np.ndarray) -> np.ndarray:
        depth_m = np.asarray(depth_m, dtype=np.float32)
        if depth_m.shape != DEPTH_SHAPE:
            raise ValueError(f"Expected depth {DEPTH_SHAPE}, got {depth_m.shape}")
        depth_cfg = self.interface_spec["inputs"]["depth"]
        scale = float(depth_cfg["scale_m_per_unit"])
        maximum = float(depth_cfg["maximum_m"])
        valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= maximum)
        encoded = np.zeros(DEPTH_SHAPE, dtype=np.uint16)
        encoded[valid] = np.clip(
            np.rint(depth_m[valid] / scale), 1, np.iinfo(np.uint16).max
        ).astype(np.uint16)
        return encoded

    def append(
        self,
        *,
        step: int,
        control_time_s: float,
        camera_frame_id: int,
        camera_timestamp_s: float,
        camera_is_new: bool,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        policy_state: np.ndarray,
        action_command: np.ndarray,
        action_applied_joint_target_rad: np.ndarray,
        teacher: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Append one 20 Hz control row referencing the latest 1 Hz frame."""
        if self._closed:
            raise RuntimeError("Episode writer is already closed")
        if step != self._count:
            raise ValueError(f"Expected contiguous step {self._count}, got {step}")
        state = np.asarray(policy_state, dtype=np.float32).reshape(-1)
        action = np.asarray(action_command, dtype=np.float32).reshape(-1)
        applied = np.asarray(action_applied_joint_target_rad, dtype=np.float32).reshape(-1)
        if state.shape != (POLICY_STATE_DIM,):
            raise ValueError(f"Expected policy state {(POLICY_STATE_DIM,)}, got {state.shape}")
        if action.shape != (ACTION_DIM,) or applied.shape != (ACTION_DIM,):
            raise ValueError("Command and applied joint target must both have shape (9,)")
        if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(applied).all():
            raise ValueError("Non-finite state/action encountered")
        if np.max(np.abs(action)) > 1.0001:
            raise ValueError("Normalized action is outside [-1, 1]")

        if camera_is_new:
            expected_frame_id = self._camera_frame_count
            if camera_frame_id != expected_frame_id:
                raise ValueError(
                    f"Expected new camera frame {expected_frame_id}, got {camera_frame_id}"
                )
            stem = f"{camera_frame_id:06d}"
            rgb_rel = Path("rgb") / f"{stem}.png"
            depth_rel = Path("depth") / f"{stem}.png"
            Image.fromarray(self._rgb_uint8(rgb), mode="RGB").save(
                self.temp_dir / rgb_rel, compress_level=3
            )
            Image.fromarray(self._depth_uint16(depth_m), mode="I;16").save(
                self.temp_dir / depth_rel, compress_level=3
            )
            self._camera_frame_count += 1
            self._last_camera_frame_id = camera_frame_id
            self._last_rgb_rel = rgb_rel
            self._last_depth_rel = depth_rel
        else:
            if self._last_camera_frame_id is None:
                raise ValueError("The first control row must contain a new camera frame")
            if camera_frame_id != self._last_camera_frame_id:
                raise ValueError("A stale row must reference the latest camera frame")
            rgb_rel = self._last_rgb_rel
            depth_rel = self._last_depth_rel
        assert rgb_rel is not None and depth_rel is not None
        record = {
            "schema_version": self.interface_spec["schema_version"],
            "episode_id": self.episode_id,
            "step": step,
            "control_time_s": float(control_time_s),
            "camera_frame_id": int(camera_frame_id),
            "camera_timestamp_s": float(camera_timestamp_s),
            "camera_is_new": bool(camera_is_new),
            "rgb_path": rgb_rel.as_posix(),
            "depth_path": depth_rel.as_posix(),
            "policy_state": state.tolist(),
            "action_command": action.tolist(),
            "action_applied_joint_target_rad": applied.tolist(),
            "teacher": teacher,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        self._steps.write(json.dumps(record, default=_jsonable, separators=(",", ":")) + "\n")
        self._steps.flush()
        self._count += 1

    def close(
        self,
        *,
        success: bool,
        termination_reason: str,
        extra_summary: dict[str, Any] | None = None,
    ) -> Path:
        """Commit the completed episode and append it to the dataset index."""
        if self._closed:
            return self.final_dir
        self._steps.close()
        episode_meta = {
            "schema_version": self.interface_spec["schema_version"],
            "interface_sha256": self.interface_sha256,
            "episode_id": self.episode_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "num_steps": self._count,
            "num_camera_frames": self._camera_frame_count,
            "success": bool(success),
            "termination_reason": termination_reason,
            "metadata": self.metadata,
            "summary": extra_summary or {},
        }
        (self.temp_dir / "episode.json").write_text(
            json.dumps(episode_meta, indent=2, default=_jsonable) + "\n",
            encoding="utf-8",
        )
        os.replace(self.temp_dir, self.final_dir)
        index_record = {
            "episode_id": self.episode_id,
            "path": f"episodes/{self.episode_id}",
            "num_steps": self._count,
            "num_camera_frames": self._camera_frame_count,
            "success": bool(success),
            "termination_reason": termination_reason,
        }
        with (self.dataset_root / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(index_record, separators=(",", ":")) + "\n")
        self._closed = True
        return self.final_dir

    def abort(self) -> None:
        """Remove an incomplete episode without touching committed data."""
        if not self._closed:
            self._steps.close()
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self._closed = True
