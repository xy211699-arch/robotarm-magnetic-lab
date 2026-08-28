"""Reproducible, split-safe pose batches for TASK-009D0."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from robotarm_magnetic_lab.coverage.entry_pose_library import (
    POSE_LIBRARY_MANIFEST_SCHEMA,
    manifest_hash,
    read_jsonl,
    stable_record_is_valid,
)

from .task009d0_config import validate_task009d0_repository_inputs


AUTHORIZED_SPLITS = frozenset(("train", "validation", "test"))


@dataclass(frozen=True)
class PoseBatch:
    env_ids: np.ndarray
    pose_ids: Sequence[str]
    poses_world_xyzw: np.ndarray
    episode_indices: np.ndarray
    rng_seeds: np.ndarray


def derived_env_episode_seed(training_seed: int, env_id: int, episode_index: int) -> int:
    for name, value in (("env_id", env_id), ("episode_index", episode_index)):
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    sequence = np.random.SeedSequence(
        [int(training_seed), int(env_id), int(episode_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class Task009D0PoseBatchSampler:
    """Resolve only hash-verified poses belonging to one authorized split."""

    def __init__(
        self,
        records: Sequence[dict],
        *,
        authorized_split: str,
        training_seed: int,
        expected_split_counts: dict[str, int],
    ) -> None:
        if authorized_split not in AUTHORIZED_SPLITS:
            raise ValueError(f"unknown authorized pose split {authorized_split!r}")
        self.authorized_split = authorized_split
        self.training_seed = int(training_seed)
        self._by_id: dict[str, dict] = {}
        self._by_split: dict[str, list[dict]] = {name: [] for name in AUTHORIZED_SPLITS}
        for record in records:
            pose_id = str(record.get("pose_id", ""))
            split = str(record.get("split", ""))
            if split not in AUTHORIZED_SPLITS or not stable_record_is_valid(record):
                raise RuntimeError(f"TASK-009D0 invalid frozen pose record: {pose_id!r}")
            if pose_id in self._by_id:
                raise RuntimeError(f"TASK-009D0 duplicate pose ID: {pose_id}")
            if not pose_id.startswith(f"{split}-"):
                raise RuntimeError(f"TASK-009D0 pose ID/split mismatch: {pose_id}")
            self._by_id[pose_id] = record
            self._by_split[split].append(record)
        actual_counts = {key: len(value) for key, value in self._by_split.items()}
        if actual_counts != {key: int(value) for key, value in expected_split_counts.items()}:
            raise RuntimeError(
                f"TASK-009D0 pose split counts mismatch: {actual_counts}"
            )
        for values in self._by_split.values():
            values.sort(key=lambda item: item["pose_id"])

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        authorized_split: str,
        training_seed: int,
        repository_root: Path,
    ) -> "Task009D0PoseBatchSampler":
        inputs = validate_task009d0_repository_inputs(
            config, repository_root=repository_root
        )
        manifest = json.loads(inputs["pose_manifest"].read_text(encoding="utf-8"))
        if manifest.get("schema") != POSE_LIBRARY_MANIFEST_SCHEMA:
            raise RuntimeError("TASK-009D0 pose manifest schema mismatch")
        payload = {key: value for key, value in manifest.items() if key != "config_sha256"}
        if manifest_hash(payload) != manifest.get("config_sha256"):
            raise RuntimeError("TASK-009D0 pose manifest deterministic hash mismatch")
        pose_cfg = config["pose_library"]
        if manifest["config_sha256"] != pose_cfg["manifest_config_sha256"]:
            raise RuntimeError("TASK-009D0 pose manifest config hash mismatch")
        if manifest.get("split_counts") != pose_cfg["split_counts"]:
            raise RuntimeError("TASK-009D0 pose manifest split counts mismatch")
        records = read_jsonl(inputs["pose_data"])
        if len(records) != sum(int(value) for value in pose_cfg["split_counts"].values()):
            raise RuntimeError("TASK-009D0 pose record count mismatch")
        manifest_ids = {
            split: tuple(sorted(str(value) for value in manifest["split_pose_ids"][split]))
            for split in AUTHORIZED_SPLITS
        }
        record_ids = {
            split: tuple(sorted(str(item["pose_id"]) for item in records if item["split"] == split))
            for split in AUTHORIZED_SPLITS
        }
        if record_ids != manifest_ids:
            raise RuntimeError("TASK-009D0 pose ID sets differ from the Git manifest")
        return cls(
            records,
            authorized_split=authorized_split,
            training_seed=training_seed,
            expected_split_counts=pose_cfg["split_counts"],
        )

    @staticmethod
    def _validated_rows(env_ids: np.ndarray, episode_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        env = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        episodes = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
        if len(env) != len(episodes):
            raise ValueError("pose batch requires one episode index per environment row")
        if len(np.unique(env)) != len(env):
            raise ValueError("pose batch environment rows must be unique")
        if np.any(env < 0) or np.any(episodes < 0):
            raise ValueError("pose batch environment and episode indices must be non-negative")
        return env, episodes

    def _batch(self, env: np.ndarray, episodes: np.ndarray, records: Sequence[dict]) -> PoseBatch:
        seeds = np.asarray(
            [
                derived_env_episode_seed(self.training_seed, int(row), int(episode))
                for row, episode in zip(env, episodes, strict=True)
            ],
            dtype=np.uint32,
        )
        poses = np.asarray([record["pose_world_xyzw"] for record in records], dtype=np.float64)
        if poses.shape != (len(env), 7) or not np.isfinite(poses).all():
            raise RuntimeError("TASK-009D0 resolved pose batch is non-finite or malformed")
        return PoseBatch(
            env_ids=env.copy(),
            pose_ids=tuple(str(record["pose_id"]) for record in records),
            poses_world_xyzw=poses,
            episode_indices=episodes.copy(),
            rng_seeds=seeds,
        )

    def sample(self, env_ids: np.ndarray, episode_indices: np.ndarray) -> PoseBatch:
        if self.authorized_split != "train":
            raise ValueError("random sampling is available only for the training split")
        env, episodes = self._validated_rows(env_ids, episode_indices)
        records = self._by_split["train"]
        chosen = [records[derived_env_episode_seed(self.training_seed, int(row), int(ep)) % len(records)] for row, ep in zip(env, episodes, strict=True)]
        return self._batch(env, episodes, chosen)

    def resolve_explicit(self, env_ids: np.ndarray, pose_ids: Sequence[str]) -> PoseBatch:
        env = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        requested = tuple(str(value) for value in pose_ids)
        if len(env) != len(requested):
            raise ValueError("explicit pose batch requires one pose ID per environment row")
        if len(np.unique(env)) != len(env):
            raise ValueError("explicit pose batch environment rows must be unique")
        if self.authorized_split == "train":
            raise ValueError("training sampler cannot access explicit pose IDs")
        selected: list[dict] = []
        for pose_id in requested:
            record = self._by_id.get(pose_id)
            if record is None or record["split"] != self.authorized_split:
                raise ValueError(
                    f"explicit pose split mismatch for {pose_id!r}: expected {self.authorized_split}"
                )
            selected.append(record)
        episodes = np.zeros(len(env), dtype=np.int64)
        return self._batch(env, episodes, selected)
