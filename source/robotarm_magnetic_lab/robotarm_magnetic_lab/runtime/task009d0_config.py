"""Validated configuration loading for TASK-009D0 vector infrastructure."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from robotarm_magnetic_lab.coverage.entry_pose_library import file_sha256, manifest_hash


TASK009D0_SCHEMA = "robotarm_magnetic_lab.task009d0_vectorized_training"
TASK009D0_FROZEN_SCHEMA = "robotarm_magnetic_lab.task009d0_vectorized_training_frozen"
TASK009D0_VERSION = 1
TASK009D0_TASK_ID = "Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0"
TASK009D0_CODE_BASELINE = "7c4c5a18780b980ad3882ce75f1d64733fc3080d"
TASK009D0_CONFIG_PATH = (
    Path(__file__).resolve().parents[4]
    / "configs/task009d0/vectorized_training_candidates_v1.json"
)

_CANDIDATE_KEYS = {
    "schema",
    "version",
    "config_sha256",
    "task_id",
    "exact_code_baseline",
    "planning_head",
    "num_env_candidates",
    "env_spacing_m",
    "clocks",
    "camera",
    "force_ratio_mg",
    "pose_library",
    "coverage",
    "unreachable_region",
    "episode",
    "benchmark",
    "training_seed",
    "artifact_root",
}
_FROZEN_EXTRA_KEYS = {"selected_num_envs", "selection"}


def _require_exact_keys(record: dict[str, Any], *, frozen: bool) -> None:
    expected = _CANDIDATE_KEYS | (_FROZEN_EXTRA_KEYS if frozen else set())
    unknown = sorted(set(record) - expected)
    missing = sorted(expected - set(record))
    if unknown:
        raise ValueError(f"TASK-009D0 config has unknown top-level keys: {unknown}")
    if missing:
        raise ValueError(f"TASK-009D0 config is missing top-level keys: {missing}")


def _finite(value: Any, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def load_task009d0_config(path: Path, *, frozen: bool = False) -> dict[str, Any]:
    """Load and fully validate the versioned TASK-009D0 contract."""
    source = Path(path)
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("TASK-009D0 config must be a JSON object")
    _require_exact_keys(record, frozen=frozen)
    expected_schema = TASK009D0_FROZEN_SCHEMA if frozen else TASK009D0_SCHEMA
    if record["schema"] != expected_schema or int(record["version"]) != TASK009D0_VERSION:
        raise ValueError("TASK-009D0 config schema/version mismatch")
    payload = {key: value for key, value in record.items() if key != "config_sha256"}
    if manifest_hash(payload) != record["config_sha256"]:
        raise ValueError("TASK-009D0 config deterministic hash mismatch")
    if record["task_id"] != TASK009D0_TASK_ID:
        raise ValueError("TASK-009D0 task ID mismatch")
    if record["exact_code_baseline"] != TASK009D0_CODE_BASELINE:
        raise ValueError("TASK-009D0 exact code baseline mismatch")
    if record["num_env_candidates"] != [1, 2, 4, 8]:
        raise ValueError("TASK-009D0 candidate environments must be [1, 2, 4, 8]")
    if _finite(record["env_spacing_m"], "env_spacing_m") != 4.0:
        raise ValueError("TASK-009D0 environment spacing must be 4.0 m")
    if record["clocks"] != {
        "physics_hz": 240,
        "control_hz": 10,
        "physics_steps_per_action": 24,
    }:
        raise ValueError("TASK-009D0 clocks must be 240/10 Hz with 24 substeps")
    if record["camera"] != {
        "width": 1280,
        "height": 720,
        "hz": 10,
        "fov_deg": 120.0,
    }:
        raise ValueError("TASK-009D0 camera contract mismatch")
    expected_force = {
        "MOVE": {"minimum": 0.7, "maximum": 1.4},
        "VIEW": {"minimum": 0.2, "maximum": 0.5},
        "UP": {"minimum": 0.8, "maximum": 1.05},
        "HOLD": {"minimum": 0.0, "maximum": 0.0},
    }
    if record["force_ratio_mg"] != expected_force:
        raise ValueError("TASK-009D0 force ranges differ from the frozen controller")
    episode = record["episode"]
    expected_episode = {
        "duration_s": 120.0,
        "formal_steps": 1200,
        "hold_steps": 10,
        "coverage_points": 1201,
        "formal_physics_substeps": 28800,
        "inter_episode_hold_substeps": 240,
        "thresholds": [0.8, 0.9, 0.95],
        "long_audit_duration_s": 300.0,
    }
    if episode != expected_episode:
        raise ValueError("TASK-009D0 episode contract mismatch")
    benchmark = record["benchmark"]
    if benchmark != {
        "warmup_steps": 50,
        "measured_steps": 300,
        "repeats": 3,
        "minimum_free_memory_fraction": 0.2,
        "near_tie_fraction": 0.1,
    }:
        raise ValueError("TASK-009D0 benchmark contract mismatch")
    coverage = record["coverage"]
    if (
        coverage["max_distance_m"] != 0.07
        or coverage["target_vertex_count"] != 24529
        or coverage["target_triangle_count"] != 49047
        or coverage["reachable_positive_weight_vertex_count"] != 17055
    ):
        raise ValueError("TASK-009D0 coverage contract mismatch")
    if frozen:
        selected = int(record["selected_num_envs"])
        if selected not in record["num_env_candidates"]:
            raise ValueError("selected_num_envs is not an authorized candidate")
    return record


def validate_task009d0_repository_inputs(
    config: dict[str, Any], *, repository_root: Path
) -> dict[str, Path]:
    """Resolve and hash-check immutable Git/external inputs without loading simulation."""
    root = Path(repository_root).resolve()
    resolved: dict[str, Path] = {}
    for key, section, path_key, hash_key in (
        ("pose_manifest", "pose_library", "manifest_path", "manifest_file_sha256"),
        ("coverage_manifest", "coverage", "manifest_path", "manifest_file_sha256"),
        ("unreachable_region", "unreachable_region", "path", "file_sha256"),
    ):
        source = root / config[section][path_key]
        if not source.is_file() or file_sha256(source) != config[section][hash_key]:
            raise RuntimeError(f"TASK-009D0 {key} is missing or has the wrong hash")
        resolved[key] = source
    data = Path(config["pose_library"]["data_path"])
    if (
        not data.is_file()
        or data.stat().st_size != int(config["pose_library"]["data_bytes"])
        or file_sha256(data) != config["pose_library"]["data_sha256"]
    ):
        raise RuntimeError("TASK-009D0 external pose library is missing or has the wrong hash")
    resolved["pose_data"] = data
    return resolved
