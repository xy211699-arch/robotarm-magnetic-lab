"""Frozen orchestration configuration for the TASK-010 visual-dependence study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from robotarm_magnetic_lab.runtime.task010_config import (
    TASK010_CONFIG_PATH,
    load_task010_config,
)


VISUAL_DEPENDENCE_SCHEMA = "robotarm_magnetic_lab.task010_visual_dependence"
VISUAL_DEPENDENCE_VERSION = 1
VISUAL_DEPENDENCE_CONFIG_PATH = (
    Path(__file__).resolve().parents[4]
    / "configs/task010/visual_dependence_v1.json"
)


@dataclass(frozen=True)
class BaseConfigReference:
    path: str
    sha256: str


@dataclass(frozen=True)
class VisualDependenceConfig:
    schema_version: int
    config_sha256: str
    base_config: BaseConfigReference
    formal_seeds: tuple[int, ...]
    validation_pose_ids: tuple[str, ...]
    donor_pose_by_target: Mapping[str, str]
    primary_update: int
    sensitivity_update: int
    episode_steps: int
    coverage_points: int
    training_conditions: tuple[str, ...]
    primary_conditions: tuple[str, ...]
    sensitivity_conditions: tuple[str, ...]
    bootstrap_seed: int
    bootstrap_replicates: int
    training_stall_after_s: int
    training_failure_after_s: int


_TOP_LEVEL = {field.name for field in fields(VisualDependenceConfig)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, BaseConfigReference):
        return {
            "path": value.path,
            "sha256": value.sha256,
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    return value


def canonical_visual_dependence_sha256(
    config: VisualDependenceConfig | Mapping[str, Any],
) -> str:
    payload = _canonical(config)
    payload.pop("config_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact(record: Mapping[str, Any]) -> None:
    unknown = sorted(set(record) - _TOP_LEVEL)
    missing = sorted(_TOP_LEVEL - set(record))
    if unknown:
        raise ValueError(f"unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"missing field: {missing[0]}")


def _finite_tree(value: Any, name: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _tuple_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must be a list of integers")
    return tuple(int(item) for item in value)


def _tuple_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(str(item) for item in value)


def _required_positive_int(record: Mapping[str, Any], name: str) -> int:
    value = record[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def load_visual_dependence_config(
    path: Path = VISUAL_DEPENDENCE_CONFIG_PATH,
) -> VisualDependenceConfig:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("TASK-010 visual-dependence config must be an object")
    _exact(raw)
    if raw["schema_version"] != VISUAL_DEPENDENCE_VERSION:
        raise ValueError("TASK-010 visual-dependence schema version mismatch")

    base_raw = raw["base_config"]
    if not isinstance(base_raw, dict):
        raise ValueError("base_config must be an object")
    unknown = set(base_raw) - {"path", "sha256"}
    missing = {"path", "sha256"} - set(base_raw)
    if unknown or missing:
        raise ValueError("base_config has invalid fields")
    base = BaseConfigReference(
        path=str(base_raw["path"]),
        sha256=str(base_raw["sha256"]),
    )
    base_path = source.parent / base.path
    if base.path != "cnn_gru_development_v1.json" or not base_path.is_file():
        raise ValueError("base_config.path must reference the frozen TASK-010 config")
    if file_sha256(base_path) != base.sha256:
        raise ValueError("base_config file hash mismatch")
    frozen_base = load_task010_config(base_path)

    seeds = _tuple_ints(raw["formal_seeds"], "formal_seeds")
    poses = _tuple_strings(raw["validation_pose_ids"], "validation_pose_ids")
    if seeds != (991001, 991002, 991003):
        raise ValueError("formal_seeds must be the frozen 991001/991002/991003 set")
    if poses != frozen_base.validation.pose_ids:
        raise ValueError("validation_pose_ids must equal the frozen base validation set")

    donor = MappingProxyType(
        dict(zip(poses, poses[1:] + poses[:1]))
    )
    if raw["donor_pose_by_target"] != dict(donor):
        raise ValueError("donor_pose_by_target must be the frozen cyclic derangement")

    training_conditions = _tuple_strings(
        raw["training_conditions"], "training_conditions"
    )
    primary_conditions = _tuple_strings(raw["primary_conditions"], "primary_conditions")
    sensitivity_conditions = _tuple_strings(
        raw["sensitivity_conditions"], "sensitivity_conditions"
    )
    expected = {
        "training_conditions": ("blind",),
        "primary_conditions": ("normal", "blind", "donor", "first_frame"),
        "sensitivity_conditions": ("normal", "blind"),
    }
    if (
        training_conditions != expected["training_conditions"]
        or primary_conditions != expected["primary_conditions"]
        or sensitivity_conditions != expected["sensitivity_conditions"]
    ):
        raise ValueError("visual-dependence condition matrix differs from the frozen design")

    cfg = VisualDependenceConfig(
        schema_version=int(raw["schema_version"]),
        config_sha256=str(raw["config_sha256"]),
        base_config=base,
        formal_seeds=seeds,
        validation_pose_ids=poses,
        donor_pose_by_target=donor,
        primary_update=_required_positive_int(raw, "primary_update"),
        sensitivity_update=_required_positive_int(raw, "sensitivity_update"),
        episode_steps=_required_positive_int(raw, "episode_steps"),
        coverage_points=_required_positive_int(raw, "coverage_points"),
        training_conditions=training_conditions,
        primary_conditions=primary_conditions,
        sensitivity_conditions=sensitivity_conditions,
        bootstrap_seed=_required_positive_int(raw, "bootstrap_seed"),
        bootstrap_replicates=_required_positive_int(raw, "bootstrap_replicates"),
        training_stall_after_s=_required_positive_int(raw, "training_stall_after_s"),
        training_failure_after_s=_required_positive_int(raw, "training_failure_after_s"),
    )
    if cfg.primary_update != 750 or cfg.sensitivity_update != 1000:
        raise ValueError("primary/sensitivity update contract mismatch")
    if cfg.episode_steps != 1200 or cfg.coverage_points != 1201:
        raise ValueError("episode/coverage point contract mismatch")
    if cfg.bootstrap_seed != 20260903 or cfg.bootstrap_replicates != 10000:
        raise ValueError("bootstrap contract mismatch")
    if cfg.training_stall_after_s != 300 or cfg.training_failure_after_s != 900:
        raise ValueError("stall/failure thresholds must remain 300/900 seconds")
    if cfg.config_sha256 != canonical_visual_dependence_sha256(cfg):
        raise ValueError("TASK-010 visual-dependence config hash mismatch")
    _finite_tree(_canonical(cfg))
    return cfg


def stamp_visual_dependence_config(path: Path) -> str:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("TASK-010 visual-dependence config must be an object")
    original_hash = raw.get("config_sha256")
    raw.pop("config_sha256", None)
    provisional = dict(raw)
    provisional["config_sha256"] = ""
    expected = canonical_visual_dependence_sha256(provisional)
    if original_hash is not None and original_hash != expected:
        raise ValueError("refusing to overwrite a non-canonical config_sha256")
    raw["config_sha256"] = expected
    temporary = source.with_name(source.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(raw, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, source)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Stamp the TASK-010 visual-dependence config")
    parser.add_argument("--stamp", type=Path, required=True)
    args = parser.parse_args()
    digest = stamp_visual_dependence_config(args.stamp)
    print(f"stamped={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
