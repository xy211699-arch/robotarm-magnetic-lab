"""Hashed binary feature-bank persistence for TASK-010 visual-dependence runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


FEATURE_STEPS = 1200
FEATURE_DIM = 512
MANIFEST_NAME = "manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "pose_id",
        "training_seed",
        "checkpoint_update",
        "checkpoint_sha256",
        "base_config_sha256",
        "visual_dependence_config_sha256",
        "feature_steps",
        "feature_dim",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"feature-bank metadata missing fields: {sorted(missing)}")
    unknown = set(metadata) - required
    if unknown:
        raise ValueError(f"feature-bank metadata unknown fields: {sorted(unknown)}")
    if int(metadata["feature_steps"]) != FEATURE_STEPS:
        raise ValueError("feature-bank metadata feature_steps must be 1200")
    if int(metadata["feature_dim"]) != FEATURE_DIM:
        raise ValueError("feature-bank metadata feature_dim must be 512")
    return {str(key): value for key, value in metadata.items()}


def _validated_features(features: torch.Tensor) -> torch.Tensor:
    if features.shape != (FEATURE_STEPS, FEATURE_DIM):
        raise ValueError(
            f"feature-bank tensor must be [1200, 512], got {tuple(features.shape)}"
        )
    value = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(value).all().item():
        raise ValueError("feature-bank tensor must be finite")
    return value


def _pose_path(root: Path, pose_id: str) -> Path:
    safe = str(pose_id)
    if not safe or "/" in safe or "\\" in safe:
        raise ValueError("pose_id must be a simple filename-safe identifier")
    return Path(root) / f"{safe}.pt"


def _read_manifest(root: Path) -> dict[str, Any]:
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        return {"schema": "robotarm_magnetic_lab.task010_feature_bank_manifest", "entries": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("feature-bank manifest must be an object")
    return raw


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    path = Path(root) / MANIFEST_NAME
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def manifest_sha256(root: Path) -> str:
    return file_sha256(Path(root) / MANIFEST_NAME)


def save_pose_feature_sequence(
    root: Path,
    metadata: Mapping[str, Any],
    features: torch.Tensor,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base_metadata = _validated_metadata(metadata)
    pose_id = str(base_metadata["pose_id"])
    values = _validated_features(features)
    payload = {
        "metadata": base_metadata,
        "features": values,
    }
    destination = _pose_path(root, pose_id)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    file_digest = file_sha256(temporary)
    os.replace(temporary, destination)
    manifest = _read_manifest(root)
    manifest.setdefault(
        "schema", "robotarm_magnetic_lab.task010_feature_bank_manifest"
    )
    entries = manifest.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError("feature-bank manifest entries must be an object")
    entries[pose_id] = {
        "pose_id": pose_id,
        "path": destination.name,
        "file_sha256": file_digest,
        "metadata": base_metadata,
    }
    _write_manifest(root, manifest)
    return destination


def load_pose_feature_sequence(
    root: Path,
    pose_id: str,
    expected_metadata: Mapping[str, Any],
) -> torch.Tensor:
    root = Path(root)
    expected = _validated_metadata(expected_metadata)
    if str(expected["pose_id"]) != str(pose_id):
        raise ValueError("expected feature-bank pose_id does not match requested pose")
    manifest = _read_manifest(root)
    entries = manifest.get("entries")
    if not isinstance(entries, dict) or str(pose_id) not in entries:
        raise ValueError(f"feature-bank entry is missing: {pose_id}")
    entry = entries[str(pose_id)]
    for key, value in expected.items():
        if entry["metadata"].get(key) != value:
            raise ValueError(f"feature-bank metadata mismatch for {key}")
    destination = Path(root) / entry["path"]
    if not destination.is_file():
        raise ValueError(f"feature-bank file is missing: {destination}")
    if file_sha256(destination) != entry["file_sha256"]:
        raise ValueError("feature-bank file hash mismatch")
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    if payload["metadata"] != expected:
        raise ValueError("feature-bank payload metadata mismatch")
    return _validated_features(payload["features"]).clone()
