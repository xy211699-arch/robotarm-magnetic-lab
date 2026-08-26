"""Versioned P0 metadata/JSONL records and artifact inventories."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
DEPLOYABLE_FIELDS = frozenset(
    {"joint_position_rad", "joint_velocity_rad_s", "external_magnet_pose"}
)
PRIVILEGED_TOKENS = (
    "capsule",
    "coverage",
    "contact",
    "depth",
    "stomach",
    "ray",
    "wrench",
    "force",
    "torque",
)


def deployable_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce the deployable-observation whitelist at a serialization boundary."""
    unknown = set(values) - DEPLOYABLE_FIELDS
    privileged = sorted(
        key for key in unknown if any(token in key.lower() for token in PRIVILEGED_TOKENS)
    )
    if privileged:
        raise ValueError(f"privileged evaluator fields are forbidden: {privileged}")
    if unknown:
        raise ValueError(f"fields are outside the deployable whitelist: {sorted(unknown)}")
    return {key: values[key] for key in sorted(values)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(paths: Iterable[Path], root: Path | None = None) -> list[dict[str, Any]]:
    result = []
    for item in sorted((Path(path) for path in paths), key=lambda path: str(path)):
        if not item.is_file():
            raise ValueError(f"artifact is not a file: {item}")
        display = item.relative_to(root) if root is not None else item
        result.append(
            {"path": str(display), "byte_size": item.stat().st_size, "sha256": _sha256(item)}
        )
    return result


class CoverageRecordWriter:
    """Write bounded records into a partial directory, then rename atomically."""

    def __init__(self, final_directory: Path, metadata: Mapping[str, Any]) -> None:
        self.final_directory = Path(final_directory)
        if self.final_directory.exists():
            raise FileExistsError(self.final_directory)
        self.partial_directory = self.final_directory.parent / (
            f".{self.final_directory.name}.partial-{uuid.uuid4().hex}"
        )
        self.partial_directory.mkdir(parents=True, exist_ok=False)
        self._action_events: set[tuple[int, str]] = set()
        self._frame_ids: set[str] = set()
        self._finalized = False
        versioned = {"schema_version": SCHEMA_VERSION, **dict(metadata)}
        self._write_json(self.partial_directory / "metadata.json", versioned)
        (self.partial_directory / "actions.jsonl").touch()
        (self.partial_directory / "frames.jsonl").touch()

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("record writer is already finalized")

    def append_action(self, record: Mapping[str, Any]) -> None:
        self._ensure_open()
        required = ("request_id", "event", "timestamp_s")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"action record is missing {missing}")
        key = (int(record["request_id"]), str(record["event"]))
        if key in self._action_events:
            raise ValueError(f"duplicate request_id/event pair: {key}")
        self._action_events.add(key)
        self._append_jsonl(self.partial_directory / "actions.jsonl", record)

    def append_frame(self, record: Mapping[str, Any]) -> None:
        self._ensure_open()
        if "frame_id" not in record:
            raise ValueError("frame record is missing frame_id")
        frame_key = json.dumps(record["frame_id"], sort_keys=True)
        if frame_key in self._frame_ids:
            raise ValueError(f"duplicate frame_id: {record['frame_id']}")
        if all(key in record for key in ("cumulative_count", "vertex_count", "coverage_fraction")):
            vertex_count = int(record["vertex_count"])
            if vertex_count <= 0:
                raise ValueError("vertex_count must be positive")
            if "cumulative_area_m2" in record and "total_area_m2" in record:
                total_area = float(record["total_area_m2"])
                if not math.isfinite(total_area) or total_area <= 0.0:
                    raise ValueError("total_area_m2 must be finite and positive")
                expected = float(record["cumulative_area_m2"]) / total_area
            else:
                expected = int(record["cumulative_count"]) / vertex_count
            if not math.isclose(float(record["coverage_fraction"]), expected, abs_tol=1.0e-12):
                raise ValueError("coverage_fraction is inconsistent with cumulative_count")
        self._frame_ids.add(frame_key)
        self._append_jsonl(self.partial_directory / "frames.jsonl", record)

    def finalize(self) -> Path:
        self._ensure_open()
        self.final_directory.parent.mkdir(parents=True, exist_ok=True)
        if self.final_directory.exists():
            raise FileExistsError(self.final_directory)
        os.replace(self.partial_directory, self.final_directory)
        self._finalized = True
        return self.final_directory

    def abort(self) -> None:
        if not self._finalized:
            shutil.rmtree(self.partial_directory, ignore_errors=True)
            self._finalized = True
