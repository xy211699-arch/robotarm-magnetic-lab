"""Versioned and atomically finalized P0 record tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotarm_magnetic_lab.coverage.records import (
    CoverageRecordWriter,
    artifact_inventory,
    deployable_fields,
)


def test_versioned_append_only_records_and_atomic_finalize(tmp_path: Path):
    final = tmp_path / "session"
    writer = CoverageRecordWriter(final, metadata={"task_id": "TASK-001"})
    assert not final.exists()
    writer.append_action({"request_id": 1, "event": "REQUEST", "timestamp_s": 0.0})
    writer.append_action({"request_id": 1, "event": "DONE", "timestamp_s": 1.0})
    writer.append_frame(
        {
            "frame_id": 7,
            "candidate_count": 3,
            "visible_count": 2,
            "newly_covered_count": 2,
            "cumulative_count": 2,
            "vertex_count": 10,
            "coverage_fraction": 0.2,
            "timestamp_s": 1.0,
        }
    )
    with pytest.raises(ValueError, match="frame_id"):
        writer.append_frame({"frame_id": 7})
    writer.finalize()

    assert final.is_dir()
    metadata = json.loads((final / "metadata.json").read_text())
    assert metadata["schema_version"] == 1
    action_lines = [json.loads(line) for line in (final / "actions.jsonl").read_text().splitlines()]
    frame_lines = [json.loads(line) for line in (final / "frames.jsonl").read_text().splitlines()]
    assert [line["event"] for line in action_lines] == ["REQUEST", "DONE"]
    assert frame_lines[0]["cumulative_count"] == 2


def test_inconsistent_frame_counts_are_rejected(tmp_path: Path):
    writer = CoverageRecordWriter(tmp_path / "session", metadata={})
    with pytest.raises(ValueError, match="coverage_fraction"):
        writer.append_frame(
            {
                "frame_id": 1,
                "cumulative_count": 2,
                "vertex_count": 10,
                "coverage_fraction": 0.3,
            }
        )


def test_deployable_whitelist_rejects_privileged_fields():
    safe = deployable_fields(
        {
            "joint_position_rad": [0.0] * 9,
            "joint_velocity_rad_s": [0.0] * 9,
            "external_magnet_pose": [0.0] * 7,
        }
    )
    assert set(safe) == {"joint_position_rad", "joint_velocity_rad_s", "external_magnet_pose"}
    with pytest.raises(ValueError, match="privileged"):
        deployable_fields({"capsule_pose": [0.0] * 7})


def test_artifact_inventory_has_size_and_sha256(tmp_path: Path):
    artifact = tmp_path / "mask.npy"
    artifact.write_bytes(b"coverage-mask")
    item = artifact_inventory([artifact], root=tmp_path)[0]
    assert item["path"] == "mask.npy"
    assert item["byte_size"] == len(b"coverage-mask")
    assert len(item["sha256"]) == 64
