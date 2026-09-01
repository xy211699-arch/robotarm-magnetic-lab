from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stomach_coverage/query_random_baseline_20pose_status.py"
SPEC = importlib.util.spec_from_file_location("task009c_20pose_status", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_not_started(tmp_path: Path):
    assert MODULE.query_status(tmp_path)["state"] == "not_started"


def test_reports_verified_progress_and_completion(tmp_path: Path):
    run = tmp_path / "formal-run"
    run.mkdir()
    manifest = run / "run_manifest.jsonl"
    rows = [
        {"record_type": "run_start", "started_at_utc": "2026-09-01T00:00:00+00:00"},
        {"record_type": "episode", "status": "pass", "episode_id": "e1", "policy_id": "R1",
         "pose_id": "validation-0006", "C_final_reachable": 0.5},
        {"record_type": "run_complete", "status": "pass"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "latest_formal_manifest.json").write_text(json.dumps({
        "run_id": "formal-run", "manifest_path": str(manifest),
        "manifest_bytes": manifest.stat().st_size, "manifest_sha256": digest,
    }))
    result = MODULE.query_status(tmp_path)
    assert result["state"] == "completed"
    assert result["completed_episodes"] == 1
    assert result["per_policy_completed"]["R1"] == 1
    assert result["latest_final_reachable_coverage"] == 0.5


def test_pause_marker_changes_incomplete_run_state(tmp_path: Path):
    run = tmp_path / "formal-run"
    run.mkdir()
    manifest = run / "run_manifest.jsonl"
    manifest.write_text(json.dumps({"record_type": "run_start"}) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "latest_formal_manifest.json").write_text(json.dumps({
        "run_id": "formal-run", "manifest_path": str(manifest),
        "manifest_bytes": manifest.stat().st_size, "manifest_sha256": digest,
    }))
    (run / "pause_summary.json").write_text("{}")
    assert MODULE.query_status(tmp_path)["state"] == "paused"
