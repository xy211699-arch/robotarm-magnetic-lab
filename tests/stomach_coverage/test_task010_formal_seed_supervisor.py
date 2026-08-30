from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stomach_coverage/task010_formal_seed_supervisor.py"
DRIVER = ROOT / "tests/fixtures/task010_formal_fake_stage.py"
CONFIG = ROOT / "configs/task010/cnn_gru_development_v1.json"
SPEC = importlib.util.spec_from_file_location("task010_formal_seed_supervisor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _environment():
    return {**os.environ, "TASK010_FORMAL_TEST_MODE": "1"}


def _start(tmp_path: Path, plan: dict):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "runs"
    begin = time.monotonic()
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "start",
            "--config", str(CONFIG),
            "--output-root", str(output),
            "--test-driver", str(DRIVER),
            "--test-plan", str(plan_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_environment(),
    )
    return time.monotonic() - begin, output, Path(json.loads(result.stdout)["run_dir"])


def _wait(run_dir: Path, states=("completed", "paused_on_error"), timeout: float = 15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = json.loads((run_dir / "status.json").read_text())
        if state["state"] in states:
            return state
        time.sleep(0.05)
    raise AssertionError(f"coordinator did not reach {states}")


def _invocations(run_dir: Path):
    rows = [json.loads(line) for line in (run_dir / "fake_invocations.jsonl").read_text().splitlines()]
    return [(row["seed"], row["stage"], row["attempt"]) for row in rows if row["event"] == "start"]


def test_start_returns_and_runs_three_seeds_in_fixed_nonconcurrent_order(tmp_path: Path):
    elapsed, _, run_dir = _start(tmp_path, {"delay_s": 0.15})
    assert elapsed < 1.0
    state = _wait(run_dir)
    assert state["state"] == "completed"
    assert _invocations(run_dir) == [
        (991001, "training", 1), (991001, "validating", 1),
        (991002, "training", 1), (991002, "validating", 1),
        (991003, "training", 1), (991003, "validating", 1),
    ]
    assert not any(
        json.loads(line).get("event") == "overlap"
        for line in (run_dir / "fake_invocations.jsonl").read_text().splitlines()
    )
    assert all(item["state"] == "validated" for item in state["seeds"].values())


def test_raw_curves_seed_means_and_formal_mean_std_are_recomputable(tmp_path: Path):
    _, _, run_dir = _start(tmp_path, {})
    state = _wait(run_dir)
    aggregate = Path(state["aggregate_csv"])
    with aggregate.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1201
    assert tuple(rows[0]) == (
        "time_s", "seed_991001_mean_coverage", "seed_991002_mean_coverage",
        "seed_991003_mean_coverage", "formal_mean_coverage", "formal_std_coverage",
    )
    for seed in MODULE.FORMAL_SEEDS:
        validation = run_dir / "seeds" / f"seed_{seed}" / "validation"
        trajectories = [json.loads(line) for line in (validation / "coverage_trajectories.jsonl").read_text().splitlines()]
        assert len(trajectories) == 20
        assert {len(item["coverage_fraction"]) for item in trajectories} == {1201}
        raw_mean = sum(item["coverage_fraction"][777] for item in trajectories) / 20.0
        assert float(rows[777][f"seed_{seed}_mean_coverage"]) == pytest.approx(raw_mean)
    values = [float(rows[777][f"seed_{seed}_mean_coverage"]) for seed in MODULE.FORMAL_SEEDS]
    mean = sum(values) / 3.0
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / 3.0)
    assert float(rows[777]["formal_mean_coverage"]) == pytest.approx(mean)
    assert float(rows[777]["formal_std_coverage"]) == pytest.approx(std)


def test_failure_pauses_without_starting_later_seed_and_continue_retries_failed_stage(tmp_path: Path):
    _, output, run_dir = _start(tmp_path, {"fail_once": {"991002:validating": 23}})
    state = _wait(run_dir)
    assert state["state"] == "paused_on_error"
    assert state["current_seed"] == 991002 and state["current_stage"] == "validating"
    assert not any(seed == 991003 for seed, _, _ in _invocations(run_dir))
    deadline = time.monotonic() + 3.0
    while MODULE._pid_alive(state["worker_pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "continue", "--run-dir", str(run_dir), "--output-root", str(output)],
        check=True, text=True, capture_output=True, env=_environment(),
    )
    assert json.loads(result.stdout)["retry_stage"] == "validating"
    completed = _wait(run_dir, states=("completed",))
    assert completed["state"] == "completed"
    invocations = _invocations(run_dir)
    assert (991002, "validating", 2) in invocations
    assert invocations[-2:] == [(991003, "training", 1), (991003, "validating", 1)]


@pytest.mark.parametrize("invalid", ("points", "pose_count", "nonfinite"))
def test_invalid_validation_evidence_pauses_before_next_seed(tmp_path: Path, invalid: str):
    _, _, run_dir = _start(tmp_path, {"invalid": {"991001:validating": invalid}})
    state = _wait(run_dir)
    assert state["state"] == "paused_on_error"
    assert state["current_seed"] == 991001
    assert state["current_stage"] == "validating"
    assert not any(seed != 991001 for seed, _, _ in _invocations(run_dir))


def test_status_is_read_only_and_reports_required_progress_fields(tmp_path: Path):
    _, output, run_dir = _start(tmp_path, {})
    _wait(run_dir)
    status_path = run_dir / "status.json"
    before = status_path.read_bytes()
    before_mtime = status_path.stat().st_mtime_ns
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "status", "--run-dir", str(run_dir), "--output-root", str(output)],
        check=True, text=True, capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert {"current_seed", "current_stage", "training_update", "validation_pose_progress", "latest_checkpoint", "runtime_s", "heartbeat_age_s", "error_summary"} <= set(payload)
    assert payload["training_update"] == 1000
    assert payload["validation_pose_progress"] == 20
    assert payload["latest_checkpoint"].endswith("update_1000.pt")
    assert status_path.read_bytes() == before
    assert status_path.stat().st_mtime_ns == before_mtime


def test_duplicate_start_is_rejected_while_coordinator_is_active(tmp_path: Path):
    _, output, run_dir = _start(tmp_path, {"delay_s": 0.5})
    duplicate = subprocess.run(
        [
            sys.executable, str(SCRIPT), "start",
            "--config", str(CONFIG),
            "--output-root", str(output),
            "--test-driver", str(DRIVER),
            "--test-plan", str(tmp_path / "plan.json"),
        ],
        text=True,
        capture_output=True,
        env=_environment(),
    )
    assert duplicate.returncode != 0
    assert "already active" in duplicate.stderr
    assert _wait(run_dir, timeout=20.0)["state"] == "completed"


def test_stale_or_dead_worker_is_reported_as_paused_without_mutating_status(tmp_path: Path):
    run_dir = tmp_path / "stale"
    run_dir.mkdir()
    state = MODULE._initial_state(run_dir, "stale-run")
    state["worker_pid"] = 999_999_999
    state["heartbeat_epoch_s"] = time.time() - MODULE.STALE_AFTER_S - 1.0
    MODULE._atomic_json(run_dir / "status.json", state)
    before = (run_dir / "status.json").read_bytes()
    payload = MODULE._status_payload(run_dir)
    assert payload["state"] == "paused_on_error"
    assert "stale" in payload["error_summary"]
    assert (run_dir / "status.json").read_bytes() == before


def test_continue_can_acknowledge_a_stale_worker_before_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "stale-continue"
    run_dir.mkdir()
    state = MODULE._initial_state(run_dir, "stale-run")
    state.update(worker_pid=999_999_999, heartbeat_epoch_s=time.time() - 100.0)
    MODULE._atomic_json(run_dir / "status.json", state)
    MODULE._atomic_json(run_dir / "manifest.json", {"run_id": "stale-run"})
    captured = {}

    def fake_spawn(path, payload, *, continuation):
        captured.update(path=path, state=payload["state"], continuation=continuation)
        return 12345

    monkeypatch.setattr(MODULE, "_spawn_worker", fake_spawn)
    result = MODULE._continue(SimpleNamespace(run_dir=run_dir, output_root=tmp_path))
    assert result["worker_pid"] == 12345
    assert captured == {"path": run_dir, "state": "paused_on_error", "continuation": True}
    persisted = json.loads((run_dir / "status.json").read_text())
    assert persisted["state"] == "paused_on_error"
    assert "stale" in persisted["error_summary"]


def test_production_commands_keep_frozen_contract():
    record = {
        "training_dir": "/tmp/formal/seed_991001/training",
        "validation_dir": "/tmp/formal/seed_991001/validation",
    }
    manifest = {"config": {"path": str(CONFIG)}, "test_driver": None}
    training = MODULE._training_command(manifest, 991001, record, 1)
    validation = MODULE._validation_command(manifest, 991001, record, 1)
    assert training[training.index("--max-updates") + 1] == "1000"
    assert training[training.index("--save-interval") + 1] == "50"
    assert training[training.index("--seed") + 1] == "991001"
    assert training[training.index("--device") + 1] == "cuda:0"
    assert validation[validation.index("--device") + 1] == "cuda:0"
    assert validation[validation.index("--checkpoint") + 1].endswith("update_1000.pt")
