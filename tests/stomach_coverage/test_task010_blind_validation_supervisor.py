from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stomach_coverage/task010_blind_validation_supervisor.py"
SPEC = importlib.util.spec_from_file_location("task010_blind_validation_supervisor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
FAKE_DRIVER = ROOT / "tests/fixtures/task010_blind_validation_fake_child.py"


def _source_run(tmp_path: Path) -> Path:
    run = tmp_path / "training"
    stages = {}
    for seed in MODULE.SEEDS:
        checkpoint = run / "training/blind" / f"seed_{seed}" / "checkpoints/update_1000.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        stages[f"train_blind_seed_{seed}"] = {
            "state": "completed",
            "completion_audit": {
                "complete": True,
                "observed_update": 1000,
                "latest_checkpoint": str(checkpoint),
                "checkpoint_sha256": MODULE._sha256(checkpoint),
            },
        }
    (run / "status.json").write_text(
        json.dumps({"state": "completed_training", "stages": stages}), encoding="utf-8"
    )
    return run


def _write_complete_validation(validation: Path, seed: int) -> None:
    validation.mkdir(parents=True)
    masks = validation / "final_masks"
    masks.mkdir()
    pose_records = []
    trajectories = []
    telemetry = []
    for index, pose_id in enumerate(MODULE.VALIDATION_POSE_IDS):
        coverage = np.linspace(0.05, 0.8 + index * 0.001, 1201).tolist()
        pose_records.append({"pose_id": pose_id, "final_coverage": coverage[-1]})
        trajectories.append({"pose_id": pose_id, "control_hz": 10, "coverage_fraction": coverage})
        telemetry.append(
            {
                "pose_id": pose_id,
                "control_hz": 10,
                "state_points": 1201,
                "action_points": 1200,
                "quaternion_order": "wxyz",
                "coverage_fraction": coverage,
                "position_world_m": [[0.0, 0.0, 0.0]] * 1201,
                "quaternion_wxyz": [[1.0, 0.0, 0.0, 0.0]] * 1201,
                "linear_velocity_world_m_s": [[0.0, 0.0, 0.0]] * 1201,
                "angular_velocity_world_rad_s": [[0.0, 0.0, 0.0]] * 1201,
                "action_mode": [0] * 1200,
                "action_alpha": [0.0] * 1200,
                "reward": [0.0] * 1200,
            }
        )
        np.savez_compressed(masks / f"{pose_id}.npz", reachable_coverage_mask=np.zeros(8, dtype=bool))
    for name, rows in (
        ("pose_records.jsonl", pose_records),
        ("coverage_trajectories.jsonl", trajectories),
        ("telemetry_10hz.jsonl", telemetry),
    ):
        (validation / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def test_discovers_three_completed_update1000_checkpoints(tmp_path: Path):
    checkpoints = MODULE.discover_checkpoints(_source_run(tmp_path))
    assert tuple(checkpoints) == MODULE.SEEDS
    assert all(path.name == "update_1000.pt" for path in checkpoints.values())


def test_validation_command_is_blind_deterministic_full_telemetry(tmp_path: Path):
    checkpoint = tmp_path / "update_1000.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = {
        "config": str(ROOT / "configs/task010/cnn_gru_development_v1.json"),
        "experiment_config": str(ROOT / "configs/task010/visual_dependence_v1.json"),
        "checkpoints": {"991001": str(checkpoint)},
        "test_driver": None,
    }
    command = MODULE.validation_command(manifest, tmp_path, seed=991001)
    assert command[command.index("--visual-condition") + 1] == "blind"
    assert command[command.index("--training-seed") + 1] == "991001"
    assert "--save-full-telemetry" in command
    assert command[command.index("--device") + 1] == "cuda:0"


def test_audit_requires_twenty_complete_telemetry_records_and_masks(tmp_path: Path):
    validation = tmp_path / "validation"
    _write_complete_validation(validation, seed=991001)
    result = MODULE.audit_validation(validation)
    assert result["pose_count"] == 20
    assert result["coverage_points"] == 1201
    assert result["telemetry_records"] == 20
    assert result["final_masks"] == 20
    telemetry = validation / "telemetry_10hz.jsonl"
    rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
    rows.pop()
    telemetry.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry"):
        MODULE.audit_validation(validation)


def test_initial_state_contains_only_three_sequential_validation_stages(tmp_path: Path):
    state = MODULE.initial_state(tmp_path, "run")
    assert tuple(state["seeds"]) == tuple(str(seed) for seed in MODULE.SEEDS)
    assert all(row["state"] == "queued" for row in state["seeds"].values())
    assert state["execution"]["max_parallel_children"] == 1


def test_git_drift_allows_only_project_run_log_updates():
    assert MODULE.git_drift_is_allowed([])
    assert MODULE.git_drift_is_allowed(["docs/PROJECT_RUN_LOG.md"])
    assert not MODULE.git_drift_is_allowed(
        ["docs/PROJECT_RUN_LOG.md", "scripts/stomach_coverage/validate_task010_checkpoint.py"]
    )
    assert not MODULE.git_drift_is_allowed(["configs/task010/cnn_gru_development_v1.json"])


def _wait_terminal(run_dir: Path, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = json.loads((run_dir / "status.json").read_text())
        if state["state"] in {"completed", "paused_on_error"}:
            return state
        time.sleep(0.02)
    raise AssertionError("validation supervisor did not reach a terminal state")


def test_fake_pipeline_runs_three_validations_sequentially(tmp_path: Path, monkeypatch):
    source = _source_run(tmp_path)
    monkeypatch.setenv("TASK010_BLIND_VALIDATION_TEST_MODE", "1")
    args = MODULE.parser().parse_args(
        [
            "start",
            "--source-run", str(source),
            "--output-root", str(tmp_path / "runs"),
            "--test-driver", str(FAKE_DRIVER),
        ]
    )
    started = MODULE._start(args)
    state = _wait_terminal(Path(started["run_dir"]))
    assert state["state"] == "completed"
    events = [
        json.loads(line)
        for line in Path(started["run_dir"], "fake_invocations.jsonl").read_text().splitlines()
    ]
    assert [(row["event"], row["seed"]) for row in events] == [
        ("start", 991001), ("finish", 991001),
        ("start", 991002), ("finish", 991002),
        ("start", 991003), ("finish", 991003),
    ]
    assert all(row["state"] == "validated" for row in state["seeds"].values())


def test_fake_failure_pauses_before_later_seed_and_continue_retries(tmp_path: Path, monkeypatch):
    source = _source_run(tmp_path)
    monkeypatch.setenv("TASK010_BLIND_VALIDATION_TEST_MODE", "1")
    monkeypatch.setenv("TASK010_BLIND_VALIDATION_FAIL_ONCE_SEED", "991002")
    args = MODULE.parser().parse_args(
        [
            "start", "--source-run", str(source),
            "--output-root", str(tmp_path / "runs"),
            "--test-driver", str(FAKE_DRIVER),
        ]
    )
    started = MODULE._start(args)
    run_dir = Path(started["run_dir"])
    paused = _wait_terminal(run_dir)
    assert paused["state"] == "paused_on_error"
    assert paused["current_seed"] == 991002
    assert paused["seeds"]["991003"]["attempts"] == 0
    deadline = time.monotonic() + 3.0
    while MODULE._pid_alive(paused["worker_pid"]) and time.monotonic() < deadline:
        time.sleep(0.02)
    monkeypatch.delenv("TASK010_BLIND_VALIDATION_FAIL_ONCE_SEED")
    continued = MODULE._continue(
        MODULE.parser().parse_args(["continue", "--run-dir", str(run_dir)])
    )
    assert continued["state"] == "queued"
    completed = _wait_terminal(run_dir)
    assert completed["state"] == "completed"
    assert completed["seeds"]["991001"]["attempts"] == 1
    assert completed["seeds"]["991002"]["attempts"] == 2
    assert completed["seeds"]["991003"]["attempts"] == 1
