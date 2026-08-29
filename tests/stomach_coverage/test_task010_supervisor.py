from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "scripts/stomach_coverage/task010_training_supervisor.py"
CONFIG = ROOT / "configs/task010/cnn_gru_development_v1.json"
FAILURE = ROOT / "tests/fixtures/task010_failure_worker.py"
SPEC = importlib.util.spec_from_file_location("task010_training_supervisor", SUPERVISOR)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _wait(run_dir: Path, timeout: float = 12.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = json.loads((run_dir / "status.json").read_text())
        if value["state"] in ("completed", "failed"):
            return value
        time.sleep(0.1)
    raise AssertionError("worker did not reach terminal state")


def _start(tmp_path: Path, *worker_args: str):
    start = time.monotonic()
    result = subprocess.run([
        sys.executable, str(SUPERVISOR), "start", "--config", str(CONFIG),
        "--output-root", str(tmp_path), "--worker-command", str(FAILURE),
        *(f"--worker-arg={item}" for item in worker_args),
    ], check=True, text=True, capture_output=True)
    return time.monotonic() - start, Path(json.loads(result.stdout)["run_dir"])


def test_start_returns_before_worker_finishes(tmp_path: Path):
    elapsed, run_dir = _start(tmp_path, "--sleep=3", "--exit-code=23")
    assert elapsed < 1.5
    assert (run_dir / "launch_manifest.json").is_file()
    assert _wait(run_dir)["exit_code"] == 23


def test_launch_manifest_captures_reproducibility_identity(tmp_path: Path):
    _, run_dir = _start(tmp_path, "--sleep=1", "--exit-code=0")
    manifest = json.loads((run_dir / "launch_manifest.json").read_text())
    assert manifest["command"]
    assert manifest["hostname"]
    assert manifest["worker_pid"] > 0
    assert len(manifest["git"]["commit"]) == 40
    assert len(manifest["git"]["worktree_status_sha256"]) == 64
    assert manifest["config"]["path"] == str(CONFIG)
    assert len(manifest["config"]["sha256"]) == 64
    assert manifest["config"]["num_envs"] == 12
    assert manifest["python"]["version"]
    assert manifest["packages"]["torch"]
    assert "devices" in manifest["gpu"] or "query_error" in manifest["gpu"]
    assert _wait(run_dir)["state"] == "completed"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert any(item["event"] == "preflight_verified" for item in events)


def test_failure_preserves_exit_code_stderr_and_traceback(tmp_path: Path):
    _, run_dir = _start(tmp_path, "--exit-code=23")
    status = _wait(run_dir)
    log = (run_dir / "console.log").read_text()
    assert status["state"] == "failed" and status["exit_code"] == 23
    assert "TASK010_CONTROLLED_FAILURE" in log and "Traceback" in log
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "worker_finished"


def test_active_output_root_rejects_concurrent_start(tmp_path: Path):
    _, run_dir = _start(tmp_path, "--sleep=3", "--exit-code=23")
    duplicate = subprocess.run([
        sys.executable, str(SUPERVISOR), "start", "--config", str(CONFIG),
        "--output-root", str(tmp_path), "--worker-command", str(FAILURE),
    ], text=True, capture_output=True)
    assert duplicate.returncode != 0 and "active worker" in duplicate.stderr
    _wait(run_dir)


@pytest.mark.parametrize("parent_save_interval", (50, 1))
def test_resume_preserves_parent_checkpoint_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_save_interval: int
):
    parent = tmp_path / "parent"
    checkpoint = parent / "checkpoints" / "update_0050.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"task010-checkpoint")
    manifest = {
        "run_id": "parent-run",
        "save_interval": parent_save_interval,
        "command": [
            "run_isaaclab.sh", "-p", "train_task010.py",
            "--config", str(CONFIG), "--seed", "991000",
        ],
    }
    (parent / "launch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    captured = {}

    def fake_launch(launch, *, label, parent):
        captured["parameters"] = launch
        captured["label"] = label
        captured["parent"] = parent
        captured["launch_manifest"] = {"save_interval": int(launch.save_interval)}
        return captured

    monkeypatch.setattr(MODULE, "_launch", fake_launch)
    result = MODULE._resume(
        SimpleNamespace(
            run_dir=parent,
            checkpoint=checkpoint.name,
            additional_updates=25,
        )
    )

    assert result["parameters"].max_updates == 25
    assert result["parameters"].save_interval == parent_save_interval
    assert result["launch_manifest"]["save_interval"] == parent_save_interval
    assert result["label"] == "latest-resume"
