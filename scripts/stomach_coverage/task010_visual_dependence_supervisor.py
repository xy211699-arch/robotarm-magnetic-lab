#!/usr/bin/env python3
"""Detached supervisor for the TASK-010 visual-dependence validation study."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY / "configs/task010/visual_dependence_v1.json"
DEFAULT_ARTIFACT_ROOT = (
    Path("/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence")
)
FORMAL_SEEDS = (991001, 991002, 991003)
HEARTBEAT_INTERVAL_S = 5.0
STALE_AFTER_S = 60.0
ACTIVE_STATES = {"queued", "running", "validating", "summarizing", "auditing"}


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"time_ns": time.time_ns(), **payload}, sort_keys=True) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _pid_alive(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        stat = Path(f"/proc/{int(pid)}/stat")
        if stat.is_file() and stat.read_text(encoding="utf-8").split()[2] == "Z":
            return False
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _update_link(link: Path, target: Path) -> None:
    temporary = link.with_name(link.name + f".{os.getpid()}.tmp")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    temporary.symlink_to(target)
    os.replace(temporary, link)


def _resolve_run(value: Path | None, output_root: Path) -> Path:
    candidate = Path(value) if value is not None else Path(output_root) / "latest"
    return candidate.resolve(strict=True)


def stage_names() -> tuple[str, ...]:
    names = []
    for seed in FORMAL_SEEDS:
        names.append(f"train_blind_seed_{seed}")
    for seed in FORMAL_SEEDS:
        for condition in ("normal", "blind", "donor", "first_frame"):
            names.append(f"validate_update750_{condition}_seed_{seed}")
    for seed in FORMAL_SEEDS:
        for condition in ("normal", "blind"):
            names.append(f"validate_update1000_{condition}_seed_{seed}")
    names.extend(("summarize", "audit_artifacts"))
    return tuple(names)


def _initial_state(run_dir: Path, run_id: str) -> dict:
    now = time.time()
    return {
        "schema": "robotarm_magnetic_lab.task010_visual_dependence_status",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state": "queued",
        "current_stage": None,
        "worker_pid": -1,
        "child_pid": None,
        "started_epoch_s": now,
        "heartbeat_epoch_s": now,
        "finished_epoch_s": None,
        "error_summary": None,
        "stages": {
            name: {
                "state": "queued",
                "attempts": 0,
                "exit_code": None,
                "started_at": None,
                "finished_at": None,
            }
            for name in stage_names()
        },
    }


def _spawn_worker(run_dir: Path, state: dict, *, continuation: bool) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--run-dir",
        str(run_dir),
    ]
    if continuation:
        command.append("--continuation")
    console = (run_dir / "coordinator.log").open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        stdin=subprocess.DEVNULL,
        stdout=console,
        stderr=console,
        start_new_session=True,
    )
    console.close()
    state["worker_pid"] = process.pid
    state["heartbeat_epoch_s"] = time.time()
    _atomic_json(run_dir / "status.json", state)
    return process.pid


def _stage_command(manifest: dict, stage: str, run_dir: Path, attempt: int) -> list[str]:
    if manifest.get("test_driver"):
        return [
            sys.executable,
            manifest["test_driver"],
            "--stage",
            stage,
            "--run-dir",
            str(run_dir),
            "--attempt",
            str(attempt),
        ]
    parts = stage.split("_")
    if stage.startswith("train_blind_seed_"):
        seed = parts[-1]
        return [
            sys.executable,
            str(REPOSITORY / "scripts/stomach_coverage/train_task010.py"),
            "--config",
            manifest["base_config"],
            "--visual-condition",
            "blind",
            "--seed",
            seed,
            "--max-updates",
            "1000",
            "--save-interval",
            "50",
            "--validation",
            "disabled",
            "--output-dir",
            str(run_dir / "training" / "blind" / f"seed_{seed}"),
            "--backend",
            "isaac",
            "--device",
            "cuda:0",
        ]
    if stage.startswith("validate_update750_"):
        _, _, _, condition, _, seed = parts
        update = "750"
        checkpoint = (
            Path(manifest["b0_run_dir"])
            / "seeds" / f"seed_{seed}" / "training" / "checkpoints" / f"update_{update}.pt"
        )
        output = run_dir / "validation" / "update750" / condition / f"seed_{seed}"
        command = [
            sys.executable,
            str(REPOSITORY / "scripts/stomach_coverage/validate_task010_checkpoint.py"),
            "--config",
            manifest["base_config"],
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--visual-condition",
            condition,
            "--experiment-config",
            manifest["config"],
            "--training-seed",
            seed,
        ]
        if condition == "normal":
            command += [
                "--save-feature-bank",
                str(run_dir / "feature_banks" / "update750" / f"seed_{seed}"),
            ]
        if condition == "donor":
            command += [
                "--donor-bank",
                str(run_dir / "feature_banks" / "update750" / f"seed_{seed}"),
            ]
        return command
    if stage.startswith("validate_update1000_"):
        _, _, _, condition, _, seed = parts
        checkpoint = (
            Path(manifest["b0_run_dir"])
            / "seeds" / f"seed_{seed}" / "training" / "checkpoints" / "update_1000.pt"
        )
        return [
            sys.executable,
            str(REPOSITORY / "scripts/stomach_coverage/validate_task010_checkpoint.py"),
            "--config",
            manifest["base_config"],
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(run_dir / "validation" / "update1000" / condition / f"seed_{seed}"),
            "--visual-condition",
            condition,
            "--experiment-config",
            manifest["config"],
            "--training-seed",
            seed,
        ]
    if stage == "summarize":
        return [
            sys.executable,
            str(REPOSITORY / "scripts/stomach_coverage/summarize_task010_visual_dependence.py"),
            "--run-dir",
            str(run_dir),
            "--config",
            manifest["config"],
        ]
    if stage == "audit_artifacts":
        return [
            sys.executable,
            str(REPOSITORY / "scripts/stomach_coverage/validate_task010_visual_dependence_gate.py"),
            "--config",
            manifest["config"],
            "--output",
            str(run_dir / "gates" / "gate_report.json"),
            "--self-check",
        ]
    raise ValueError(f"unknown visual-dependence stage: {stage}")


def _run_child(run_dir: Path, manifest: dict, state: dict, stage: str, attempt: int) -> int:
    command = _stage_command(manifest, stage, run_dir, attempt)
    log = run_dir / "logs" / f"{stage}_attempt_{attempt:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl(
        run_dir / "events.jsonl",
        {"event": "stage_started", "stage": stage, "attempt": attempt, "command": command},
    )
    with log.open("ab", buffering=0) as console:
        child = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=console,
            start_new_session=False,
        )
        state["child_pid"] = child.pid
        next_heartbeat = 0.0
        while child.poll() is None:
            now = time.time()
            if now >= next_heartbeat:
                state["heartbeat_epoch_s"] = now
                _atomic_json(run_dir / "status.json", state)
                next_heartbeat = now + HEARTBEAT_INTERVAL_S
            time.sleep(0.1)
    exit_code = int(child.returncode)
    state["child_pid"] = None
    state["heartbeat_epoch_s"] = time.time()
    _atomic_json(run_dir / "status.json", state)
    _append_jsonl(
        run_dir / "events.jsonl",
        {"event": "stage_finished", "stage": stage, "attempt": attempt, "exit_code": exit_code},
    )
    return exit_code


def _pause(run_dir: Path, state: dict, stage: str, error: BaseException | str) -> int:
    message = str(error)
    record = state["stages"][stage]
    record["state"] = "paused_on_error"
    record["error_summary"] = message
    state.update(
        state="paused_on_error",
        current_stage=stage,
        child_pid=None,
        heartbeat_epoch_s=time.time(),
        error_summary=message,
    )
    _atomic_json(run_dir / "status.json", state)
    _append_jsonl(
        run_dir / "events.jsonl",
        {"event": "paused_on_error", "stage": stage, "error": message},
    )
    return 1


def _worker(run_dir: Path, continuation: bool) -> int:
    run_dir = Path(run_dir).resolve(strict=True)
    lock_stream = (run_dir / ".coordinator.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 73
    manifest = _read_json(run_dir / "manifest.json")
    state = _read_json(run_dir / "status.json")
    state.update(worker_pid=os.getpid(), heartbeat_epoch_s=time.time(), error_summary=None)
    _atomic_json(run_dir / "status.json", state)
    _append_jsonl(
        run_dir / "events.jsonl",
        {"event": "worker_started", "continuation": continuation, "worker_pid": os.getpid()},
    )
    try:
        for stage in stage_names():
            record = state["stages"][stage]
            if record["state"] == "completed":
                continue
            record["attempts"] += 1
            attempt = int(record["attempts"])
            record.update(
                state="running",
                error_summary=None,
                started_at=time.time(),
                finished_at=None,
            )
            state.update(state=stage, current_stage=stage, error_summary=None)
            _atomic_json(run_dir / "status.json", state)
            if _run_child(run_dir, manifest, state, stage, attempt) != 0:
                return _pause(run_dir, state, stage, "stage process failed")
            record.update(state="completed", exit_code=0, finished_at=time.time())
            state.update(heartbeat_epoch_s=time.time(), current_stage=None)
            _atomic_json(run_dir / "status.json", state)
        state.update(
            state="completed",
            current_stage=None,
            child_pid=None,
            finished_epoch_s=time.time(),
            heartbeat_epoch_s=time.time(),
            error_summary=None,
        )
        _atomic_json(run_dir / "status.json", state)
        _append_jsonl(run_dir / "events.jsonl", {"event": "completed"})
        return 0
    except BaseException as error:
        with (run_dir / "coordinator.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        stage = state.get("current_stage") or stage_names()[0]
        return _pause(run_dir, state, stage, error)
    finally:
        lock_stream.close()


def _status_payload(run_dir: Path) -> dict:
    state = _read_json(run_dir / "status.json")
    now = time.time()
    result = json.loads(json.dumps(state))
    result["runtime_s"] = max(0.0, now - float(state.get("started_epoch_s", now)))
    result["heartbeat_age_s"] = max(0.0, now - float(state.get("heartbeat_epoch_s", 0.0)))
    if state.get("state") in ACTIVE_STATES and (
        result["heartbeat_age_s"] > STALE_AFTER_S or not _pid_alive(state.get("worker_pid"))
    ):
        result["state"] = "paused_on_error"
        result["error_summary"] = (
            f"coordinator state is stale: heartbeat_age_s={result['heartbeat_age_s']:.3f}, "
            f"worker_pid={state.get('worker_pid')}"
        )
    return result


def _start(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("visual-dependence config schema mismatch")
    b0_run_dir = Path(args.b0_run_dir).resolve(strict=True)
    for seed in FORMAL_SEEDS:
        for update in (750, 1000):
            checkpoint = (
                b0_run_dir / "seeds" / f"seed_{seed}" / "training" / "checkpoints" / f"update_{update}.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing B0 checkpoint: {checkpoint}")
    root = Path(args.artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            existing = _status_payload(_resolve_run(latest, root))
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {"state": "unknown"}
        if existing.get("state") in ACTIVE_STATES:
            raise RuntimeError("visual-dependence supervisor is already active")
    if args.test_driver and os.environ.get("TASK010_VISUAL_DEPENDENCE_TEST_MODE") != "1":
        raise PermissionError("test driver requires TASK010_VISUAL_DEPENDENCE_TEST_MODE=1")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = root / run_id
    run_dir.mkdir(parents=False)
    manifest = {
        "schema": "robotarm_magnetic_lab.task010_visual_dependence_manifest",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": config.get("config_sha256"),
        "base_config": str(config_path.parent / config["base_config"]["path"]),
        "b0_run_dir": str(b0_run_dir),
        "test_driver": str(Path(args.test_driver).resolve()) if args.test_driver else None,
        "stages": list(stage_names()),
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    state = _initial_state(run_dir, run_id)
    _atomic_json(run_dir / "status.json", state)
    _append_jsonl(run_dir / "events.jsonl", {"event": "queued", "stages": list(stage_names())})
    pid = _spawn_worker(run_dir, state, continuation=False)
    _update_link(latest, run_dir)
    latest_path = root / "latest_run_path.txt"
    latest_path.write_text(str(run_dir) + "\n", encoding="utf-8")
    result = {"run_id": run_id, "run_dir": str(run_dir), "worker_pid": pid, "state": "queued"}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _status(args) -> dict:
    run_dir = _resolve_run(args.run_dir, args.output_root)
    result = _status_payload(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _watch(args) -> None:
    run_dir = _resolve_run(args.run_dir, args.output_root)
    try:
        while True:
            result = _status_payload(run_dir)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            if result.get("state") in {"completed", "paused_on_error"}:
                return
            time.sleep(int(args.interval))
    except KeyboardInterrupt:
        print("watch interrupted; background worker remains alive", flush=True)


def _continue(args) -> dict:
    run_dir = _resolve_run(args.run_dir, args.output_root)
    state = _read_json(run_dir / "status.json")
    effective = _status_payload(run_dir)
    if state.get("state") in ACTIVE_STATES and effective.get("state") == "paused_on_error":
        state.update(
            state="paused_on_error",
            child_pid=None,
            heartbeat_epoch_s=time.time(),
            error_summary=effective["error_summary"],
        )
        stage = state.get("current_stage")
        if stage:
            state["stages"][stage]["state"] = "paused_on_error"
        _atomic_json(run_dir / "status.json", state)
    if state.get("state") != "paused_on_error":
        raise RuntimeError("continue requires persisted state=paused_on_error")
    if _pid_alive(state.get("worker_pid")):
        raise RuntimeError("cannot continue while previous coordinator PID is alive")
    pid = _spawn_worker(run_dir, state, continuation=True)
    result = {"run_id": state["run_id"], "run_dir": str(run_dir), "worker_pid": pid, "state": "paused_on_error"}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    start.add_argument("--b0-run-dir", type=Path, required=True)
    start.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    start.add_argument("--test-driver", help=argparse.SUPPRESS)
    start.add_argument("--kit_args", help=argparse.SUPPRESS)
    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", type=Path)
    status.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    status.add_argument("--kit_args", help=argparse.SUPPRESS)
    watch = subparsers.add_parser("watch")
    watch.add_argument("--run-dir", type=Path)
    watch.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    watch.add_argument("--interval", type=int, default=60)
    watch.add_argument("--kit_args", help=argparse.SUPPRESS)
    continuation = subparsers.add_parser("continue")
    continuation.add_argument("--run-dir", type=Path, required=True)
    continuation.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    continuation.add_argument("--kit_args", help=argparse.SUPPRESS)
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--continuation", action="store_true")
    worker.add_argument("--kit_args", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "start":
        _start(args)
        return 0
    if args.command == "status":
        _status(args)
        return 0
    if args.command == "watch":
        _watch(args)
        return 0
    if args.command == "continue":
        _continue(args)
        return 0
    return _worker(args.run_dir, args.continuation)


if __name__ == "__main__":
    raise SystemExit(main())
