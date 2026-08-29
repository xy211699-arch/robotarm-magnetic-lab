#!/usr/bin/env python3
"""Detached TASK-010 training supervisor with durable status and failure evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid


REPOSITORY = Path(__file__).resolve().parents[2]
HEARTBEAT_INTERVAL_S = 5.0
STALE_AFTER_S = 60.0


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_event(run_dir: Path, payload: dict) -> None:
    payload = {"time_ns": time.time_ns(), **payload}
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _resolved_run(path: Path) -> Path:
    return Path(path).resolve(strict=True)


def _status(run_dir: Path) -> dict:
    run_dir = _resolved_run(run_dir)
    path = run_dir / "status.json"
    if not path.is_file():
        return {"state": "unknown", "run_dir": str(run_dir)}
    value = _read_json(path)
    if value.get("state") in ("starting", "running"):
        age = time.time() - float(value.get("heartbeat_epoch_s", 0.0))
        if age > STALE_AFTER_S or not _pid_alive(int(value.get("worker_pid", -1))):
            value = {**value, "state": "stale", "heartbeat_age_s": age}
    return value


def _update_link(link: Path, target: Path) -> None:
    temporary = link.with_name(link.name + f".{os.getpid()}.tmp")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    temporary.symlink_to(target)
    os.replace(temporary, link)


def _command_from_args(args, run_dir: Path) -> list[str]:
    if args.worker_command:
        worker = Path(args.worker_command)
        prefix = [sys.executable, str(worker)] if worker.suffix == ".py" else [str(worker)]
        return prefix + list(args.worker_arg or [])
    command = [
        str(REPOSITORY / "run_isaaclab.sh"), "-p",
        "scripts/stomach_coverage/train_task010.py",
        "--config", str(Path(args.config).resolve()),
        "--output-dir", str(run_dir),
        "--max-updates", str(args.max_updates),
        "--save-interval", str(args.save_interval),
        "--validation", str(args.validation),
        "--backend", "isaac", "--device", "cuda:0", "--headless",
    ]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if getattr(args, "resume_checkpoint", None):
        command += ["--resume-checkpoint", str(Path(args.resume_checkpoint).resolve())]
    return command


def _launch(args, *, label: str = "latest", parent: dict | None = None) -> dict:
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    link = root / label
    if link.exists() or link.is_symlink():
        try:
            existing = _status(link)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {"state": "unknown"}
        if existing.get("state") in ("starting", "running"):
            raise RuntimeError(f"TASK-010 active worker already exists at {link}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = root / run_id
    run_dir.mkdir(parents=False)
    command = _command_from_args(args, run_dir)
    manifest = {
        "schema": "robotarm_magnetic_lab.task010_launch_manifest",
        "run_id": run_id, "run_dir": str(run_dir), "command": command,
        "created_epoch_s": time.time(), "parent": parent,
    }
    _atomic_json(run_dir / "launch_manifest.json", manifest)
    initial = {"state": "starting", "run_id": run_id, "run_dir": str(run_dir), "heartbeat_epoch_s": time.time(), "worker_pid": -1}
    _atomic_json(run_dir / "status.json", initial)
    supervisor = [sys.executable, str(Path(__file__).resolve()), "_worker", "--run-dir", str(run_dir)]
    console = (run_dir / "console.log").open("ab", buffering=0)
    process = subprocess.Popen(supervisor, cwd=REPOSITORY, stdin=subprocess.DEVNULL, stdout=console, stderr=console, start_new_session=True)
    console.close()
    initial["worker_pid"] = process.pid
    _atomic_json(run_dir / "status.json", initial)
    _update_link(link, run_dir)
    result = {"run_id": run_id, "pid": process.pid, "run_dir": str(run_dir), "status": "starting", "link": str(link)}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _worker(run_dir: Path) -> int:
    run_dir = Path(run_dir).resolve()
    manifest = _read_json(run_dir / "launch_manifest.json")
    status = {"state": "running", "run_id": manifest["run_id"], "run_dir": str(run_dir), "worker_pid": os.getpid(), "heartbeat_epoch_s": time.time(), "exit_code": None}
    _atomic_json(run_dir / "status.json", status); _append_event(run_dir, {"event": "worker_started", "pid": os.getpid()})
    try:
        with (run_dir / "console.log").open("ab", buffering=0) as console:
            child = subprocess.Popen(manifest["command"], cwd=REPOSITORY, stdin=subprocess.DEVNULL, stdout=console, stderr=console, start_new_session=False)
            next_heartbeat = 0.0
            while child.poll() is None:
                now = time.time()
                if now >= next_heartbeat:
                    status["heartbeat_epoch_s"] = now; status["child_pid"] = child.pid
                    _atomic_json(run_dir / "status.json", status)
                    next_heartbeat = now + HEARTBEAT_INTERVAL_S
                time.sleep(0.1)
            code = int(child.returncode)
        status.update(state="completed" if code == 0 else "failed", exit_code=code, heartbeat_epoch_s=time.time(), finished_epoch_s=time.time())
        _atomic_json(run_dir / "status.json", status); _append_event(run_dir, {"event": "worker_finished", "exit_code": code, "state": status["state"]})
        return code
    except BaseException:
        with (run_dir / "console.log").open("a", encoding="utf-8") as console:
            traceback.print_exc(file=console)
        status.update(state="failed", exit_code=1, heartbeat_epoch_s=time.time(), finished_epoch_s=time.time())
        _atomic_json(run_dir / "status.json", status); _append_event(run_dir, {"event": "worker_exception", "exit_code": 1})
        return 1


def _resume(args) -> dict:
    parent_dir = _resolved_run(args.run_dir)
    manifest = _read_json(parent_dir / "launch_manifest.json")
    checkpoint = parent_dir / "checkpoints" / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TASK-010 resume checkpoint not found: {checkpoint}")
    original = manifest["command"]
    config_index = original.index("--config") + 1
    class ResumeArgs: pass
    launch = ResumeArgs()
    launch.output_root = parent_dir.parent
    launch.worker_command = None; launch.worker_arg = []
    launch.config = original[config_index]
    launch.max_updates = int(args.additional_updates); launch.save_interval = 1
    launch.validation = "disabled"; launch.seed = None
    launch.resume_checkpoint = checkpoint
    return _launch(launch, label="latest-resume", parent={"run_id": manifest["run_id"], "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "additional_updates": int(args.additional_updates)})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--config", type=Path, required=True); start.add_argument("--output-root", type=Path, required=True)
    start.add_argument("--seed", type=int); start.add_argument("--max-updates", type=int, default=2)
    start.add_argument("--save-interval", type=int, default=1); start.add_argument("--validation", choices=("enabled", "disabled"), default="disabled")
    start.add_argument("--worker-command"); start.add_argument("--worker-arg", action="append", default=[])
    start.add_argument("--kit_args", help=argparse.SUPPRESS)
    status = sub.add_parser("status"); status.add_argument("--run-dir", type=Path, required=True); status.add_argument("--kit_args", help=argparse.SUPPRESS)
    resume = sub.add_parser("resume"); resume.add_argument("--run-dir", type=Path, required=True); resume.add_argument("--checkpoint", required=True); resume.add_argument("--additional-updates", type=int, required=True); resume.add_argument("--kit_args", help=argparse.SUPPRESS)
    worker = sub.add_parser("_worker"); worker.add_argument("--run-dir", type=Path, required=True); worker.add_argument("--kit_args", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "start": _launch(args); return 0
    if args.command == "status": print(json.dumps(_status(args.run_dir), indent=2, sort_keys=True)); return 0
    if args.command == "resume": _resume(args); return 0
    return _worker(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
