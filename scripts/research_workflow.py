"""Research records and repository checks without simulator dependencies.

This module never starts experiments, installs packages, or accesses the network.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"draft", "authorized", "in_progress", "partial", "complete", "needs_decision", "blocked", "historical"}
ID_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\Z")


def _object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError(f"{path}: expected an object with schema_version=1")
    return value


def _relative_path(root: Path, value: str) -> Path:
    if (not isinstance(value, str) or not value or "\\" in value or
            PureWindowsPath(value).drive or Path(value).is_absolute() or ".." in Path(value).parts):
        raise ValueError(f"Invalid repository-relative path: {value!r}")
    root = root.resolve()
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes its root: {value}")
    return resolved


def _existing(root: Path, value: str) -> Path:
    path = _relative_path(root, value)
    if not path.exists():
        raise ValueError(f"Missing path: {value}")
    return path


def _local_links(root: Path, document: Path) -> list[str]:
    errors = []
    # Fenced examples are not actual navigation links.
    content = document.read_text(encoding="utf-8-sig")
    lines, fence = [], None
    for line in content.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif fence == marker.group(1)[0]:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", "\n".join(lines)):
        target = target.strip().strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        relative = unquote(parsed.path)
        destination = (document.parent / relative).resolve()
        if not destination.is_relative_to(root) or not destination.exists():
            errors.append(f"{document.relative_to(root)}: broken or escaping link {target}")
    return errors


def check_repository(root: Path) -> list[str]:
    """Check the managed documentation, architecture map, and task references."""
    root = root.resolve()
    errors = []
    try:
        layout = _object(root / "docs/repository_layout.json")
        for field in ("required_paths", "managed_documents"):
            values = layout.get(field)
            if not isinstance(values, list) or not values:
                errors.append(f"repository_layout.json: {field} must be a nonempty list")
                continue
            for value in values:
                try:
                    path = _existing(root, value)
                    if field == "managed_documents":
                        if not path.is_file() or path.suffix != ".md":
                            raise ValueError(f"Expected Markdown document: {value}")
                        errors.extend(_local_links(root, path))
                except (OSError, ValueError) as exc:
                    errors.append(str(exc))
        components = layout.get("components")
        if not isinstance(components, list):
            errors.append("repository_layout.json: components must be a list")
        else:
            for component in components:
                try:
                    if not isinstance(component, dict) or not isinstance(component.get("purpose"), str):
                        raise ValueError("Invalid architecture component")
                    _existing(root, component.get("path"))
                except ValueError as exc:
                    errors.append(str(exc))
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    try:
        state = _object(root / "docs/research_state.json")
        date.fromisoformat(state.get("as_of", ""))
        tasks = state.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("research_state.json: tasks must be a list")
        ids = set()
        for task in tasks:
            if not isinstance(task, dict):
                errors.append("research_state.json: invalid task object")
                continue
            task_id = task.get("id")
            if not isinstance(task_id, str) or not ID_PATTERN.fullmatch(task_id):
                errors.append(f"Invalid task ID: {task_id!r}")
                continue
            if task_id in ids:
                errors.append(f"Duplicate task ID: {task_id}")
            ids.add(task_id)
            if task.get("status") not in STATUSES:
                errors.append(f"{task_id}: invalid task status")
            for field in ("title", "summary"):
                if not isinstance(task.get(field), str) or not task[field].strip():
                    errors.append(f"{task_id}: missing {field}")
            if not task.get("contract"):
                errors.append(f"{task_id}: missing contract")
            for field in ("contract", "plan", "report"):
                if task.get(field) is not None:
                    try:
                        path = _existing(root, task[field])
                        if not path.is_file():
                            raise ValueError(f"{task_id}: {field} is not a file")
                    except ValueError as exc:
                        errors.append(str(exc))
        if state.get("active_task") not in ids:
            errors.append(f"Unknown active task: {state.get('active_task')}")
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"research_state.json: {exc}")
    return errors


def _validate_id(value: str) -> None:
    if not ID_PATTERN.fullmatch(value):
        raise ValueError("ID must contain uppercase letters, digits and separating hyphens")


def create_task(root: Path, task_id: str, title: str) -> Path:
    """Create a draft contract exclusively; never modify the active task registry."""
    _validate_id(task_id)
    if not title.strip() or "\n" in title or "\r" in title:
        raise ValueError("A nonempty, single-line title is required")
    state = _object(root / "docs/research_state.json")
    tasks = state.get("tasks")
    if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
        raise ValueError("research_state.json: tasks must contain task objects")
    if any(task.get("id") == task_id for task in tasks):
        raise ValueError(f"Task ID is already registered: {task_id}")
    template = _existing(root, "docs/templates/task.md").read_text(encoding="utf-8")
    content = template.replace("{task_id}", task_id).replace("{title}", title)
    path = _relative_path(root, f"handoffs/active/{task_id}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return path


def _git_identity(root: Path) -> dict:
    def read(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root.resolve().as_posix()}", "-C", str(root), *args],
            text=True, encoding="utf-8", stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    try:
        if Path(read("rev-parse", "--show-toplevel")).resolve() != root.resolve():
            raise ValueError("Requested root is not the Git checkout root")
        return {"head": read("rev-parse", "HEAD"), "branch": read("branch", "--show-current"),
                "dirty": bool(read("status", "--porcelain")), "available": True}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"head": None, "branch": None, "dirty": None, "available": False}


def create_run(root: Path, experiment_id: str, output_root: Path | None = None) -> Path:
    """Initialize a record, with no simulation, training, or command execution."""
    _validate_id(experiment_id)
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    destination = (_relative_path(root, "artifacts/research") if output_root is None else output_root.resolve())
    run = _relative_path(destination, f"{experiment_id}/{run_id}")
    run.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "experiment_id": experiment_id, "run_id": run_id,
        "created_at_utc": now.isoformat(), "status": "initialized", "code": _git_identity(root),
        "hypothesis": None, "command": None, "config": None, "seeds": [],
        "artifacts": [], "metrics": {},
        "verification_scope": "File integrity only; initialization is not experiment execution.",
    }
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run


def verify_run(run: Path) -> list[str]:
    """Verify listed local artifact bytes, without inferring scientific validity."""
    errors = []
    try:
        manifest = _object(run / "manifest.json")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return ["manifest.json: no artifact evidence has been registered"]
        seen = set()
        for artifact in artifacts:
            try:
                if not isinstance(artifact, dict):
                    raise ValueError("Invalid artifact entry")
                path = _existing(run, artifact.get("path"))
                if path in seen:
                    raise ValueError(f"Duplicate artifact: {artifact.get('path')}")
                seen.add(path)
                if not path.is_file() or path == (run / "manifest.json").resolve():
                    raise ValueError("Artifact must be a file other than its own manifest")
                size, expected = artifact.get("bytes"), artifact.get("sha256")
                if type(size) is not int or size < 0 or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise ValueError(f"Invalid size or SHA-256 for {artifact.get('path')}")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if path.stat().st_size != size or digest.hexdigest() != expected:
                    errors.append(f"Integrity mismatch: {artifact.get('path')}")
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (defaults to this script's checkout)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check managed repository documentation and references")
    commands.add_parser("context", help="Print the reading order and recorded research tasks")
    task_parser = commands.add_parser("new-task", help="Create a draft; does not grant execution authority")
    task_parser.add_argument("task_id")
    task_parser.add_argument("--title", required=True)
    run_parser = commands.add_parser("new-run", help="Initialize a run record; does not run an experiment")
    run_parser.add_argument("experiment_id")
    run_parser.add_argument("--output-root", type=Path, help="Explicit record storage override")
    verify_parser = commands.add_parser("verify-run", help="Verify registered artifact sizes and SHA-256 hashes")
    verify_parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command in {"check", "context"}:
            errors = check_repository(root)
        elif args.command == "verify-run":
            errors = verify_run(args.directory.resolve())
        else:
            errors = []
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        if args.command == "context":
            state = _object(root / "docs/research_state.json")
            print(f"Repository: {root}\nRecorded as of: {state['as_of']}\nActive task: {state['active_task']}")
            print("Read AGENTS.md, docs/STATUS.md, docs/TODO.md, then the active contract and its plan.")
            for task in state["tasks"]:
                print(f"{task['id']} [{task['status']}]: {task['summary']}")
                print(f"Contract: {task['contract']} | Plan: {task.get('plan')} | Report: {task.get('report')}")
            print("Repository records are not live remote status; new drafts do not authorize execution.")
        elif args.command == "new-task":
            print(create_task(root, args.task_id, args.title))
        elif args.command == "new-run":
            print(create_run(root, args.experiment_id, args.output_root))
        elif args.command == "check":
            print("PASS: managed repository paths, document links and task references")
        else:
            print("PASS: listed artifact sizes and SHA-256 only; scientific conclusions were not verified")
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
