#!/usr/bin/env python3
"""Read-only status query for the R1--R7 frozen 20-pose comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/task009c_random_baseline_20pose_comparison"
EXPECTED_EPISODES = 140


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def query_status(output_root: Path) -> dict:
    pointer_path = Path(output_root) / "latest_formal_manifest.json"
    if not pointer_path.is_file():
        return {"state": "not_started", "completed_episodes": 0, "expected_episodes": EXPECTED_EPISODES}
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = Path(pointer["manifest_path"])
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size != int(pointer["manifest_bytes"])
        or _sha256(manifest_path) != pointer["manifest_sha256"]
    ):
        raise RuntimeError("latest formal manifest pointer failed size/SHA-256 validation")
    rows = _read_jsonl(manifest_path)
    starts = [row for row in rows if row.get("record_type") == "run_start"]
    episodes = [
        row for row in rows
        if row.get("record_type") == "episode" and row.get("status") == "pass"
    ]
    failures = [row for row in rows if row.get("record_type") == "episode_failure"]
    completed = any(row.get("record_type") == "run_complete" for row in rows)
    per_policy = {
        f"R{index}": sum(row.get("policy_id") == f"R{index}" for row in episodes)
        for index in range(1, 8)
    }
    started_at = None if not starts else starts[0].get("started_at_utc")
    elapsed_s = None
    if started_at:
        started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        elapsed_s = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
    latest = None if not episodes else episodes[-1]
    return {
        "state": "completed" if completed else "failed" if failures else "running",
        "run_id": pointer.get("run_id"),
        "completed_episodes": len(episodes),
        "expected_episodes": EXPECTED_EPISODES,
        "progress_percent": 100.0 * len(episodes) / EXPECTED_EPISODES,
        "per_policy_completed": per_policy,
        "latest_episode_id": None if latest is None else latest.get("episode_id"),
        "latest_policy_id": None if latest is None else latest.get("policy_id"),
        "latest_pose_id": None if latest is None else latest.get("pose_id"),
        "latest_final_reachable_coverage": (
            None if latest is None else latest.get("C_final_reachable")
        ),
        "failure_count": len(failures),
        "latest_failure": None if not failures else failures[-1],
        "elapsed_s": elapsed_s,
        "manifest_path": str(manifest_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(query_status(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
