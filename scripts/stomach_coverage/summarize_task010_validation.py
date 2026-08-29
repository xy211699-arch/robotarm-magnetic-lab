#!/usr/bin/env python3
"""Strict offline summarizer for TASK-010's frozen twenty-pose validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VALIDATION_POSE_IDS = (
    "validation-0006", "validation-0011", "validation-0015", "validation-0017",
    "validation-0019", "validation-0035", "validation-0040", "validation-0042",
    "validation-0045", "validation-0046", "validation-0051", "validation-0058",
    "validation-0060", "validation-0063", "validation-0067", "validation-0068",
    "validation-0069", "validation-0092", "validation-0095", "validation-0097",
)


def validation_batches(pose_ids=VALIDATION_POSE_IDS, batch_size: int = 12):
    values = tuple(pose_ids)
    if values != VALIDATION_POSE_IDS or batch_size != 12:
        raise ValueError("TASK-010 validation pose IDs and 12-row batching are frozen")
    return values[:12], values[12:]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(records_path: Path, *, checkpoint_sha256: str, config_sha256: str) -> dict:
    rows = [json.loads(line) for line in Path(records_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("pose_id")) for row in rows]
    if len(rows) != 20 or len(set(ids)) != 20 or set(ids) != set(VALIDATION_POSE_IDS):
        raise ValueError("TASK-010 validation requires exactly twenty unique frozen pose IDs")
    if any(int(row.get("formal_steps", -1)) != 1200 for row in rows):
        raise ValueError("TASK-010 validation record must contain exactly 1200 formal steps")
    if any(row.get("checkpoint_sha256") != checkpoint_sha256 for row in rows):
        raise ValueError("TASK-010 validation checkpoint hash mismatch")
    if any(row.get("config_sha256") != config_sha256 for row in rows):
        raise ValueError("TASK-010 validation config hash mismatch")
    numeric = ("final_coverage", "total_reward", "mean_alpha")
    for row in rows:
        for name in numeric:
            value = float(row[name])
            if not (-float("inf") < value < float("inf")):
                raise ValueError(f"TASK-010 validation has non-finite {name}")
    ordered = sorted(rows, key=lambda row: VALIDATION_POSE_IDS.index(row["pose_id"]))
    return {
        "schema": "robotarm_magnetic_lab.task010_validation_summary",
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "pose_ids": list(VALIDATION_POSE_IDS),
        "pose_count": 20,
        "batch_sizes": [12, 8],
        "mean_final_coverage": sum(float(row["final_coverage"]) for row in ordered) / 20.0,
        "mean_total_reward": sum(float(row["total_reward"]) for row in ordered) / 20.0,
        "all_finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.records, checkpoint_sha256=file_sha256(args.checkpoint), config_sha256=args.config_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
