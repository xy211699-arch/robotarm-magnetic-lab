#!/usr/bin/env python3
"""Fake TASK-010 trainer used only by supervisor process tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, required=True)
    parser.add_argument("--save-interval", type=int, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--base-config-sha256", required=True)
    parser.add_argument("--visual-config-sha256", required=True)
    parser.add_argument("--dependency-audit-sha256", required=True)
    args = parser.parse_args()

    trace = Path(os.environ["TASK010_V2_FAKE_TRACE"])
    _append(
        trace,
        {
            "event": "start",
            "seed": args.seed,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
            "max_updates": args.max_updates,
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        },
    )
    delay = float(os.environ.get("TASK010_V2_FAKE_DELAY", "0.01"))
    time.sleep(delay)

    start_update = 0
    if args.resume_checkpoint is not None:
        start_update = int(
            torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)["current_update"]
        )
    fail_seed = int(os.environ.get("TASK010_V2_FAKE_FAIL_SEED", "-1"))
    fail_mode = os.environ.get("TASK010_V2_FAKE_FAIL_MODE", "")
    requested_final = start_update + args.max_updates
    if args.seed == fail_seed and fail_mode == "incomplete_zero":
        metric_update = min(requested_final - 1, max(437, start_update + 37))
        checkpoint_update = max(start_update, min(400, metric_update))
        exit_code = 0
    elif args.seed == fail_seed and fail_mode == "nonzero":
        _append(trace, {"event": "end", "seed": args.seed, "pid": os.getpid(), "time_ns": time.time_ns()})
        return 23
    else:
        metric_update = requested_final
        checkpoint_update = requested_final
        exit_code = 0

    output = args.output_dir
    _append(
        output / "events.jsonl",
        {
            "event": "runner_initialized",
            "seed": args.seed,
            "experiment_metadata": {
                "visual_condition": "blind",
                "base_config_sha256": args.base_config_sha256,
                "visual_dependence_config_sha256": args.visual_config_sha256,
            },
        },
    )
    _append(
        output / "metrics.jsonl",
        {
            "schema": "robotarm_magnetic_lab.task010_update_metric",
            "update": metric_update,
            "total_transitions": metric_update * 768,
            "elapsed_s": 20.0,
            "time_ns": time.time_ns(),
            "transitions_per_second": 38.4,
            "all_finite": True,
        },
    )
    checkpoint = output / "checkpoints" / f"update_{checkpoint_update:04d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "robotarm_magnetic_lab.task010_checkpoint",
            "current_update": checkpoint_update,
            "git_commit": args.git_commit,
            "config_hash": args.base_config_sha256,
            "base_config_sha256": args.base_config_sha256,
            "visual_dependence_config_sha256": args.visual_config_sha256,
            "dependency_audit_hash": args.dependency_audit_sha256,
            "experiment_metadata": {
                "visual_condition": "blind",
                "base_config_sha256": args.base_config_sha256,
                "visual_dependence_config_sha256": args.visual_config_sha256,
            },
        },
        checkpoint,
    )
    _append(trace, {"event": "end", "seed": args.seed, "pid": os.getpid(), "time_ns": time.time_ns()})
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
