#!/usr/bin/env python3
"""CPU-only fake stage process for the TASK-010 formal coordinator tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def append(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--stage", choices=("training", "validating"), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--training-dir", type=Path, required=True)
    result.add_argument("--validation-dir", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--attempt", type=int, required=True)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--plan", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    plan = json.loads(args.plan.read_text()) if args.plan else {}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    active = args.run_dir / "fake_gpu_active.lock"
    try:
        descriptor = os.open(active, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        append(args.run_dir / "fake_invocations.jsonl", {"event": "overlap", "seed": args.seed, "stage": args.stage})
        return 99
    os.close(descriptor)
    key = f"{args.seed}:{args.stage}"
    append(
        args.run_dir / "fake_invocations.jsonl",
        {"event": "start", "seed": args.seed, "stage": args.stage, "attempt": args.attempt},
    )
    try:
        time.sleep(float(plan.get("delay_s", 0.02)))
        if plan.get("fail_once", {}).get(key) is not None and args.attempt == 1:
            print(f"FAKE_CONTROLLED_FAILURE {key}", flush=True)
            return int(plan["fail_once"][key])
        if args.stage == "training":
            checkpoints = args.training_dir / "checkpoints"
            checkpoints.mkdir(parents=True, exist_ok=True)
            (checkpoints / "update_1000.pt").write_bytes(f"fake-checkpoint-{args.seed}".encode())
            append(args.training_dir / "metrics.jsonl", {"update": 1000, "seed": args.seed, "finite": True})
            return 0

        args.validation_dir.mkdir(parents=True, exist_ok=True)
        config = json.loads(args.config.read_text())
        pose_ids = list(config["validation"]["pose_ids"])
        invalid = plan.get("invalid", {}).get(key)
        if invalid == "pose_count":
            pose_ids = pose_ids[:-1]
        points = 1200 if invalid == "points" else 1201
        checkpoint = args.training_dir / "checkpoints/update_1000.pt"
        checkpoint_hash = sha256(checkpoint)
        config_hash = sha256(args.config)
        records_path = args.validation_dir / "pose_records.jsonl"
        trajectories_path = args.validation_dir / "coverage_trajectories.jsonl"
        records_path.unlink(missing_ok=True)
        trajectories_path.unlink(missing_ok=True)
        for row, pose_id in enumerate(pose_ids):
            values = [
                min(0.99, 0.01 * (args.seed - 991000) + 0.0001 * row + 0.5 * index / 1200.0)
                for index in range(points)
            ]
            if invalid == "nonfinite" and row == 0:
                values[17] = float("nan")
            append(
                records_path,
                {
                    "pose_id": pose_id,
                    "formal_steps": 1200,
                    "checkpoint_sha256": checkpoint_hash,
                    "config_sha256": config_hash,
                    "final_coverage": values[-1],
                    "total_reward": 1.0,
                    "mean_alpha": 0.5,
                },
            )
            append(
                trajectories_path,
                {
                    "pose_id": pose_id,
                    "checkpoint_sha256": checkpoint_hash,
                    "config_sha256": config_hash,
                    "control_hz": 10,
                    "coverage_fraction": values,
                },
            )
        (args.validation_dir / "summary.json").write_text(
            json.dumps({"pose_count": len(pose_ids), "all_finite": invalid != "nonfinite"}) + "\n",
            encoding="utf-8",
        )
        return 0
    finally:
        append(
            args.run_dir / "fake_invocations.jsonl",
            {"event": "finish", "seed": args.seed, "stage": args.stage, "attempt": args.attempt},
        )
        active.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
