#!/usr/bin/env python3
"""CPU-only complete telemetry generator for Blind validation supervisor tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


POSE_IDS = (
    "validation-0006", "validation-0011", "validation-0015", "validation-0017",
    "validation-0019", "validation-0035", "validation-0040", "validation-0042",
    "validation-0045", "validation-0046", "validation-0051", "validation-0058",
    "validation-0060", "validation-0063", "validation-0067", "validation-0068",
    "validation-0069", "validation-0092", "validation-0095", "validation-0097",
)


def append(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    trace = args.run_dir / "fake_invocations.jsonl"
    active = args.run_dir / "fake_gpu_active.lock"
    try:
        descriptor = os.open(active, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        append(trace, {"event": "overlap", "seed": args.seed})
        return 99
    os.close(descriptor)
    append(trace, {"event": "start", "seed": args.seed, "attempt": args.attempt})
    try:
        if (
            os.environ.get("TASK010_BLIND_VALIDATION_FAIL_ONCE_SEED") == str(args.seed)
            and args.attempt == 1
        ):
            return 23
        args.output_dir.mkdir(parents=True, exist_ok=True)
        masks = args.output_dir / "final_masks"
        masks.mkdir(exist_ok=True)
        records = args.output_dir / "pose_records.jsonl"
        trajectories = args.output_dir / "coverage_trajectories.jsonl"
        telemetry_path = args.output_dir / "telemetry_10hz.jsonl"
        for path in (records, trajectories, telemetry_path):
            path.unlink(missing_ok=True)
        for index, pose_id in enumerate(POSE_IDS):
            coverage = np.linspace(0.05, 0.80 + 0.001 * index, 1201).tolist()
            append(records, {"pose_id": pose_id, "final_coverage": coverage[-1]})
            append(
                trajectories,
                {"pose_id": pose_id, "control_hz": 10, "coverage_fraction": coverage},
            )
            append(
                telemetry_path,
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
                },
            )
            np.savez_compressed(
                masks / f"{pose_id}.npz",
                reachable_coverage_mask=np.zeros(8, dtype=np.bool_),
            )
        return 0
    finally:
        append(trace, {"event": "finish", "seed": args.seed, "attempt": args.attempt})
        active.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
