#!/usr/bin/env python3
"""Minimal fake stage for TASK-010 visual-dependence supervisor tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    delay = float(os.environ.get("TASK010_VISUAL_DEPENDENCE_FAKE_DELAY", "0.0"))
    fail_stage = os.environ.get("TASK010_VISUAL_DEPENDENCE_FAKE_FAIL_STAGE")
    if delay:
        time.sleep(delay)
    event = {
        "stage": args.stage,
        "attempt": args.attempt,
        "delay": delay,
        "failed": fail_stage is not None and fail_stage == args.stage,
    }
    events = args.run_dir / "fake_stage_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
    if event["failed"]:
        return 23
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
