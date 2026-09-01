#!/usr/bin/env python3
"""Launch the frozen R1--R7 comparison for 150 seconds on twenty poses."""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
target = ROOT / "scripts/stomach_coverage/run_random_baseline_preexperiment.py"
arguments = [
    sys.executable,
    str(target),
    "--formal",
    "--config",
    str(ROOT / "configs/task009c/random_baseline_20pose_comparison_150s_v1.json"),
    "--output_root",
    str(ROOT / "artifacts/task009c_random_baseline_20pose_comparison_150s"),
    "--save_best_pose_snapshots",
    *sys.argv[1:],
]
os.execv(sys.executable, arguments)
