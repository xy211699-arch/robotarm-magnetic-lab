#!/usr/bin/env python3
"""Headless all-eleven-action and P0 coverage integration validator."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--coverage_samples", type=int, default=5)
parser.add_argument("--output_directory", type=Path, default=None)
args, passthrough = parser.parse_known_args()
launcher = Path(__file__).with_name("teleop_atomic_stomach_coverage.py")
argv = [
    str(launcher),
    "--headless",
    "--num_envs",
    str(args.num_envs),
    "--scripted_actions",
    ",".join(str(value) for value in range(11)),
    "--minimum_coverage_samples",
    str(args.coverage_samples),
]
if args.output_directory is not None:
    argv.extend(["--output_directory", str(args.output_directory)])
argv.extend(passthrough)
sys.argv = argv
runpy.run_path(str(launcher), run_name="__main__")
