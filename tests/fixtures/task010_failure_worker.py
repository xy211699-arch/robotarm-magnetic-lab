#!/usr/bin/env python3
"""Controlled worker used to verify TASK-010 supervisor evidence retention."""

import argparse
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--exit-code", type=int, default=23)
parser.add_argument("--sleep", type=float, default=0.0)
args = parser.parse_args()
time.sleep(args.sleep)
try:
    raise RuntimeError("TASK010_CONTROLLED_FAILURE")
except RuntimeError:
    traceback.print_exc()
sys.exit(args.exit_code)
