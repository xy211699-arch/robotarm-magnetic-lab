#!/usr/bin/env python3
"""Run the dependency-light stage-one action-layer contract tests."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(
        ROOT
        / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers"
    ),
)


def main() -> int:
    failures = []
    count = 0
    for path in sorted((ROOT / "tests/action_layer").glob("test_*.py")):
        namespace = runpy.run_path(str(path))
        for name, function in sorted(namespace.items()):
            if name.startswith("test_") and callable(function):
                count += 1
                try:
                    function()
                    print(f"PASS {path.name}::{name}")
                except Exception:
                    failures.append(f"{path.name}::{name}")
                    traceback.print_exc()
    print(f"ACTION_LAYER_PURE_TESTS total={count} failed={len(failures)}")
    if failures:
        print("FAILED=" + ",".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
