"""Stomach-scene entry points for the validated open-loop motion controller.

The motion generators and magnetic inverse are shared with the flat-table
acceptance suite.  Only the registered Isaac Lab task, output directory and
surface profile change here.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_DIR = Path("/mnt/isaac-linux/robotarm_magnetic_lab")
TABLE_MOTION_DIR = PROJECT_DIR / "scripts" / "table_motion"
if str(TABLE_MOTION_DIR) not in sys.path:
    sys.path.insert(0, str(TABLE_MOTION_DIR))

from table_test_common import run_cli  # noqa: E402


STOMACH_TASK = "Template-Robotarm-Magnetic-Stomach-Lab-v0"
RESULT_ROOT = PROJECT_DIR / "logs" / "stomach_motion"


def run_stomach_cli(scenario: str) -> None:
    """Run one controller mode inside the horizontal stomach asset."""
    run_cli(
        scenario,
        default_task=STOMACH_TASK,
        result_root=RESULT_ROOT,
        environment_label="STOMACH",
        # Retain the stomach task's configured gastric-fluid angular drag.
        dry_surface=False,
    )
