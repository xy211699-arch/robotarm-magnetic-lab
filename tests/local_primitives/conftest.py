"""TASK-004 pure-test import setup."""

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(PACKAGE_ROOT))
