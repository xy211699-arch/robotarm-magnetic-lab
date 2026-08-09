"""Make the task worktree package importable without changing Isaac Lab's install."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))
for module_name in tuple(sys.modules):
    if module_name == "robotarm_magnetic_lab" or module_name.startswith("robotarm_magnetic_lab."):
        del sys.modules[module_name]
