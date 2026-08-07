from __future__ import annotations

from pathlib import Path
import sys


CONTROLLERS = (
    Path(__file__).resolve().parents[2]
    / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers"
)
sys.path.insert(0, str(CONTROLLERS))
