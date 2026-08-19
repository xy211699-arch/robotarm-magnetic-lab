from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/robotarm_magnetic_lab"
CONTROLLERS = SOURCE / "robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(CONTROLLERS))
os.environ.setdefault(
    "ROBOTARM_MAGPYLIB_VENDOR",
    "/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor",
)
