"""Import TASK-008 pure controller modules without installing the package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "source" / "robotarm_magnetic_lab"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
