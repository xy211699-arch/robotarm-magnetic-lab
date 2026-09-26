"""Make the linked worktree package visible ahead of the main editable install."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
PACKAGE_DIR = PACKAGE_ROOT / "robotarm_magnetic_lab"

sys.path.insert(0, str(PACKAGE_ROOT))

# Isaac Lab's startup imports the editable installation from the main checkout
# before pytest collects this directory. Extend that package's search path so
# submodules created in the linked worktree are the versions under test.
import robotarm_magnetic_lab

package_path = str(PACKAGE_DIR)
if package_path not in robotarm_magnetic_lab.__path__:
    robotarm_magnetic_lab.__path__.insert(0, package_path)

import robotarm_magnetic_lab.tasks
import robotarm_magnetic_lab.tasks.manager_based
import robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab

for package, relative in (
    (robotarm_magnetic_lab.tasks, "tasks"),
    (robotarm_magnetic_lab.tasks.manager_based, "tasks/manager_based"),
    (
        robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab,
        "tasks/manager_based/robotarm_magnetic_lab",
    ),
):
    candidate = str(PACKAGE_DIR / relative)
    if candidate not in package.__path__:
        package.__path__.insert(0, candidate)
