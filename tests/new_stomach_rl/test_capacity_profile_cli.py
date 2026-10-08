"""CLI-only coverage, without constructing AppLauncher or invoking simulation."""
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl/profile_capacity.py'


def test_run_help_and_missing_inputs_do_not_launch_kit():
    result = subprocess.run([sys.executable,str(SCRIPT),'run','--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--timing' in result.stdout and '--tolerance_manifest' in result.stdout
    assert 'Simulation App' not in result.stdout
    result = subprocess.run([sys.executable,str(SCRIPT),'run'], capture_output=True, text=True)
    assert result.returncode != 0
    assert '必填' in result.stderr
    assert 'Simulation App' not in result.stdout


def test_audit_help_requires_no_isaac_application():
    result = subprocess.run([sys.executable,str(SCRIPT),'audit','--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--single_reference' in result.stdout and '--chunk_reference' in result.stdout
    assert 'Simulation App' not in result.stdout
