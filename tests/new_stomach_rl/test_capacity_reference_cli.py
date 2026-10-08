"""CLI parsing only: no AppLauncher construction, Kit, GPU or simulation."""
from pathlib import Path
import subprocess
import sys


SCRIPT=Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl/capture_capacity_reference.py'


def test_help_does_not_require_assets_or_mode():
    result=subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert '--pose_manifest' in result.stdout and '--mode' in result.stdout
    assert 'Simulation App' not in result.stdout


def test_missing_inputs_fail_before_creating_simulation():
    result=subprocess.run([sys.executable,str(SCRIPT)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'required' in result.stderr or '必填' in result.stderr
    assert 'Simulation App' not in result.stdout
