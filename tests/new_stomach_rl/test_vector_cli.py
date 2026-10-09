import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np


def test_vector_help_and_bad_count_exit_before_kit():
    script=Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl/validate_vector_isolation.py'
    result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--check_inputs' in result.stdout and '--tolerance_manifest' in result.stdout
    result=subprocess.run([sys.executable,str(script),'--num_envs','4'],capture_output=True,text=True)
    assert result.returncode==2 and 'P2' in result.stderr
    assert 'Simulation App Startup' not in result.stdout


def test_comparison_cannot_hide_shapes_nan_or_nonzero_error():
    scripts=Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('_p2_compare',scripts/'validate_vector_isolation.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        x=np.zeros((2,9))
        assert module.compare_arrays(x,x,0.,'same')['passed']
        assert not module.compare_arrays(x,np.ones_like(x)*1e-12,0.,'tiny')['passed']
        assert not module.compare_arrays(x,x[:1],0.,'shape')['passed']
        assert not module.compare_arrays(x,np.full_like(x,np.nan),0.,'nan')['passed']
    finally:
        sys.path.pop(0)
