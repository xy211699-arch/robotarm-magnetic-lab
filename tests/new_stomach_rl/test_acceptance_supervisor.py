import importlib.util
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('failed',[None,'single_verify','chunk_calibration','chunk_verify'])
def test_finite_serial_order_and_stop_on_failure_without_gpu(tmp_path,failed):
    script=Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl/supervise_isolation_acceptance.py'
    spec=importlib.util.spec_from_file_location('_acceptance_supervisor',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'launch.json').write_text(json.dumps(dict(single_manifest='existing-manifest',pose_manifest='poses',mask='mask')))
    calls=[]
    def job(command,log,state,folder):
        name=state['current_stage'];calls.append(name)
        assert '--device' in command and command[command.index('--device')+1]=='cuda:0'
        assert not any('train' in token for token in command)
        calibration=name=='chunk_calibration'
        data=dict(status='fail' if name==failed else 'calibrated' if calibration else 'pass',
            model_updates=0,devices={'gpu_dynamics':True},
            runs=[dict(global_batches=20)]*3 if calibration else [dict(global_batches=121,row_1_valid_transitions=120,row_1_timeout_count=1)]*2,
            comparisons=[{'passed':True}]*19,reset_invariants=[{'checks':{'unchanged':True}}]*7)
        return data,tmp_path/name/'summary.json'
    state=module.worker(tmp_path,job)
    expected=[x[0] for x in module.STEPS]
    assert calls==expected if failed is None else calls==expected[:expected.index(failed)+1]
    assert state['status']==('completed' if failed is None else 'paused_on_error')
    assert json.loads((tmp_path/'status.json').read_text())['status']==state['status']
