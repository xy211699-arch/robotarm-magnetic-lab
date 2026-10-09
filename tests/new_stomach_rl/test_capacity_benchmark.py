import importlib.util
import json
from pathlib import Path
import sys
import pytest


def load(name):
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'scripts/new_stomach_rl'))
    spec=importlib.util.spec_from_file_location('_'+name,root/'scripts/new_stomach_rl'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_fixed_budget_and_strict_physical_counts():
    module=load('benchmark_capacity')
    assert module.LEVELS==(1,4,8,12,16,20)
    assert sum(a+b for _,a,b in module.schedule())==224
    assert module.schedule()[-1][2]==144
    assert module.check_batch(240,[240,240],[True,False],[False,False])==1
    for args in [(239,[240],[True],[False]),(240,[239],[True],[False]),
                 (240,[240],[True],[True]),(240,[240],[],[False])]:
        with pytest.raises(RuntimeError): module.check_batch(*args)


@pytest.mark.parametrize('failure',[None,(4,'D',True),(1,'C',False),(4,'B',False)])
def test_serial_ladder_stop_on_serious_failure_without_kit(tmp_path,failure):
    module=load('supervise_capacity')
    (tmp_path/'launch.json').write_text(json.dumps(dict(pose_manifest='poses',mask='mask')))
    calls=[]
    def job(command,log,state,folder):
        key=(state['current_num_envs'],state['current_group'],state['screening_only']);calls.append(key)
        assert '--device' in command and command[command.index('--device')+1]=='cuda:0'
        assert not any('train' in x for x in command)
        n,g,screening=key
        assert ('--screening' in command)==screening
        data=dict(status='fail' if key==failure else 'screened' if screening else 'pass',model_updates=0,global_batches=8 if screening else 224,
            devices={'gpu_dynamics':True},timeouts_by_row=[1]*n,
            partial_reset_checks=[] if n==1 else [{'unchanged':True}])
        return data,tmp_path/f'{n}_{g}'/'summary.json'
    state=module.worker(tmp_path,job)
    expected=list(module.STEPS)
    assert calls==(expected if failure is None else expected[:expected.index(failure)+1])
    assert state['status']==('completed' if failure is None else 'paused_on_error')
    assert state['model_updates']==0
    assert len(state['not_run'])==(0 if failure is None else len(expected)-len(calls))


def test_no_unapproved_failure_can_be_deferred(tmp_path):
    module=load('supervise_capacity')
    single=tmp_path/'single.json';chunk=tmp_path/'chunk.json'
    single.write_text(json.dumps({'status':'pass','mode':'single','devices':{'gpu_dynamics':True},
        'comparisons':[{'passed':True}]*19,'reset_invariants':[{'checks':{'unchanged':True}}]*7}))
    chunk.write_text(json.dumps({'mode':'chunk','calibration_only':True,'error':{'type':'RuntimeError','message':'nonfinite'}}))
    with pytest.raises(ValueError,match='only the explicitly approved'): module.gate(single,chunk)
    data=json.loads(single.read_text());data['reset_invariants'][0]['checks']['unchanged']=False
    single.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Single'): module.gate(single,chunk)


def test_finalize_persists_result_before_kit_shutdown(tmp_path,monkeypatch,capsys):
    module=load('benchmark_capacity');events=[]
    from types import SimpleNamespace
    def persist(report,output,shutdown):
        assert report['status']=='fail' and output==tmp_path
        events.append('persist');shutdown()
    monkeypatch.setattr(module,'persist_before_shutdown',persist)
    module.finalize({'status':'fail'},tmp_path,
        SimpleNamespace(close=lambda:events.append('env_close')),
        SimpleNamespace(app=SimpleNamespace(close=lambda:events.append('app_close'))))
    assert events==['persist','env_close','app_close']
    assert 'CAPACITY_RESULT' in capsys.readouterr().out
