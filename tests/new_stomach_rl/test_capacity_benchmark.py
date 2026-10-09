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


@pytest.mark.parametrize('fail_at',[None,3])
def test_quick_plan_stops_at_global_budget_and_keeps_results(tmp_path,fail_at):
    module=load('supervise_capacity')
    import time
    (tmp_path/'launch.json').write_text(json.dumps(dict(pose_manifest='poses',mask='mask',quick=True,
        budget_seconds=7200,started_epoch_s=time.time(),deadline_epoch_s=time.time()+7200)))
    calls=[]
    def job(command,log,state,folder):
        assert '--quick' in command and '--screening' not in command
        if len(calls)==fail_at: raise module.CapacityBudgetExceeded('fixture deadline')
        n,g=state['current_num_envs'],state['current_group'];calls.append((n,g))
        return dict(status='screened',model_updates=0,global_batches=8,devices={'gpu_dynamics':True},
            partial_reset_checks=[] if n==1 else [{'unchanged':True}]),tmp_path/f'{n}_{g}'/'summary.json'
    state=module.worker(tmp_path,job)
    expected=[(n,g) for n,g,_ in module.QUICK_STEPS]
    assert calls==(expected if fail_at is None else expected[:fail_at])
    assert len(expected)==9 and expected[:6]==[(n,'D') for n in (1,4,8,12,16,20)]
    assert state['status']==('screening_completed' if fail_at is None else 'budget_exhausted')
    assert len(state['results'])==len(calls) and state['model_updates']==0
    bench=load('benchmark_capacity')
    assert bench.quick_schedule()==[('quick',2,6)]
    assert len(bench.QUICK_ACTION_INDICES)==8


def test_budget_reserves_shutdown_time_and_kills_stuck_child(monkeypatch):
    module=load('supervise_capacity')
    assert not module.budget_expiring({'_deadline_monotonic':100},69)
    assert module.budget_expiring({'_deadline_monotonic':100},70)
    assert not module.budget_expiring({},10000)
    import subprocess
    from types import SimpleNamespace
    signals=[]; waits=[]
    monkeypatch.setattr(module.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    def wait(timeout):
        waits.append(timeout)
        if len(waits)==1: raise subprocess.TimeoutExpired('fake',timeout)
    module.stop_child(SimpleNamespace(pid=123,wait=wait))
    assert signals==[(123,module.signal.SIGINT),(123,module.signal.SIGKILL)]
    assert waits==[20,2]


def test_run_job_enforces_deadline_while_child_still_running(tmp_path,monkeypatch):
    module=load('supervise_capacity')
    from types import SimpleNamespace
    fake=SimpleNamespace(pid=123,poll=lambda:None)
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:fake)
    monkeypatch.setattr(module,'stop_child',lambda child:None)
    checks=iter([False,True]);monkeypatch.setattr(module,'budget_expiring',lambda state:next(checks))
    state={'child_pid':None}
    with pytest.raises(module.CapacityBudgetExceeded): module.run_job(['fake'],tmp_path/'job.log',state,tmp_path)
    assert state['child_pid'] is None
