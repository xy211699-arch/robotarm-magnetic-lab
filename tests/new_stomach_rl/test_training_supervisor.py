import importlib.util
import json
from pathlib import Path
import sys
import pytest


def module():
    p=Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl/supervise_training.py'
    spec=importlib.util.spec_from_file_location('_training_supervisor',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('case',['healthy','exit','missing','nan','oom','stale','stop'])
def test_fake_workers_pause_on_fault_without_training(tmp_path,case):
    m=module();script=tmp_path/'fake.py'
    script.write_text('''import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);case=sys.argv[2]
if case in ('exit','oom'):
 print('CUDA out of memory' if case=='oom' else 'fixture failure');sys.exit(3)
if case=='missing': sys.exit(0)
for k in range(4):
 h=dict(run_id=p.name,heartbeat_epoch_s=time.time(),stage='sampling',update_count=0,effective_samples=k,metrics={'loss':float('nan') if case=='nan' else 1.},latest_checkpoint=None)
 temp=p/'heartbeat.tmp';temp.write_text(json.dumps(h));temp.replace(p/'heartbeat.json')
 if case=='stop': (p/'stop.request').write_text('user requested')
 if case=='stale': time.sleep(2)
 time.sleep(.07)
(p/'summary.json').write_text(json.dumps(dict(run_id=p.name,status='completed',update_count=3,effective_samples=1536,global_boundaries=192)))
''')
    config=dict(max_wall_seconds=4,startup_timeout_seconds=1,heartbeat_timeout_seconds=.3,
        stop_grace_seconds=.1,max_updates=3,rollout_steps=64)
    result=m.monitor([sys.executable,str(script),str(tmp_path),case],tmp_path,config,poll=.01)
    assert result['status']==('completed' if case=='healthy' else 'interrupted' if case=='stop' else 'paused_on_error')
    assert result['child_pid'] is None
    assert (tmp_path/'worker.log').exists()
    # Querying state is strictly read-only.
    p=tmp_path/'status.json';before=p.stat().st_mtime_ns
    assert m.read(p)['status']==result['status'] and p.stat().st_mtime_ns==before


def test_formal_or_changed_evidence_cannot_launch(tmp_path):
    m=module();p=tmp_path/'config.json'
    p.write_text(json.dumps({'purpose':'formal','num_envs':8}))
    with pytest.raises(ValueError,match='formal'): m.validate_config(p)
    p.write_text(json.dumps(dict(purpose='r2_smoke',num_envs=8,max_updates=3,rollout_steps=64,group='A',
        max_wall_seconds=7200,startup_timeout_seconds=1200,heartbeat_timeout_seconds=180,stop_grace_seconds=10,
        evidence_files=[{'path':str(p),'sha256':'0'*64}])))
    with pytest.raises(ValueError,match='bytes changed'): m.validate_config(p)


def test_timeouts_use_measured_slowest_healthy_step(tmp_path):
    m=module();summary=tmp_path/'summary.json';steps=tmp_path/'boundaries.jsonl'
    summary.write_text(json.dumps(dict(num_envs=8,startup_build_wall_seconds=636.046)))
    steps.write_text(json.dumps(dict(elapsed_s=18.2))+'\n'+json.dumps(dict(elapsed_s=22.7))+'\n')
    limits=m.capacity_timeouts(summary,steps)
    assert limits['startup_timeout_seconds']==1909 and limits['heartbeat_timeout_seconds']==227
    assert len(limits['evidence_files'])==2
