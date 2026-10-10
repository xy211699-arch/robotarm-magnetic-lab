import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pytest
from test_full_training_entry import full_config, supervisor
from robotarm_magnetic_lab.runtime.finite_magnet_cache import cache_finite_geometry
from robotarm_magnetic_lab.runtime.finite_magnet_cache import SameTickPoseCache


def bounded_config(tmp_path):
    cfg=full_config(tmp_path)
    cfg.update(max_updates=200,rollout_steps=8,sequence_length=8,max_wall_seconds=43200,
        approval='user_confirmed_2026-10-10_200x8_N8_total12h_save50')
    return cfg


@pytest.mark.parametrize('group',list('ABCD'))
def test_final_four_group_200x8_budget(tmp_path,group):
    cfg=bounded_config(tmp_path);cfg['group']=group
    p=tmp_path/'final.json';p.write_text(json.dumps(cfg))
    checked=supervisor().validate_config(p)
    assert (checked['max_updates'],checked['rollout_steps'],checked['sequence_length'],checked['max_wall_seconds'])==(200,8,8,43200)


@pytest.mark.parametrize('field,value',[('max_updates',201),('rollout_steps',32),('sequence_length',32),
    ('max_wall_seconds',None),('max_wall_seconds',43201),('save_interval',10),('num_envs',12)])
def test_final_budget_is_not_silently_changed(tmp_path,field,value):
    cfg=bounded_config(tmp_path);cfg[field]=value
    p=tmp_path/'bad.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):supervisor().validate_config(p)


def test_budget_expiry_retains_data_and_consumes_previous_run_time(tmp_path):
    m=supervisor();cfg=bounded_config(tmp_path)
    # CPU fixture scales twelve hours to seconds. No Kit/GPU/learning process.
    cfg.update(max_wall_seconds=2.5,stop_grace_seconds=.1,startup_timeout_seconds=10,heartbeat_timeout_seconds=10)
    evidence=tmp_path/'boundaries.jsonl';evidence.write_text('existing evidence\n')
    start=time.monotonic()
    result=m.monitor([sys.executable,'-c','import time;time.sleep(60)'],tmp_path,cfg,poll=.01,budget_used_seconds=.2)
    assert result['status']=='interrupted' and result['budget_exhausted']
    assert result['stop_reason']=='twelve-hour group budget reached'
    assert result['child_pid'] is None and time.monotonic()-start<2.5
    assert result['group_elapsed_seconds']>=result['elapsed_seconds']+.19
    assert evidence.read_text()=='existing evidence\n'


def test_fake_200_update_completion_and_checkpoint_schedule(tmp_path):
    m=supervisor();cfg=bounded_config(tmp_path)
    script=tmp_path/'fixture.py'
    script.write_text('''import json,pathlib,hashlib,sys
p=pathlib.Path(sys.argv[1]); ck=p/'update_0200.pt';ck.write_bytes(b'CPU fixture only')
(p/'summary.json').write_text(json.dumps(dict(run_id=p.name,status='completed',update_count=200,
 effective_samples=12800,global_boundaries=1600,checkpoint_update=200,
 checkpoint=dict(path=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),kind='training_boundary'))))
''')
    result=m.monitor([sys.executable,str(script),str(tmp_path)],tmp_path,cfg,poll=.01)
    assert result['status']=='completed' and result['update_count']==200
    assert [u for u in range(1,201) if m.checkpoint_due(u,50)]==[50,100,150,200]


def test_pose_cache_only_reuses_same_tick_and_invalidates_on_reset():
    c=SameTickPoseCache();calls=[]
    def fetch():calls.append(1);return object()
    first=c.read(24,fetch)
    assert all(c.read(24,fetch) is first for _ in range(8))
    assert len(calls)==1
    assert c.read(25,fetch) is not first
    c.invalidate();c.read(25,fetch)
    assert len(calls)==3


@pytest.mark.parametrize('probe',[False,True])
def test_approved_500x32_and_bounded_probe(tmp_path,probe):
    cfg=full_config(tmp_path)
    cfg.update(max_updates=500,rollout_steps=32,sequence_length=32,
        approval='user_confirmed_2026-10-10_500x32_N8_save50',
        max_wall_seconds=900 if probe else None,time_limited_probe=probe)
    p=tmp_path/'cfg.json';p.write_text(json.dumps(cfg))
    assert supervisor().validate_config(p)['rollout_steps']==32
    cfg['max_wall_seconds']=901;cfg['time_limited_probe']=True;p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):supervisor().validate_config(p)


def test_cached_finite_mesh_matches_original_repeated_poses():
    sys.path.append(os.environ['ROBOTARM_MAGPYLIB_VENDOR'])
    root=Path(__file__).resolve().parents[2]
    path=root/'dependencies/legacy_magnetic_sim/robotarm/magnetic_sim/magnetics/field_models.py'
    spec=importlib.util.spec_from_file_location('original_finite_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=json.loads((root/'dependencies/legacy_magnetic_sim/data/config/default.json').read_text())
    original=module.FiniteMagnetSystem(cfg)
    cached=cache_finite_geometry(module.FiniteMagnetSystem(cfg))
    rng=np.random.default_rng(1008)
    from scipy.spatial.transform import Rotation
    for _ in range(12):
        p=rng.uniform(-.02,.02,3);q=p+np.array([.01,.02,.13])
        a,b=Rotation.random(2,random_state=rng).as_matrix()
        for name,args in (('force_torque_si',(p,a,q,b)),('force_torque_on_cube_si',(q,b,p,a))):
            actual=getattr(cached,name)(*args);expected=getattr(original,name)(*args)
            np.testing.assert_array_equal(actual,expected)
    assert len(cached._geometry_cache)==3
