import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest
from test_full_training_entry import full_config, supervisor
from robotarm_magnetic_lab.runtime.finite_magnet_cache import cache_finite_geometry
from robotarm_magnetic_lab.runtime.finite_magnet_cache import SameTickPoseCache


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
