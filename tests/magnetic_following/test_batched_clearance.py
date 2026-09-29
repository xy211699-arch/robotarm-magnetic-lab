import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def test_batched_clearance_matches_original():
    kin = load('scalar_kin',ROOT/'action_layer/kinematics.py')
    batch = load('batch_kin',ROOT/'batched_arm_clearance.py')
    model = kin.UrdfXrdfSafetyModel('/home/multirobo/Desktop/robotarm/urdf/robotarm.urdf',
        '/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/data/planning/robot.xrdf')
    q = np.random.default_rng(913).uniform(-2,2,(100,6))
    actual = batch.clearance_by_frame(model,q)
    fast = batch.minimum_clearance_fast(model,q)
    for i, config in enumerate(q):
        for frame, expected in model.asm_clearance_by_frame(config).items():
            np.testing.assert_allclose(actual[frame][i],expected,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(fast[i],model.minimum_asm_clearance(config),atol=1e-12,rtol=1e-12)

def test_batched_world_preserves_all_query_points():
    from types import SimpleNamespace
    kin=load('world_scalar',ROOT/'action_layer/kinematics.py')
    batch=load('world_batch',ROOT/'batched_arm_clearance.py')
    model=kin.UrdfXrdfSafetyModel('/home/multirobo/Desktop/robotarm/urdf/robotarm.urdf',
        '/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/data/planning/robot.xrdf')
    q=np.random.default_rng(914).uniform(-1,1,(10,6));position=np.array([.2,.3,.4])
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    # Analytic query substitutes for the mesh kernel; geometry and transform
    # comparison is against the original scalar URDF world-sphere method.
    checker=SimpleNamespace(kinematics=model,base_position=position,base_rotation=rotation,
        _query=lambda p:(np.linalg.norm(p,axis=1),np.zeros(len(p))))
    expected=[]
    for row in q:
        sets=model.environment_world_spheres(row)
        p=np.concatenate([v[0] for v in sets.values()]);r=np.concatenate([v[1] for v in sets.values()])
        expected.append(np.min(np.linalg.norm(position+p@rotation.T,axis=1)-r))
    np.testing.assert_allclose(batch.minimum_world_clearance(checker,q),expected,atol=1e-12)
