import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

PATH=Path(__file__).resolve().parents[2]/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/actuator_vector.py'
spec=importlib.util.spec_from_file_location('actuator_vector',PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_chunk_shape_finite_and_norm_limits():
    for a in (np.zeros(9),np.zeros((3,9)),np.full((4,9),np.nan)):
        with pytest.raises(ValueError): m.decode(a)
    a,d=m.decode(np.full((4,9),5.))
    np.testing.assert_allclose(np.linalg.norm(a[:,:3],axis=1),1)
    np.testing.assert_allclose(d,a*m.SCALES)

def test_all_nine_axes_enabled():
    for i in range(9):
        a=np.zeros((4,9));a[0,i]=1
        _,d=m.decode(a)
        assert d[0,i]>0 and np.count_nonzero(d)==1

def test_goals_cumulative_base_translation_local_rotation():
    r=Rotation.from_euler('x',.4).as_matrix()
    base=Rotation.from_euler('z',np.pi/2).as_matrix()
    d=np.zeros((4,9));d[:,0]=.001;d[:,4]=.02;d[:,8]=.1
    g=m.goals(np.zeros(3),r,np.zeros(3),d,base)
    for i,(p,rr,b) in enumerate(g,1):
        np.testing.assert_allclose(p,[0,i*.001,0],atol=1e-12)
        np.testing.assert_allclose(rr,r@Rotation.from_rotvec([0,i*.02,0]).as_matrix(),atol=1e-12)
        np.testing.assert_allclose(b,[0,0,i*.1])

def test_quintic_endpoints_and_reversal_continuity():
    knots=np.zeros((5,9));knots[1]=.02;knots[3]=-.02
    q,v,a=m.quintic(knots)
    assert q.shape==(961,9)
    np.testing.assert_allclose(q[::240],knots,atol=1e-12)
    np.testing.assert_allclose(v[::240],0,atol=1e-12)
    np.testing.assert_allclose(a[::240],0,atol=1e-12)
    assert np.max(np.abs(v))<=.12 and np.max(np.abs(a))<=.24

def test_plan_first_second_and_future_collision():
    def fk(q): return q[:3],Rotation.from_rotvec(q[3:]).as_matrix()
    d=np.zeros((4,9));d[:,0]=.001;d[:,6]=.02
    targets=m.goals(np.zeros(3),np.eye(3),np.zeros(3),d,np.eye(3))
    result=m.plan(np.zeros(9),fk,targets,np.tile([-3.,3.],(9,1)),lambda q:.1)
    assert len(result['q'][:241])==241
    np.testing.assert_allclose(result['q'][240,0],.001,atol=1e-8)
    np.testing.assert_allclose(result['q'][-1,0],.004,atol=1e-8)
    with pytest.raises(m.Infeasible,match='clearance'):
        m.plan(np.zeros(9),fk,targets,np.tile([-3.,3.],(9,1)),lambda q:0 if q[-1,0]>.003 else .1)

def test_ball_acceleration_limit_cannot_be_bypassed():
    def fk(q): return q[:3],Rotation.from_rotvec(q[3:]).as_matrix()
    d=np.zeros((4,9));d[0,6]=.4
    targets=m.goals(np.zeros(3),np.eye(3),np.zeros(3),d,np.eye(3))
    with pytest.raises(m.Infeasible,match='velocity or acceleration'):
        m.plan(np.zeros(9),fk,targets,np.tile([-10.,10.],(9,1)),lambda q:.1)


def test_clearance_bound_is_conservative_under_rotation_and_translation():
    rng=np.random.default_rng(914)
    centers=rng.normal(size=(40,3))*.1
    initial=np.eye(4)
    transforms=np.tile(initial,(100,1,1))
    transforms[:,:3,:3]=Rotation.from_rotvec(rng.normal(size=(100,3))*.1).as_matrix()
    transforms[:,:3,3]=rng.normal(size=(100,3))*.001
    moved=np.einsum('nij,pj->npi',transforms[:,:3,:3],centers)+transforms[:,None,:3,3]
    actual_max=np.linalg.norm(moved-centers,axis=2).max()
    bound=m.clearance_lower_bound({'ee':transforms},{'ee':initial},
        ({'ee':(centers,np.zeros(40))},),1.,float('inf'))
    assert bound<=1-2*actual_max
