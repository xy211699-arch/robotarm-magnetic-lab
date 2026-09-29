import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest

root=Path(__file__).resolve().parents[2]/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/baselines'
package=types.ModuleType('escape_test');package.__path__=[str(root)]
sys.modules['escape_test']=package
spec=importlib.util.spec_from_file_location('escape_test.magnetic_roll_escape',root/'magnetic_roll_escape.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.mark.parametrize('force',[.010,.015,.020,.030])
def test_tangent_force_compensates_gravity_without_scaling_normal(force):
    normal=np.array([0.,.2,1.]);normal/=np.linalg.norm(normal)
    tangent=np.cross(normal,[1.,0,0]);tangent/=np.linalg.norm(tangent)
    weight=.005735*9.81
    F=m.force_target(tangent,normal,weight,force)
    assert np.isclose(F@normal,.012)
    assert np.isclose((F+[0,0,-weight])@tangent,force)


def test_bounds_and_unapproved_force_rejected():
    assert np.all(m.source_bounds_residual([0,0,.12],np.zeros(3))==0)
    assert np.any(m.source_bounds_residual([0,0,.119],np.zeros(3))>0)
    with pytest.raises(ValueError):m.force_target(np.array([1,0,0]),np.array([0,0,1]),.056,.025)


def test_area_scan_multiple_headings_continuous_and_tangent():
    normal=np.array([0.,0.,1.]);scan=m.AreaScanHeading([1.,0.,0.],normal)
    axes=[];angles=[]
    for _ in range(260):
        axis,record=scan.advance(normal)
        axes.append(axis);angles.append(record['command_heading_deg'])
        assert abs(axis@normal)<1e-12
        assert np.isclose(np.linalg.norm(axis),1.)
    assert np.max(np.abs(np.diff(angles)))<=2.
    assert min(angles)<-60 and max(angles)>=120
    assert np.linalg.matrix_rank(np.asarray(axes)[:,:2])==2


def test_area_scan_transports_reference_to_curved_surface():
    scan=m.AreaScanHeading([1.,0.,0.],[0.,0.,1.])
    for theta in np.linspace(0,.4,120):
        normal=np.array([0.,np.sin(theta),np.cos(theta)])
        axis,_=scan.advance(normal)
        assert np.isfinite(axis).all()
        assert abs(axis@normal)<1e-12


@pytest.mark.parametrize('own,world,reject',[(.025,.006,False),(.004,.006,True),(.025,.004,True),(np.nan,.006,True),(.025,np.nan,True)])
def test_area_termination_uses_audited_geometry_and_fails_closed(monkeypatch,own,world,reject):
    import ast
    import torch
    from types import SimpleNamespace as NS
    script=Path(__file__).resolve().parents[2]/'scripts/magnetic_following/screen_vector_random.py'
    node=next(n for n in ast.parse(script.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='audited_collision')
    dependency='robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.batched_arm_clearance'
    monkeypatch.setitem(sys.modules,dependency,NS(minimum_clearance_fast=lambda *a:np.array([own]),minimum_world_clearance=lambda *a:np.array([world])))
    term=NS(_geometry=lambda:None,robot=NS(data=NS(joint_pos=NS(torch=torch.zeros((1,9))))),ids=list(range(9)),
        kinematics=None,world_checker=None,cfg=NS(required_clearance_m=.005))
    env=NS(action_manager=NS(get_term=lambda name:term),device='cpu',_legacy_bridge_state={'collision':torch.tensor([[True]])})
    namespace={'np':np,'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(script),'exec'),namespace)
    assert bool(namespace['audited_collision'](env)[0]) is reject
    assert env._screen_clearance['legacy_collision'] is True


@pytest.mark.parametrize('x',[np.zeros(9),np.linspace(-.006,.006,9)])
def test_batched_geometry_jacobian_equals_scalar_at_interior_and_bounds(x):
    bounds=np.full(9,.006);start=np.linspace(-.01,.01,9);calls=[]
    def gaps(points):
        calls.append(len(points));return .0055+np.sin(points[:,:6]).sum(axis=1)*.01
    def residual(p,clearance_value=None):
        if clearance_value is None:clearance_value=gaps((start+p)[None])[0]
        return np.r_[np.sin(p),p*p,10*max(0,.006-clearance_value)/.001]
    expected=m.bounded_jacobian(residual,x,bounds)
    calls.clear()
    actual=m.batched_clearance_jacobian(residual,gaps,x,bounds,start)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    assert calls==[18]


def test_legacy_snapshot_scalar_safe_load(tmp_path):
    import torch
    file=tmp_path/'state.pt'
    torch.save({'probe':{'distance':np.float64(.2)},'velocity':torch.ones(6)},file)
    with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,type(np.dtype('float64'))]):
        result=torch.load(file,weights_only=True)
    assert result['probe']['distance']==.2
    assert torch.equal(result['velocity'],torch.ones(6))


def test_mesh_audit_does_not_assume_vertex_union_covers_face():
    path=root.parent/'tasks/manager_based/robotarm_magnetic_lab/controllers/mesh_cover_audit.py'
    spec=importlib.util.spec_from_file_location('mesh_cover_test',path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    triangle=np.array([[[1.,0,0],[-1,0,0],[0,1,0]]])
    result=audit.certify_triangles(triangle,triangle[0],np.array([.2,.2,.2]))
    assert result['certified'] is False
    assert audit.certify_triangles(triangle,np.zeros((1,3)),np.array([1.01]))['certified'] is True


def test_repair_covers_face_interior_and_never_shrinks():
    path=root.parent/'tasks/manager_based/robotarm_magnetic_lab/controllers/mesh_cover_audit.py'
    spec=importlib.util.spec_from_file_location('repair_test',path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    tri=np.array([[[.02,0,0],[-.02,0,0],[0,.02,0]]])
    centers=tri[0].copy();radii=np.full(3,.002)
    c,r=audit.cover_triangles(tri,centers,radii)
    assert np.array_equal(c[:len(centers)],centers)
    assert np.all(r[:len(radii)]>=radii)
    assert audit.certify_triangles(tri,c,r,tolerance=0)['certified']
    assert np.all(radii==.002)
    with pytest.raises(ValueError):audit.cover_triangles(tri*np.nan,centers,radii)
