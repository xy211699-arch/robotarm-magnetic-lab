import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import trimesh

PATH=Path(__file__).resolve().parents[2]/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/native_mesh_clearance.py'
spec=importlib.util.spec_from_file_location('native_mesh_test',PATH)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)

def test_signed_triangle_distance_includes_containment_and_face_interior():
    box=trimesh.creation.box(extents=[.02,.02,.02])
    query=module.ClosedMesh(box.vertices,box.faces)
    d=query.distance([[0,0,0],[.02,0,0],[.011,0,0],[.01,0,0]])
    assert d[0]<-.0099
    np.testing.assert_allclose(d[1:],[.01-1e-5,.001-1e-5,-1e-5],atol=1e-7)
    # Sphere touching the middle of a face is detected even far from vertices.
    assert d[2]-.002<0

def test_open_mesh_and_nonfinite_query_rejected():
    box=trimesh.creation.box()
    with pytest.raises(ValueError):module.ClosedMesh(box.vertices,box.faces[:-1])
    query=module.ClosedMesh(box.vertices,box.faces)
    with pytest.raises(ValueError):query.distance([[np.nan,0,0]])

def test_triangle_dispatch_does_not_skip_containment_for_separated_surfaces(monkeypatch):
    path=PATH.with_name('batched_arm_clearance.py')
    spec=importlib.util.spec_from_file_location('narrow_dispatch_test',path)
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    class Narrow:
        def minimum(self,q):
            assert len(q)==3
            return np.array([-.020,.010,-.003])
    class Model:
        joints=range(6)
        _native_narrow=Narrow()
    monkeypatch.setattr(b,'minimum_clearance_spheres',lambda m,q:np.array([.020,-.010,-.012]))
    np.testing.assert_allclose(b.minimum_clearance_fast(Model(),np.zeros((3,6))),[-.020,.010,-.003])

def test_query_failure_is_not_treated_as_clear(monkeypatch):
    box=trimesh.creation.box()
    query=module.ClosedMesh(box.vertices,box.faces)
    # No nearest triangle within the fixed search range must fail closed.
    with pytest.raises(RuntimeError):query.distance([[100,0,0]])


def test_native_solid_fully_inside_asm_is_rejected(monkeypatch):
    # No surface intersection: checking only ASM sphere centers would miss it.
    big=trimesh.creation.box(extents=[.2,.2,.2])
    small=trimesh.creation.box(extents=[.01,.01,.01])
    native=module.ClosedMesh(small.vertices,small.faces)
    asm=module.ClosedMesh(big.vertices,big.faces)
    package=types.ModuleType('native_mesh_fixture');package.__path__=[]
    batch=types.ModuleType('native_mesh_fixture.batched_arm_clearance')
    batch.link_transforms=lambda model,q:{k:np.tile(np.eye(4),(len(q),1,1)) for k in ('arm','asm')}
    monkeypatch.setitem(sys.modules,'native_mesh_fixture',package)
    monkeypatch.setitem(sys.modules,'native_mesh_fixture.batched_arm_clearance',batch)
    monkeypatch.setattr(module,'__package__','native_mesh_fixture')
    model=types.SimpleNamespace(joints=range(6),asm_frame='asm')
    query=module.NativeMeshClearance.__new__(module.NativeMeshClearance)
    query.model=model;query.meshes=[('arm',native)];query.asm_mesh=asm;query.query_count=0
    query.centers=np.asarray(big.vertices);query.radii=np.full(len(big.vertices),.001)
    assert np.all(native.distance(query.centers)-query.radii>.005)
    assert query.minimum(np.zeros((1,6)))[0]<0
