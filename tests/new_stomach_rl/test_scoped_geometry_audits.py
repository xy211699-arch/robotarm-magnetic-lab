"""CPU USD scope regression; not a GPU/vector environment acceptance test."""
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import (
    ball_envelope, mesh_cover_audit, native_mesh_clearance,
)


MOUNT = dict(rotation=np.eye(3), translation_m=np.zeros(3))
BALL = (np.array([0., 0., .02]), .01)


def model():
    frames = ('base_link', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6')
    spheres = {name: (np.zeros((1, 3)), np.array([.02])) for name in frames}
    world = {name: (np.zeros((1, 3)), np.array([.02])) for name in frames[2:]}
    return SimpleNamespace(spheres=spheres, environment_spheres=world,
                           asm_frame='l6', ignored_frames=(), joints=range(6))


def add_mesh(stage, body_path, name):
    body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    box = trimesh.creation.box(extents=[.01, .01, .01])
    mesh = UsdGeom.Mesh.Define(stage, body_path + '/' + name)
    mesh.CreatePointsAttr(box.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(box.faces))
    mesh.CreateFaceVertexIndicesAttr(box.faces.ravel().tolist())
    return mesh


def add_copy(stage, row, offset=(0., 0., 0.)):
    root = f'/World/envs/env_{row}'
    UsdGeom.Xform.Define(stage, root).AddTranslateOp().Set(offset)
    for name in ('base_link', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6'):
        add_mesh(stage, root + '/Scene/robotarm/' + name, 'visual_mesh')
    add_mesh(stage, root + '/Scene/asm/base_link_body', 'base_link')
    bodies = []
    for name in ('magl', 'ballzl'):
        mesh = add_mesh(stage, root + '/Scene/asm/' + name + '_body', name)
        bodies.append(mesh.GetPrim().GetParent().GetPath())
    for name in ('ballxj', 'ballyj', 'ballzj'):
        joint = UsdPhysics.RevoluteJoint.Define(stage, root + '/Scene/asm/joints/' + name)
        joint.CreateBody0Rel().SetTargets([bodies[0]])
        joint.CreateBody1Rel().SetTargets([bodies[1]])
        joint.CreateLocalPos0Attr((0, 0, 0))
        joint.CreateLocalPos1Attr((0, 0, 0))
    return root


def call(kind, stage, root):
    if kind == 'ball':
        return ball_envelope.audit_stage(stage, env_root=root)
    if kind == 'static':
        return mesh_cover_audit.audit_stage_static_meshes(stage, model(), MOUNT, env_root=root)
    return native_mesh_clearance.NativeMeshClearance(
        stage, model(), asm_mount=MOUNT, ball_envelope=BALL, env_root=root)


@pytest.mark.parametrize('kind', ['ball', 'static', 'native'])
def test_default_none_matches_original_single_copy(kind):
    stage = Usd.Stage.CreateInMemory()
    root = add_copy(stage, 1)
    legacy = call(kind, stage, None)
    scoped = call(kind, stage, root)
    if kind == 'native':
        assert legacy.records == scoped.records
        assert legacy.asm_certificate == scoped.asm_certificate
        np.testing.assert_array_equal(legacy.centers, scoped.centers)
        np.testing.assert_array_equal(legacy.radii, scoped.radii)
    else:
        assert legacy == scoped
    if kind == 'static':
        assert scoped['certified']
    if kind == 'ball':
        assert scoped['joint_pivot_error_max_m'] == 0


@pytest.mark.parametrize('kind', ['ball', 'static', 'native'])
def test_env_1_does_not_include_env_10_and_row_order_does_not_matter(kind):
    results = []
    for order in ((1, 10), (10, 1)):
        stage = Usd.Stage.CreateInMemory()
        for row in order:
            add_copy(stage, row, (row * 10., 0., 0.))
        result = call(kind, stage, '/World/envs/env_1')
        records = result.records if kind == 'native' else result['meshes' if kind == 'ball' else 'records']
        assert records
        assert all(r['path'].startswith('/World/envs/env_1/') for r in records)
        assert not any('/env_10/' in r['path'] for r in records)
        results.append(records)
    assert results[0] == results[1]


@pytest.mark.parametrize('kind', ['ball', 'static', 'native'])
@pytest.mark.parametrize('root', ['/World/envs/missing', 'World/envs/env_1', '/', '/World/envs/env_1.attr'])
def test_invalid_scope_never_falls_back_to_whole_stage(kind, root):
    stage = Usd.Stage.CreateInMemory()
    add_copy(stage, 1)
    with pytest.raises(ValueError, match='env_root'):
        call(kind, stage, root)


def test_other_rows_cannot_fill_missing_mesh_or_joint():
    stage = Usd.Stage.CreateInMemory()
    add_copy(stage, 1)
    add_copy(stage, 10)
    stage.RemovePrim('/World/envs/env_1/Scene/asm/joints/ballxj')
    stage.RemovePrim('/World/envs/env_1/Scene/robotarm/l3/visual_mesh')
    with pytest.raises(RuntimeError, match='rotating geometry audit'):
        call('ball', stage, '/World/envs/env_1')
    static = call('static', stage, '/World/envs/env_1')
    assert not static['certified']
    assert ('self', 'l3', 'arm') in static['missing_mesh_mappings']
    with pytest.raises(ValueError, match='missing native meshes'):
        call('native', stage, '/World/envs/env_1')


def test_ball_joint_cannot_bind_to_another_row_even_at_same_world_pivot():
    stage = Usd.Stage.CreateInMemory()
    add_copy(stage, 1)
    add_copy(stage, 10)
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/World/envs/env_1/Scene/asm/joints/ballxj'))
    joint.GetBody0Rel().SetTargets(['/World/envs/env_10/Scene/asm/magl_body'])
    with pytest.raises(ValueError, match='outside env_root'):
        call('ball', stage, '/World/envs/env_1')


@pytest.mark.parametrize('kind', ['static', 'native'])
def test_scoped_mesh_cannot_belong_to_rigid_body_outside_scope(kind):
    stage = Usd.Stage.CreateInMemory()
    add_copy(stage, 1)
    root = '/World/envs/env_1'
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath('/World/envs'))
    body = stage.GetPrimAtPath(root + '/Scene/robotarm/l1')
    body.RemoveAPI(UsdPhysics.RigidBodyAPI)
    with pytest.raises(ValueError, match='outside env_root'):
        call(kind, stage, root)


def test_scoped_native_keeps_closed_mesh_and_ball_envelope_requirements():
    stage = Usd.Stage.CreateInMemory()
    root = add_copy(stage, 1)
    with pytest.raises(ValueError, match='rotating Ball envelope required'):
        native_mesh_clearance.NativeMeshClearance(stage, model(), asm_mount=MOUNT, env_root=root)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(root + '/Scene/robotarm/l1/visual_mesh'))
    faces = list(mesh.GetFaceVertexIndicesAttr().Get())[:-3]
    mesh.GetFaceVertexIndicesAttr().Set(faces)
    mesh.GetFaceVertexCountsAttr().Set([3] * (len(faces) // 3))
    with pytest.raises(ValueError, match='closed mesh'):
        call('native', stage, root)


def test_static_audit_does_not_ignore_a_bad_mesh_in_selected_row():
    stage = Usd.Stage.CreateInMemory()
    root = add_copy(stage, 1)
    add_copy(stage, 10)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(root + '/Scene/robotarm/l1/visual_mesh'))
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    bad = call('static', stage, root)
    good = call('static', stage, '/World/envs/env_10')
    assert not bad['certified']
    assert any(r.get('reason') == 'nontriangular_mesh' for r in bad['records'])
    assert good['certified']


def test_whole_stage_native_still_rejects_two_static_asm_copies():
    stage = Usd.Stage.CreateInMemory()
    add_copy(stage, 1)
    add_copy(stage, 10)
    with pytest.raises(ValueError, match='one static ASM mesh required'):
        native_mesh_clearance.NativeMeshClearance(stage, model(), asm_mount=MOUNT, ball_envelope=BALL)


@pytest.mark.parametrize('kind', ['ball', 'static', 'native'])
def test_instance_proxy_geometry_is_not_silently_skipped(kind):
    stage = Usd.Stage.CreateInMemory()
    add_copy(stage, 10)
    root = '/World/envs/env_1'
    copy = UsdGeom.Xform.Define(stage, root).GetPrim()
    copy.GetReferences().AddInternalReference('/World/envs/env_10')
    copy.SetInstanceable(True)
    assert stage.GetPrimAtPath(root + '/Scene/asm/magl_body/magl').IsInstanceProxy()
    result = call(kind, stage, root)
    records = result.records if kind == 'native' else result['meshes' if kind == 'ball' else 'records']
    assert records and all(r['path'].startswith(root + '/') for r in records)
    if kind == 'static':
        assert result['certified']


def test_missing_rotating_mesh_and_static_asm_are_not_supplied_by_other_row():
    stage = Usd.Stage.CreateInMemory()
    root = add_copy(stage, 1)
    add_copy(stage, 10)
    stage.RemovePrim(root + '/Scene/asm/magl_body/magl')
    stage.RemovePrim(root + '/Scene/asm/base_link_body/base_link')
    with pytest.raises(RuntimeError, match='rotating geometry audit'):
        call('ball', stage, root)
    assert not call('static', stage, root)['certified']
    with pytest.raises(ValueError, match='one static ASM mesh required'):
        call('native', stage, root)


def test_scoped_ball_still_rejects_different_joint_pivots():
    stage = Usd.Stage.CreateInMemory()
    root = add_copy(stage, 1)
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(root + '/Scene/asm/joints/ballxj'))
    joint.GetLocalPos0Attr().Set((.001, 0, 0))
    with pytest.raises(RuntimeError, match='do not share the audited source pivot'):
        call('ball', stage, root)


def test_default_and_scoped_repair_build_same_sphere_cover():
    stage = Usd.Stage.CreateInMemory()
    root = add_copy(stage, 1)
    models = [model(), model()]
    # Force an actual cover expansion while retaining the original algorithm.
    for m in models:
        for mapping in (m.spheres, m.environment_spheres):
            for name, (centers, radii) in mapping.items():
                mapping[name] = (centers, radii * .1)
    legacy = mesh_cover_audit.audit_stage_static_meshes(stage, models[0], MOUNT, repair_native=True)
    scoped = mesh_cover_audit.audit_stage_static_meshes(stage, models[1], MOUNT, repair_native=True, env_root=root)
    assert legacy == scoped
    for label in ('spheres', 'environment_spheres'):
        for name in getattr(models[0], label):
            for old, new in zip(getattr(models[0], label)[name], getattr(models[1], label)[name]):
                np.testing.assert_array_equal(old, new)


def test_scoped_native_preserves_signed_distance_and_numerical_cushion(monkeypatch):
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import batched_arm_clearance
    stage = Usd.Stage.CreateInMemory()
    root = add_copy(stage, 1)
    legacy = call('native', stage, None)
    scoped = call('native', stage, root)
    def transforms(m, q):
        result = {name: np.tile(np.eye(4), (len(q), 1, 1)) for name in m.spheres}
        result[m.asm_frame][:, 0, 3] = .1
        return result
    monkeypatch.setattr(batched_arm_clearance, 'link_transforms', transforms)
    q = np.zeros((2, 6))
    np.testing.assert_array_equal(legacy.minimum(q), scoped.minimum(q))
    mesh = scoped.meshes[0][1]
    values = mesh.distance([[0, 0, 0], [.01, 0, 0]])
    assert values[0] < 0  # closed-solid containment is still detected
    np.testing.assert_allclose(values[1], .005 - 1e-5, atol=1e-7)
