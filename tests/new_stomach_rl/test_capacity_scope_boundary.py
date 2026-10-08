"""CPU USD evidence of the archived whole-stage single-copy audit boundary."""
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ball_envelope import audit_stage


def add_copy(stage, row):
    root=f'/World/envs/env_{row}/Scene/asm'
    bodies=[]
    for name in ('magl','ballzl'):
        body=UsdGeom.Xform.Define(stage,root+'/'+name+'_body').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        bodies.append(body.GetPath())
        mesh=UsdGeom.Mesh.Define(stage,str(body.GetPath())+'/'+name)
        mesh.CreatePointsAttr([(0,0,0),(.01,0,0),(0,.01,0)])
        mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    for name in ('ballxj','ballyj','ballzj'):
        joint=UsdPhysics.RevoluteJoint.Define(stage,root+'/joints/'+name)
        joint.CreateBody0Rel().SetTargets([bodies[0]])
        joint.CreateBody1Rel().SetTargets([bodies[1]])
        joint.CreateLocalPos0Attr((0,0,0));joint.CreateLocalPos1Attr((0,0,0))


def test_original_audit_single_copy_unchanged():
    stage=Usd.Stage.CreateInMemory()
    add_copy(stage,0)
    assert audit_stage(stage)['joint_pivot_error_max_m']==0


def test_original_whole_stage_audit_refuses_identical_second_copy():
    stage=Usd.Stage.CreateInMemory()
    add_copy(stage,0);add_copy(stage,1)
    with pytest.raises(RuntimeError,match='single-environment rotating geometry audit'):
        audit_stage(stage)
