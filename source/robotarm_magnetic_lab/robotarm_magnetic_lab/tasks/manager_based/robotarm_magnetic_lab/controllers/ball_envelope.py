"""Audit actual USD rotating surfaces and conservatively cover all orientations."""
import hashlib
import numpy as np


def stomach_collision_mesh(stage, configured_path=None):
    from pxr import Usd,UsdGeom,UsdPhysics
    candidates=[]
    for p in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if '/Stomach/' in str(p.GetPath()) and p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False:
                candidates.append(str(p.GetPath()))
    if configured_path is not None:
        configured_path=str(configured_path)
        if candidates != [configured_path]:
            raise RuntimeError(
                f'configured stomach collision mesh mismatch: expected {configured_path}, got {candidates}'
            )
    if len(candidates)!=1:raise RuntimeError(f'one enabled stomach collision mesh required: {candidates}')
    return candidates[0]


def _scope_prims(stage, env_root=None):
    """Keep legacy traversal, or visit exactly one validated USD prim subtree."""
    from pxr import Sdf, Usd
    if env_root is None:
        return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    path = Sdf.Path(str(env_root))
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError(f'env_root must be an absolute non-root prim path: {env_root}')
    root = stage.GetPrimAtPath(path)
    if not root.IsValid() or not root.IsActive() or not root.IsDefined():
        raise ValueError(f'env_root must identify an active defined prim: {env_root}')
    return Usd.PrimRange(root, Usd.TraverseInstanceProxies())


def _require_body_scope(body_path, env_root, label):
    """Reject cross-row ownership/bindings, including env_1 versus env_10."""
    if env_root is not None:
        from pxr import Sdf
        if not Sdf.Path(str(body_path)).HasPrefix(Sdf.Path(str(env_root))):
            raise ValueError(f'{label} body {body_path} is outside env_root {env_root}')


def audit_stage(stage, env_root=None):
    """Audit one explicit subtree; None retains the whole-stage single-copy guard."""
    from pxr import Usd,UsdGeom,UsdPhysics
    cache=UsdGeom.XformCache()
    meshes=[];joints=[]
    for p in _scope_prims(stage, env_root):
        path=str(p.GetPath())
        if '/asm/' not in path:continue
        if p.IsA(UsdGeom.Mesh) and p.GetName() in ('magl','ballzl'):meshes.append(p)
        if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in ('ballxj','ballyj','ballzj'):joints.append(p)
    if len(meshes)!=2 or len(joints)!=3:raise RuntimeError('single-environment rotating geometry audit failed')
    mag=next(p for p in meshes if p.GetName()=='magl')
    pivot=np.asarray(cache.GetLocalToWorldTransform(mag.GetParent()))[3,:3]
    joint_errors=[]
    for p in joints:
        j=UsdPhysics.RevoluteJoint(p)
        for bodies,local in ((j.GetBody0Rel().GetTargets(),j.GetLocalPos0Attr().Get()),
                             (j.GetBody1Rel().GetTargets(),j.GetLocalPos1Attr().Get())):
            if len(bodies)!=1:raise RuntimeError('Ball joint must have two identified bodies')
            _require_body_scope(bodies[0], env_root, 'Ball joint')
            if env_root is not None:
                body = stage.GetPrimAtPath(bodies[0])
                if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError('Ball joint must bind an existing scoped rigid body')
            mat=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(bodies[0])))
            point=np.r_[np.asarray(local),1.]@mat
            joint_errors.append(float(np.linalg.norm(point[:3]-pivot)))
    if max(joint_errors)>1e-5:raise RuntimeError('Ball axes do not share the audited source pivot')
    radius=0.;records=[]
    for p in meshes:
        mesh=UsdGeom.Mesh(p);pts=np.asarray(mesh.GetPointsAttr().Get(),float)
        mat=np.asarray(cache.GetLocalToWorldTransform(p))
        world=np.c_[pts,np.ones(len(pts))]@mat
        r=float(np.linalg.norm(world[:,:3]-pivot,axis=1).max());radius=max(radius,r)
        digest=hashlib.sha256(pts.tobytes()+np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int64).tobytes()+mat.tobytes()).hexdigest()
        records.append(dict(path=str(p.GetPath()),vertices=len(pts),radius_m=r,geometry_sha256=digest))
    # The sphere is convex: all polygon triangles with these vertices are
    # contained. Rigid rotation about the common pivot preserves the radius.
    return dict(radius_m=radius+1e-6,meshes=records,joint_pivot_error_max_m=max(joint_errors),
                proof='convex sphere contains mesh vertices and faces for any rotation about common pivot')


def install(model,center,radius):
    for mapping in (model.spheres,model.environment_spheres):
        points,radii=mapping[model.asm_frame]
        mapping[model.asm_frame]=(np.vstack([points,center]),np.r_[radii,radius])
    if hasattr(model,'_asm_kd_tree'):del model._asm_kd_tree
