"""Narrow phase: closed native arm triangles versus conservative ASM spheres.

No vertex-only mesh distance, no capsule state, and no asset modifications.
Only the native arm proxy is replaced; the audited ASM/Ball envelopes remain.
"""
import numpy as np
import warp as wp


@wp.kernel
def _signed_distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                     distance: wp.array(dtype=float), faces: wp.array(dtype=int)):
    i=wp.tid()
    query=wp.mesh_query_point_sign_winding_number(mesh,points[i],10.0)
    distance[i]=-1.0e6
    faces[i]=-1
    if query.result:
        closest=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
        distance[i]=query.sign*wp.length(closest-points[i])
        faces[i]=query.face


class ClosedMesh:
    def __init__(self,vertices,faces,device='cpu'):
        import trimesh
        v=np.asarray(vertices,float);f=np.asarray(faces,int)
        if not len(v) or not len(f) or not np.isfinite(v).all():
            raise ValueError('nonempty finite mesh required')
        mesh=trimesh.Trimesh(vertices=v,faces=f,process=True)
        if not mesh.is_watertight:
            raise ValueError('native narrow phase requires a closed mesh')
        # Orient the query copy consistently, without editing USD geometry.
        mesh.fix_normals(multibody=True)
        if not mesh.is_winding_consistent or mesh.volume<=0:
            raise ValueError('native mesh winding/volume invalid')
        self.vertices=np.asarray(mesh.vertices);self.faces=np.asarray(mesh.faces)
        self.device=str(device)
        self.mesh=wp.Mesh(points=wp.array(self.vertices.astype(np.float32),dtype=wp.vec3,device=device),
            indices=wp.array(self.faces.astype(np.int32).ravel(),dtype=int,device=device),
            support_winding_number=True)

    def distance(self,points):
        points=np.asarray(points,float).reshape(-1,3)
        if not np.isfinite(points).all():raise ValueError('nonfinite query points')
        values=wp.empty(len(points),dtype=float,device=self.device)
        faces=wp.empty(len(points),dtype=int,device=self.device)
        wp.launch(_signed_distance,dim=len(points),inputs=[self.mesh.id,
            wp.array(points.astype(np.float32),dtype=wp.vec3,device=self.device),values,faces],device=self.device)
        d=values.numpy().astype(float);face=faces.numpy()
        if np.any(face<0) or not np.isfinite(d).all():
            raise RuntimeError('native mesh query failed; cannot certify clearance')
        # Conservative numerical cushion in addition to the unchanged 5mm gap.
        return d-1e-5


class NativeMeshClearance:
    def __init__(self,stage,model,device='cpu',asm_mount=None,ball_envelope=None,env_root=None):
        from pxr import Usd,UsdGeom,UsdPhysics
        from .ball_envelope import _scope_prims, _require_body_scope
        self.model=model;self.meshes=[];self.records=[];self.query_count=0
        cache=UsdGeom.XformCache()
        expected=set(model.spheres)-{model.asm_frame}-set(model.ignored_frames)
        seen=set()
        for prim in _scope_prims(stage, env_root):
            if not prim.IsA(UsdGeom.Mesh) or '/robotarm/' not in str(prim.GetPath()):continue
            body=prim
            while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            if not body.IsValid():continue
            _require_body_scope(body.GetPath(), env_root, 'Native mesh')
            if body.GetName() not in expected:continue
            frame=body.GetName();mesh=UsdGeom.Mesh(prim)
            counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
            if not len(counts) or np.any(counts!=3):raise ValueError('native mesh must be triangular')
            points=np.asarray(mesh.GetPointsAttr().Get(),float)
            transform=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
            points=(np.c_[points,np.ones(len(points))]@transform)[:,:3]
            faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
            query=ClosedMesh(points,faces,device)
            self.meshes.append((frame,query));seen.add(frame)
            self.records.append(dict(path=str(prim.GetPath()),frame=frame,closed=True,
                triangles=len(query.faces),device=str(device)))
        if expected!=seen:raise ValueError(f'missing native meshes: {sorted(expected-seen)}')
        self.asm_mesh=None
        self.centers,self.radii=model.spheres[model.asm_frame]
        self.asm_certificate=None
        if asm_mount is not None:
            from .mesh_cover_audit import cover_triangles,certify_triangles
            selected=[p for p in _scope_prims(stage, env_root)
                if p.IsA(UsdGeom.Mesh) and '/asm/' in str(p.GetPath()) and p.GetName()=='base_link']
            if len(selected)!=1:raise ValueError('one static ASM mesh required')
            prim=selected[0];body=prim
            while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            if not body.IsValid():raise ValueError('ASM rigid body missing')
            _require_body_scope(body.GetPath(), env_root, 'ASM mesh')
            geometry=UsdGeom.Mesh(prim)
            if np.any(np.asarray(geometry.GetFaceVertexCountsAttr().Get())!=3):raise ValueError('ASM must be triangulated')
            vertices=np.asarray(geometry.GetPointsAttr().Get(),float)
            relative=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
            vertices=(np.c_[vertices,np.ones(len(vertices))]@relative)[:,:3]
            vertices=vertices@np.asarray(asm_mount['rotation']).T+asm_mount['translation_m']
            faces=np.asarray(geometry.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
            c,r=cover_triangles(vertices[faces],np.zeros((1,3)),np.zeros(1))
            c,r=c[1:],r[1:]  # discard construction sentinel, not real geometry
            self.asm_certificate=certify_triangles(vertices[faces],c,r,tolerance=0)
            if not self.asm_certificate['certified']:raise ValueError('tight ASM cover failed')
            self.asm_mesh=ClosedMesh(vertices,faces,device)
            if ball_envelope is None:raise ValueError('rotating Ball envelope required')
            self.centers=np.vstack([c,ball_envelope[0]])
            self.radii=np.r_[r,ball_envelope[1]]

    def minimum(self,configurations):
        from .batched_arm_clearance import link_transforms
        q=np.asarray(configurations,float).reshape(-1,len(self.model.joints))
        transforms=link_transforms(self.model,q)
        source=transforms[self.model.asm_frame]
        centers,radii=self.centers,self.radii
        output=np.full(len(q),np.inf)
        for frame,mesh in self.meshes:
            target=transforms[frame]
            rotation=np.swapaxes(target[:,:3,:3],1,2)@source[:,:3,:3]
            translation=np.einsum('nji,nj->ni',target[:,:3,:3],source[:,:3,3]-target[:,:3,3])
            local=np.einsum('nij,pj->npi',rotation,centers)+translation[:,None,:]
            distances=mesh.distance(local.reshape(-1,3)).reshape(len(q),-1)-radii[None,:]
            output=np.minimum(output,distances.min(axis=1))
            if self.asm_mesh is not None:
                # Surface envelopes alone would miss a native link fully
                # contained inside the ASM solid. Check the reverse direction.
                inverse=np.swapaxes(rotation,1,2)
                inside_points=np.einsum('nij,pj->npi',inverse,mesh.vertices)-np.einsum('nij,nj->ni',inverse,translation)[:,None,:]
                signs=self.asm_mesh.distance(inside_points.reshape(-1,3)).reshape(len(q),-1)
                contained=np.min(signs,axis=1)<0
                output[contained]=np.minimum(output[contained],np.min(signs[contained],axis=1))
        if not np.isfinite(output).all():raise RuntimeError('nonfinite narrow-phase clearance')
        self.query_count+=len(q)
        return output
