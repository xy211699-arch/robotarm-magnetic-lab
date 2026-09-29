"""Audit live static USD triangles against XRDF sphere unions without changing them."""
import numpy as np


def cover_triangles(triangles, centers, radii, padding=1e-6):
    """Add local spheres to enclose whole faces, not just their vertices.

    Each patch is assigned to one convex sphere. Original centers never move.
    This is deliberately conservative; false rejection is preferable to a gap.
    """
    triangles = np.asarray(triangles, float)
    centers = np.asarray(centers, float)
    result = np.asarray(radii, float).copy()
    if (not len(centers) or not len(triangles) or padding < 0 or
            not all(np.isfinite(x).all() for x in (triangles, centers, result))):
        raise ValueError('finite nonempty mesh and spheres required')
    # Split long CAD faces so one distant vertex cannot inflate a whole link.
    patches=[]
    for depth in range(12):
        edges=triangles-np.roll(triangles,1,axis=1)
        small=np.linalg.norm(edges,axis=-1).max(axis=1)<=.008
        patches.append(triangles[small])
        large=triangles[~small]
        if not len(large):break
        if depth==11 or sum(len(x) for x in patches)+4*len(large)>1000000:
            raise ValueError('mesh subdivision budget exceeded')
        a,b,c=large[:,0],large[:,1],large[:,2]
        ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
        triangles=np.concatenate([np.stack(v,axis=1) for v in
            ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))])
    triangles=np.concatenate(patches)
    uncovered=[]
    from scipy.spatial import cKDTree
    tree=cKDTree(centers)
    for offset in range(0, len(triangles), 128):
        tri = triangles[offset:offset + 128]
        # A missed large remote sphere only adds a redundant local sphere;
        # limiting this construction query cannot leave a patch uncovered.
        _,near=tree.query(tri.mean(axis=1),k=min(16,len(centers)))
        near=np.asarray(near).reshape(len(tri),-1)
        required=np.linalg.norm(tri[:,:,None,:]-centers[near][:,None,:,:],axis=-1).max(axis=1)
        uncovered.append(tri[np.min(required-result[near],axis=1)>0])
    missing=np.concatenate(uncovered)
    if not len(missing):return centers.copy(),result
    # Group nearby patches, avoiding large-radius expansion of remote centers.
    keys=np.floor(missing.mean(axis=1)/.008).astype(np.int64)
    _,groups=np.unique(keys,axis=0,return_inverse=True)
    added_c=[];added_r=[]
    for group in range(groups.max()+1):
        vertices=missing[groups==group].reshape(-1,3)
        center=(vertices.min(axis=0)+vertices.max(axis=0))/2
        added_c.append(center)
        added_r.append(np.linalg.norm(vertices-center,axis=1).max()+padding)
    return np.concatenate([centers,added_c]),np.concatenate([result,added_r])


def certify_triangles(triangles,centers,radii,tolerance=1e-6,max_depth=8):
    pending=np.asarray(triangles,float);centers=np.asarray(centers);radii=np.asarray(radii)
    from scipy.spatial import cKDTree
    tree=cKDTree(centers)
    count=len(pending);accepted=0
    for depth in range(max_depth+1):
        remaining=[]
        for offset in range(0,len(pending),128):
            tri=pending[offset:offset+128]
            # A covering sphere must contain the centroid. Query only possible
            # centers, then check all three vertices against the SAME sphere.
            candidates=tree.query_ball_point(tri.mean(axis=1),float(radii.max()+tolerance+1e-12))
            counts=np.fromiter(map(len,candidates),dtype=int)
            rows=np.repeat(np.arange(len(tri)),counts)
            best=np.full(len(tri),np.inf)
            if len(rows):
                ids=np.concatenate(candidates).astype(int)
                d=np.linalg.norm(tri[rows]-centers[ids,None,:],axis=-1).max(axis=1)-radii[ids]
                np.minimum.at(best,rows,d)
            covered=best<=tolerance
            accepted+=int(covered.sum())
            failed=tri[~covered]
            if len(failed):
                # Weighted nearest distance; nearest-center alone is incorrect
                # when radii differ. Use every potential containing sphere.
                vertices=failed.reshape(-1,3)
                neighbors=tree.query_ball_point(vertices,float(radii.max()+tolerance+1e-12))
                counts=np.fromiter(map(len,neighbors),dtype=int)
                rows=np.repeat(np.arange(len(vertices)),counts)
                gaps=np.full(len(vertices),np.inf)
                if len(rows):
                    ids=np.concatenate(neighbors).astype(int)
                    np.minimum.at(gaps,rows,np.linalg.norm(vertices[rows]-centers[ids],axis=-1)-radii[ids])
                if np.any(gaps>tolerance):
                    # Report an actual finite gap for the first uncovered point.
                    v=vertices[np.flatnonzero(gaps>tolerance)[0]]
                    gap=np.min(np.linalg.norm(v-centers,axis=1)-radii)
                    return dict(certified=False,reason='mesh_vertex_outside_sphere_union',
                        uncovered_vertex_gap_m=float(gap),initial_triangles=count,depth=depth)
                remaining.append(failed)
        if not remaining:return dict(certified=True,initial_triangles=count,certified_patches=accepted,depth=depth,tolerance_m=tolerance)
        pending=np.concatenate(remaining)
        if depth==max_depth or len(pending)*4>500000:
            return dict(certified=False,reason='subdivision_budget',unresolved_triangles=len(pending),depth=depth)
        a,b,c=pending[:,0],pending[:,1],pending[:,2]
        ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
        pending=np.concatenate([np.stack(v,axis=1) for v in ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))])


def audit_stage_static_meshes(stage,model,mount=None,repair_native=False):
    from pxr import Usd,UsdGeom,UsdPhysics
    cache=UsdGeom.XformCache();records=[];seen=set()
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):continue
        path=str(prim.GetPath());is_asm='/asm/' in path
        if is_asm and mount is None:continue
        if not is_asm and '/robotarm/' not in path:continue
        if is_asm and prim.GetName()!='base_link':continue  # rotating meshes audited separately
        body=prim
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if not body.IsValid():continue
        frame='l6' if is_asm else body.GetName()
        if frame not in model.spheres and frame not in model.environment_spheres:continue
        mesh=UsdGeom.Mesh(prim)
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
        if np.any(counts!=3):
            records.append(dict(path=path,certified=False,reason='nontriangular_mesh'));continue
        points=np.asarray(mesh.GetPointsAttr().Get(),float)
        relative=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
        points=(np.c_[points,np.ones(len(points))]@relative)[:,:3]
        if is_asm:points=points@np.asarray(mount['rotation']).T+mount['translation_m']
        faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
        for label,mapping in (('self',model.spheres),('world',model.environment_spheres)):
            if frame not in mapping or (not is_asm and frame=='l6' and label=='self'):continue
            c,r=mapping[frame]
            old_r=np.asarray(r).copy()
            if repair_native and not is_asm:
                c,r=cover_triangles(points[faces],c,r)
                mapping[frame]=(c,r)
            audit=certify_triangles(points[faces],c,r,tolerance=0)
            audit['maximum_radius_increase_m']=float(np.max(r[:len(old_r)]-old_r))
            audit['added_spheres']=len(r)-len(old_r)
            records.append(dict(path=path,frame=frame,geometry=label,**audit));seen.add((label,frame,'asm' if is_asm else 'arm'))
    expected={('self',frame,'arm') for frame in ('base_link','l1','l2','l3','l4','l5')}
    expected|={('world',frame,'arm') for frame in ('l2','l3','l4','l5','l6')}
    if mount is not None:expected|={('self','l6','asm'),('world','l6','asm')}
    if repair_native:
        for key in ('_asm_kd_tree','_gpu_spheres'):
            if hasattr(model,key):delattr(model,key)
    missing=sorted(expected-seen)
    return dict(certified=not missing and all(x['certified'] for x in records),
        missing_mesh_mappings=missing,records=records,
        scope='static robot/ASM triangles versus existing XRDF; rotating Ball sphere is a separate audit; not full self-collision CCD')
