"""Vectorized version of the existing XRDF ASM clearance, same geometry."""
import numpy as np


def link_transforms(model, configurations):
    q = np.asarray(configurations, float).reshape(-1, len(model.joints))
    transforms = {model.joints[0].parent: np.tile(np.eye(4), (len(q),1,1))}
    for index, joint in enumerate(model.joints):
        axis = np.asarray(joint.axis, float)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        x,y,z = axis
        skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        a = q[:,index,None,None]
        r = np.cos(a)*np.eye(3) + (1-np.cos(a))*np.outer(axis,axis) + np.sin(a)*skew
        motion = np.tile(np.eye(4), (len(q),1,1))
        motion[:,:3,:3] = r
        transforms[joint.child] = transforms[joint.parent] @ joint.origin @ motion
    return transforms


def clearance_by_frame(model, configurations):
    transforms = link_transforms(model, configurations)
    spheres = {}
    for frame,(centers,radii) in model.spheres.items():
        if frame not in transforms: continue
        t = transforms[frame]
        spheres[frame] = (np.einsum('nij,pj->npi',t[:,:3,:3],centers)+t[:,None,:3,3],radii)
    ac, ar = spheres[model.asm_frame]
    result = {}
    for frame,(centers,radii) in spheres.items():
        if frame == model.asm_frame or frame in model.ignored_frames: continue
        distances = np.linalg.norm(ac[:,:,None,:]-centers[:,None,:,:],axis=-1)-ar[None,:,None]-radii[None,None,:]
        result[frame] = np.min(distances,axis=(1,2))
    return result


def minimum_clearance_fast(model, configurations):
    """Use triangle BVH for ASM pairs when installed, old spheres otherwise.

    Do not skip signed containment checks merely because two SURFACE sphere
    covers are separated: one closed solid could be wholly inside the other.
    """
    narrow=getattr(model,'_native_narrow',None)
    if narrow is not None:
        q=np.asarray(configurations,float).reshape(-1,len(model.joints))
        return narrow.minimum(q)
    return minimum_clearance_spheres(model,configurations)


def minimum_clearance_spheres(model, configurations):
    """Exact weighted nearest-sphere query, with conservative candidate pruning."""
    if getattr(model,'_clearance_device',None):
        return minimum_clearance_gpu(model,configurations,model._clearance_device)
    from scipy.spatial import cKDTree
    transforms = link_transforms(model, configurations)
    ac, ar = model.spheres[model.asm_frame]
    if not hasattr(model, '_asm_kd_tree'):
        model._asm_kd_tree = cKDTree(ac)
    asm = transforms[model.asm_frame]
    centers, radii = [], []
    for frame,(points,rs) in model.spheres.items():
        if frame == model.asm_frame or frame in model.ignored_frames or frame not in transforms: continue
        relative = np.swapaxes(asm[:,:3,:3],1,2) @ transforms[frame][:,:3,:3]
        translation = np.einsum('nji,nj->ni',asm[:,:3,:3],transforms[frame][:,:3,3]-asm[:,:3,3])
        centers.append(np.einsum('nij,pj->npi',relative,points)+translation[:,None,:])
        radii.append(rs)
    points = np.concatenate(centers,axis=1)
    rs = np.concatenate(radii)
    flat = points.reshape(-1,3)
    distances, nearest = model._asm_kd_tree.query(flat)
    # The nearest center supplies an upper bound on weighted surface distance.
    # Any center outside this radius cannot improve that bound, even with max R.
    radius = distances+np.max(ar)-ar[nearest]+1e-12
    neighbors = model._asm_kd_tree.query_ball_point(flat,radius)
    counts = np.fromiter(map(len,neighbors),dtype=int)
    rows = np.repeat(np.arange(len(flat)),counts)
    indices = np.concatenate(neighbors).astype(int)
    values = np.linalg.norm(flat[rows]-ac[indices],axis=1)-ar[indices]-rs[rows%len(rs)]
    per_point = np.minimum.reduceat(values,np.r_[0,np.cumsum(counts)[:-1]])
    return per_point.reshape(len(points),-1).min(axis=1)


def minimum_clearance_gpu(model,configurations,device):
    """Same sphere pairs, double precision; optional fast path for live GPU env."""
    import torch
    transforms=link_transforms(model,configurations)
    asm=transforms[model.asm_frame]
    centers=[];radii=[]
    for frame,(points,rs) in model.spheres.items():
        if frame==model.asm_frame or frame in model.ignored_frames or frame not in transforms:continue
        relative=np.swapaxes(asm[:,:3,:3],1,2)@transforms[frame][:,:3,:3]
        translation=np.einsum('nji,nj->ni',asm[:,:3,:3],transforms[frame][:,:3,3]-asm[:,:3,3])
        centers.append(np.einsum('nij,pj->npi',relative,points)+translation[:,None,:]);radii.append(rs)
    points=np.concatenate(centers,axis=1);rs=np.concatenate(radii)
    ac,ar=model.spheres[model.asm_frame]
    cached=getattr(model,'_gpu_spheres',None)
    if cached is None or cached[0]!=id(ac):
        cached=(id(ac),torch.as_tensor(ac,device=device,dtype=torch.float64),
                torch.as_tensor(ar,device=device,dtype=torch.float64),torch.as_tensor(rs,device=device,dtype=torch.float64))
        model._gpu_spheres=cached
    p=torch.as_tensor(points.reshape(-1,3),device=device,dtype=torch.float64)
    distances=torch.cdist(p,cached[1],compute_mode='donot_use_mm_for_euclid_dist')-cached[2][None]
    closest=distances.min(dim=1).values.reshape(len(points),-1)-cached[3][None]
    return closest.min(dim=1).values.cpu().numpy()


def minimum_world_clearance(checker,configurations):
    """Batch the unchanged world sphere-to-triangle query, no resampling."""
    transforms=link_transforms(checker.kinematics,configurations)
    centers=[];radii=[]
    for frame,(p,r) in checker.kinematics.environment_spheres.items():
        if frame not in transforms:continue
        t=transforms[frame]
        centers.append(np.einsum('nij,pj->npi',t[:,:3,:3],p)+t[:,None,:3,3])
        radii.append(r)
    points=np.concatenate(centers,axis=1)
    world=checker.base_position+points@checker.base_rotation.T
    distance,_=checker._query(world.reshape(-1,3))
    return (distance.reshape(len(points),-1)-np.concatenate(radii)[None]).min(axis=1)
