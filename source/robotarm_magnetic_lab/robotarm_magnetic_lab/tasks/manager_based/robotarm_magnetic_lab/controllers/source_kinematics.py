"""Serial-chain product of exponentials from the live world geometric Jacobian.

Uses current joint screw axes, including the actual mounting transform. Unlike
p + J*dq, composition retains rotation order over the two-second horizon.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def source_pose(position, rotation, jacobian, joint_delta):
    delta = np.asarray(joint_delta, float)
    p = np.tile(position, (len(delta),1)).astype(float)
    r = np.tile(rotation, (len(delta),1,1)).astype(float)
    for j in reversed(range(jacobian.shape[1])):
        w, linear = jacobian[3:,j], jacobian[:3,j]
        norm2 = float(w@w)
        if norm2 < 1e-16:
            p += delta[:,j,None]*linear
            continue
        center = position + np.cross(w,linear)/norm2
        change = Rotation.from_rotvec(delta[:,j,None]*w).as_matrix()
        p = np.einsum('nij,nj->ni',change,p-center)+center
        r = change @ r
    return p,r
