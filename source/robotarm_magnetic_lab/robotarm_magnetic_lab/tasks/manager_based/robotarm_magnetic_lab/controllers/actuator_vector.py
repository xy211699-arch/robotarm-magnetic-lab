"""Simulator-independent 4 x 9 relative actuator planner (SI units).

The first second is committed; the other three are previews, never a queue.
V1 deliberately stops at each boundary. All nine dimensions are enabled.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SCALES = np.r_[[0.001]*3, np.deg2rad([0.5]*3), np.deg2rad([5.0]*3)]
SPEED = np.r_[[0.12]*6, [0.8]*3]
ACCELERATION = np.r_[[0.24]*6, [1.6]*3]
HZ = 240


class Infeasible(ValueError):
    pass


def clearance_lower_bound(transforms, initial, mappings, arm, world):
    """Rigid sphere-center motion bound for every supplied configuration."""
    displacement = 0.
    for mapping in mappings:
        for name,(centers,radii) in mapping.items():
            if name not in transforms:
                continue
            t,origin=transforms[name],initial[name]
            bound=np.linalg.norm(t[:,:3,3]-origin[:3,3],axis=1)
            # Include sphere radii: narrow phase may use the enclosed mesh,
            # whose points rotate even when a proxy center stays stationary.
            bound+=np.linalg.norm(t[:,:3,:3]-origin[:3,:3],axis=(1,2))*(np.linalg.norm(centers,axis=1)+radii).max()
            displacement=max(displacement,float(bound.max()))
    return min(arm-2*displacement,world-displacement)


def decode(chunk, scales=SCALES):
    a = np.asarray(chunk, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if a.shape != (4, 9) or not np.isfinite(a).all():
        raise ValueError('expected finite normalized (4,9) chunk')
    if scales.shape != (9,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('nine positive finite scales required')
    clipped = np.clip(a, -1, 1)
    # Bound the combined translation / rotation, not only its components.
    for part in (slice(0,3), slice(3,6), slice(6,9)):
        norm = np.linalg.norm(clipped[:,part], axis=1, keepdims=True)
        clipped[:,part] /= np.maximum(norm, 1.)
    return clipped, clipped * scales


def goals(position, rotation, ball, physical, base_rotation):
    p, r, b = np.array(position).copy(), np.array(rotation).copy(), np.array(ball).copy()
    output = []
    for delta in physical:
        p = p + base_rotation @ delta[:3]
        r = r @ Rotation.from_rotvec(delta[3:6]).as_matrix()
        b = b + delta[6:]
        output.append((p.copy(), r.copy(), b.copy()))
    return output


def quintic(knots):
    """4 seconds, 961 positions including t=0, exact analytic derivatives."""
    knots = np.asarray(knots, float)
    if knots.shape != (5,9) or not np.isfinite(knots).all():
        raise ValueError('expected finite (5,9) joint knots')
    u = np.arange(1,HZ+1)/HZ
    h = 10*u**3-15*u**4+6*u**5
    dh = 30*u**2-60*u**3+30*u**4
    ddh = 60*u-180*u**2+120*u**3
    q, v, acc = [knots[0:1]], [np.zeros((1,9))], [np.zeros((1,9))]
    for start, end in zip(knots[:-1],knots[1:]):
        delta = end-start
        q.append(start+h[:,None]*delta)
        v.append(dh[:,None]*delta)
        acc.append(ddh[:,None]*delta)
    return np.concatenate(q), np.concatenate(v), np.concatenate(acc)


def plan(reference, fk, desired, limits, clearance, margin=.005):
    """fk(q6) returns world EE pose; clearance(qNx9) must cover the path.

    No wrench inverse solver, joint-6 freeze, or state teleportation occurs.
    Caller may explicitly project the input and retry after Infeasible.
    """
    reference = np.asarray(reference,float)
    limits = np.asarray(limits,float)
    if reference.shape != (9,) or limits.shape != (9,2):
        raise ValueError('invalid joint state or limits')
    if not np.isfinite(reference).all() or np.any(reference < limits[:,0]) or np.any(reference > limits[:,1]):
        raise Infeasible('reference outside limits')
    knots = [reference.copy()]
    for p,r,b in desired:
        start = knots[-1]
        # Exact rest-to-rest quintic bounds: peak speed 1.875*dq,
        # peak acceleration (10/sqrt(3))*dq.
        radius = np.minimum(SPEED/1.875, ACCELERATION/(10/np.sqrt(3)))
        low = np.maximum(limits[:6,0],start[:6]-radius[:6])
        high = np.minimum(limits[:6,1],start[:6]+radius[:6])
        def residual(q):
            pp,rr = fk(q)
            return np.r_[(pp-p)/.001, Rotation.from_matrix(r.T@rr).as_rotvec()/.01]
        fit = least_squares(residual, start[:6],bounds=(low,high),max_nfev=45,
                            xtol=1e-10,ftol=1e-10,gtol=1e-10)
        pp,rr = fk(fit.x)
        if np.linalg.norm(pp-p)>1e-5 or Rotation.from_matrix(r.T@rr).magnitude()>1e-4:
            raise Infeasible('IK or one-second joint range')
        knots.append(np.r_[fit.x,b])
    q,v,acc = quintic(np.array(knots))
    if np.any(q<limits[:,0]-1e-9) or np.any(q>limits[:,1]+1e-9):
        raise Infeasible('joint limits')
    if np.any(np.abs(v)>SPEED+1e-9) or np.any(np.abs(acc)>ACCELERATION+1e-9):
        raise Infeasible('velocity or acceleration')
    minimum = float(clearance(q))
    if not np.isfinite(minimum) or minimum < margin:
        raise Infeasible('predicted clearance')
    return dict(q=q, velocity=v, acceleration=acc, knots=np.array(knots), clearance_m=minimum)
