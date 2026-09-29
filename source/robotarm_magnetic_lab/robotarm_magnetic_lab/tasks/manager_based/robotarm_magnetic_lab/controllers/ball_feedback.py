"""Pure bounded Ball servo math. No simulator writes and no force injection."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import lsq_linear

DT = .05
STEPS = 12


def revolute_branch_shift(reference, actual):
    """Rebase PhysX [-2pi,2pi] coordinates; preserve restoring servo error."""
    period=4*np.pi
    return period*np.rint((np.asarray(actual)-reference)/period)


def due(substep):
    return 0 <= substep < 120 and substep % STEPS == 0


def segment(q, velocity, end_velocity):
    t = np.arange(STEPS+1)[:,None]/240.
    a = (np.asarray(end_velocity)-velocity)/DT
    return q+t*velocity+.5*t*t*a, velocity+t*a


def brake_velocity(velocity, acceleration):
    return np.sign(velocity)*np.maximum(0.,np.abs(velocity)-acceleration*DT)


def stopping_path(q, velocity, acceleration, steps=480):
    """Exact maximum-deceleration stop, with no repeated half-second tail.

    Re-evaluating this curve at a later point preserves its stopping endpoint.
    Zero velocity stays zero; no direction reversal is introduced.
    """
    q,velocity,acceleration=map(lambda x:np.asarray(x,float),(q,velocity,acceleration))
    if np.any(acceleration<=0):raise ValueError('positive braking acceleration required')
    t=np.arange(steps+1)[:,None]/240.
    moving=np.minimum(t,np.abs(velocity)/acceleration)
    positions=q+velocity*moving-.5*np.sign(velocity)*acceleration*moving**2
    velocities=np.sign(velocity)*np.maximum(0.,np.abs(velocity)-acceleration*t)
    return positions,velocities


def safe_ball(q, velocity, end_velocity, limits, speed=.8, acceleration=1.6):
    path, velocities = segment(q,velocity,end_velocity)
    if not np.isfinite(path).all() or not np.isfinite(velocities).all(): return False
    if np.any(np.abs(velocities)>speed+1e-7) or np.any(np.abs(end_velocity-velocity)>acceleration*DT+1e-7): return False
    # Endpoint of constant-deceleration stop; monotonic in each joint.
    stopped = path[-1]+np.sign(end_velocity)*end_velocity**2/(2*acceleration)
    return bool(np.all(path>=limits[:,0]) and np.all(path<=limits[:,1])
                and np.all(stopped>=limits[:,0]) and np.all(stopped<=limits[:,1]))


@dataclass
class FeedbackResult:
    end_velocity: np.ndarray
    status: str
    reason: str
    elapsed_s: float
    selected_wrench: np.ndarray | None
    projection: float | None


def choose_velocity(q, velocity, nominal, limits, target, selector, evaluate,
                    *, hold=False, speed=.8, acceleration=1.6, budget_s=.04):
    """evaluate accepts N end velocities, returns filtered COM wrenches (N,6).

    The callback owns the exact magnetic model and predicted arm movement.
    Collision geometry is certified by the caller, independently of Ball angle.
    """
    start=time.monotonic()
    q,velocity,nominal,limits=map(lambda v:np.asarray(v,float),(q,velocity,nominal,limits))
    stop=brake_velocity(velocity,acceleration)
    if not safe_ball(q,velocity,stop,limits,speed,acceleration):
        raise RuntimeError('no safe Ball reference braking trajectory')
    if hold: return FeedbackResult(stop,'braking','hold',time.monotonic()-start,None,None)
    lower=np.maximum(-speed,velocity-acceleration*DT)
    upper=np.minimum(speed,velocity+acceleration*DT)
    # Also bound the segment endpoint. Full path/extremum and stopping limits
    # are checked after solving, not merely the terminal angle.
    lower=np.maximum(lower,2*(limits[:,0]-q)/DT-velocity)
    upper=np.minimum(upper,2*(limits[:,1]-q)/DT-velocity)
    if np.any(lower>=upper):
        return FeedbackResult(stop,'braking','joint_range',time.monotonic()-start,None,None)
    nominal=np.clip(nominal,lower,upper)
    eps=.01  # finite-difference end velocity => .00025rad angle perturbation
    probes=np.vstack([nominal,nominal+np.eye(3)*eps,stop])
    w=evaluate(probes)
    scales=np.array([.040]*3+[.0009]*3)
    d=((w[1:4]-w[0])/eps).T/scales[:,None]
    target_scaled=np.asarray(target)/scales
    residual=target_scaled-w[0]/scales+d@nominal
    matrix=np.vstack([d,.002*np.eye(3)/speed])
    rhs=np.r_[residual,.002*nominal/speed]
    solved=lsq_linear(matrix,rhs,bounds=(lower,upper),tol=1e-6,max_iter=30).x
    # Direction-only candidate avoids starving the main requested component.
    sel=np.asarray(selector,float)
    directional=lsq_linear(np.vstack([(sel@d)[None],.002*np.eye(3)/speed]),
        np.r_[sel@residual,.002*nominal/speed],bounds=(lower,upper),tol=1e-6,max_iter=30).x
    candidates=np.vstack([solved,directional,nominal,stop])
    actual=evaluate(candidates)
    baseline=float(sel@(w[-1]/scales))
    best=None
    for i,(v,wi) in enumerate(zip(candidates,actual)):
        if not safe_ball(q,velocity,v,limits,speed,acceleration): continue
        projection=float(sel@(wi/scales))
        if projection < min(0.,baseline)-1e-8: continue
        cost=(float(projection < -1e-8),float(np.sum((wi/scales-target_scaled)**2)),
              float(np.sum(((v-nominal)/speed)**2)))
        if best is None or cost<best[0]:best=(cost,i,projection)
    elapsed=time.monotonic()-start
    if elapsed>budget_s:
        return FeedbackResult(stop,'braking','feedback_deadline',elapsed,None,None)
    if best is None:
        return FeedbackResult(stop,'braking','no_admissible_candidate',elapsed,None,None)
    _,i,projection=best
    degraded=projection<0 or np.any(np.isclose(candidates[i],lower,atol=1e-5)|np.isclose(candidates[i],upper,atol=1e-5))
    return FeedbackResult(candidates[i],'degraded' if degraded else 'tracking',
        'direction_residual' if projection<0 else ('bounded' if degraded else 'clear'),elapsed,actual[i],projection)
