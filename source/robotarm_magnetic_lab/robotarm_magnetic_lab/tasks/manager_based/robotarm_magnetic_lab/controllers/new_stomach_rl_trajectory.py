"""Independent 1 s schedules; archived 4 s preview planner is untouched.

Quarter boundaries are rest-to-rest: position/velocity/acceleration match (C2).
IK radius and analytical extrema explicitly depend on duration, not timestamps.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .actuator_vector import SCALES, SPEED, ACCELERATION, HZ, Infeasible, decode, goals


@dataclass(frozen=True)
class JointState:
    reference: np.ndarray
    fk: Callable
    base_rotation: np.ndarray
    limits: np.ndarray
    certified_clearance: Callable


@dataclass(frozen=True)
class PlannedTrajectory:
    q: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    knots: np.ndarray
    subsegment_ticks: tuple[int, ...]
    subsegment_actions_si: np.ndarray
    projection_scale: float
    clearance_m: float
    reasons: tuple[str, ...]
    final_position: np.ndarray
    final_rotation: np.ndarray
    final_ball: np.ndarray
    analytic_peak_speed_ratio: float
    analytic_peak_acceleration_ratio: float


def duration_quintic(knots, duration):
    ticks = int(round(duration * HZ))
    u = np.arange(1, ticks + 1) / ticks
    h = 10*u**3 - 15*u**4 + 6*u**5
    dh = (30*u**2 - 60*u**3 + 30*u**4) / duration
    ddh = (60*u - 180*u**2 + 120*u**3) / duration**2
    q, v, a = [knots[0:1]], [np.zeros((1, 9))], [np.zeros((1, 9))]
    for start, end in zip(knots[:-1], knots[1:]):
        delta = end - start
        q.append(start + h[:, None] * delta)
        v.append(dh[:, None] * delta)
        a.append(ddh[:, None] * delta)
    return np.concatenate(q), np.concatenate(v), np.concatenate(a)


def _plan(state, desired, duration):
    ref, limits = np.asarray(state.reference, float), np.asarray(state.limits, float)
    if ref.shape != (9,) or limits.shape != (9, 2) or not np.isfinite(ref).all():
        raise ValueError('finite nine-joint reference and valid limits required')
    # PhysX continuous joints are unbounded. The SDK float32 soft-limit
    # calculation may expose (-inf,+inf); keep that original domain rather
    # than inventing finite Ball stops. NaN and invalid intervals still fail.
    if (np.isnan(limits).any() or np.isposinf(limits[:, 0]).any()
            or np.isneginf(limits[:, 1]).any() or np.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError('finite reference requires valid, non-NaN limit intervals')
    if np.any(ref < limits[:, 0]) or np.any(ref > limits[:, 1]):
        raise Infeasible('reference outside soft joint limits')
    radius = np.minimum(SPEED * duration / 1.875, ACCELERATION * duration**2 / (10 / np.sqrt(3)))
    knots = [ref.copy()]
    for p, r, ball in desired:
        start = knots[-1]
        low = np.maximum(limits[:6, 0], start[:6] - radius[:6])
        high = np.minimum(limits[:6, 1], start[:6] + radius[:6])

        def residual(q):
            pp, rr = state.fk(q)
            return np.r_[(pp-p)/.001, Rotation.from_matrix(r.T @ rr).as_rotvec()/.01]

        fit = least_squares(residual, start[:6], bounds=(low, high), max_nfev=45,
                            xtol=1e-10, ftol=1e-10, gtol=1e-10)
        pp, rr = state.fk(fit.x)
        if np.linalg.norm(pp - p) > 1e-5 or Rotation.from_matrix(r.T @ rr).magnitude() > 1e-4:
            raise Infeasible('duration-aware IK/joint range')
        knots.append(np.r_[fit.x, ball])
    knots = np.asarray(knots)
    delta = np.abs(np.diff(knots, axis=0))
    peak_v = float(np.max(1.875 * delta / duration / SPEED))
    peak_a = float(np.max((10 / np.sqrt(3)) * delta / duration**2 / ACCELERATION))
    if peak_v > 1 + 1e-9 or peak_a > 1 + 1e-9:
        raise Infeasible('analytic velocity or acceleration bound')
    # h is monotonic on [0,1], so endpoint limits certify the whole segment.
    if np.any(knots < limits[:, 0]) or np.any(knots > limits[:, 1]):
        raise Infeasible('soft joint limits')
    q, v, a = duration_quintic(knots, duration)
    clearance = float(state.certified_clearance(q))
    if not np.isfinite(clearance) or clearance < .005:
        raise Infeasible('predicted certified clearance below 5mm')
    return q, v, a, knots, clearance, peak_v, peak_a


def build_schedule(mode, action, start_state):
    raw = np.asarray(action, float)
    if mode not in ('single', 'chunk') or raw.shape != ((9,) if mode == 'single' else (4, 9)) or not np.isfinite(raw).all():
        raise ValueError('single finite 9D or chunk finite 4x9D required')
    if mode == 'single':
        clipped, physical = decode(np.repeat(raw[None], 4, axis=0), SCALES)
        physical = physical[:1]
        duration = 1.
    else:
        clipped, physical = decode(raw, SCALES / 4)
        duration = .25
    position, rotation = start_state.fk(np.asarray(start_state.reference)[:6])
    reasons = []
    for scale in (1., .5, .25, 0.):
        desired = goals(position, rotation, np.asarray(start_state.reference)[6:], physical * scale, start_state.base_rotation)
        try:
            q, v, a, knots, clearance, peak_v, peak_a = _plan(start_state, desired, duration)
            history = []
            previous_p, previous_r = start_state.fk(q[0, :6])
            previous_b = q[0, 6:]
            for tick in (60, 120, 180, 240):
                p, r = start_state.fk(q[tick, :6])
                history.append(np.r_[start_state.base_rotation.T @ (p - previous_p),
                                     Rotation.from_matrix(previous_r.T @ r).as_rotvec(),
                                     q[tick, 6:] - previous_b])
                previous_p, previous_r, previous_b = p, r, q[tick, 6:]
            return PlannedTrajectory(q, v, a, knots, (60, 120, 180, 240), np.asarray(history),
                scale, clearance, tuple(reasons), *desired[-1], peak_v, peak_a)
        except Infeasible as error:
            reasons.append(str(error))
    raise Infeasible('no certified schedule or HOLD: ' + repr(reasons))
