"""Gate 1: complete execution, SI budgets, C2 and unchanged protections."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import new_stomach_rl_trajectory as rl
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.actuator_vector import SCALES, SPEED, ACCELERATION, quintic


def state(clearance=lambda path: .010):
    def fk(q):
        return q[:3].copy(), Rotation.from_rotvec(q[3:6]).as_matrix()
    return rl.JointState(np.zeros(9), fk, np.eye(3), np.tile([-1., 1.], (9, 1)), clearance)


@pytest.mark.parametrize('mode,shape', [('single', (9,)), ('chunk', (4, 9))])
def test_hold_is_240_ticks_and_certified(mode, shape):
    out = rl.build_schedule(mode, np.zeros(shape), state())
    assert out.q.shape == (241, 9)
    assert np.array_equal(out.q, np.zeros((241, 9)))
    assert out.clearance_m >= .005
    assert out.subsegment_ticks == (60, 120, 180, 240)


@pytest.mark.parametrize('mode,a', [('single', np.zeros((4, 9))), ('chunk', np.zeros(9)), ('chunk', np.full((4, 9), np.nan))])
def test_invalid_interface_rejected(mode, a):
    with pytest.raises(ValueError):
        rl.build_schedule(mode, a, state())


def test_last_segment_really_executes_and_prior_segments_unchanged():
    actions = np.zeros((4, 9))
    zero = rl.build_schedule('chunk', actions, state())
    actions[3, 0] = .5
    out = rl.build_schedule('chunk', actions, state())
    assert np.array_equal(out.q[:181], zero.q[:181])
    assert out.q[240, 0] == pytest.approx(.5 * SCALES[0] / 4, abs=1e-6)
    assert out.subsegment_actions_si.shape == (4, 9)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_all_four_segments_and_one_second_budget(sign):
    a = np.zeros((4, 9)); a[:, 0] = sign
    out = rl.build_schedule('chunk', a, state())
    assert np.allclose(out.q[[60, 120, 180, 240], 0], sign * SCALES[0] * np.arange(1, 5) / 4, atol=1e-6)
    assert np.linalg.norm(out.subsegment_actions_si[:, :3], axis=1).sum() <= SCALES[0] + 1e-12


def test_c2_boundaries_and_duration_aware_limits():
    a = np.ones((4, 9))
    out = rl.build_schedule('chunk', a, state())
    assert np.all(np.abs(out.velocity) <= SPEED + 1e-9)
    assert np.all(np.abs(out.acceleration) <= ACCELERATION + 1e-9)
    assert np.max(np.abs(out.velocity[[0, 60, 120, 180, 240]])) < 1e-12
    assert np.max(np.abs(out.acceleration[[0, 60, 120, 180, 240]])) < 1e-12
    assert out.analytic_peak_speed_ratio <= 1 + 1e-9
    assert out.analytic_peak_acceleration_ratio <= 1 + 1e-9


def test_ball_quarter_requires_projection_not_unchecked_compression():
    a = np.zeros((4, 9)); a[:, 6] = 1.
    out = rl.build_schedule('chunk', a, state())
    assert 0 < out.projection_scale < 1
    assert np.max(np.abs(out.acceleration)) <= ACCELERATION.max()
    assert out.q[240, 6] > 0


def test_collision_refusal_does_not_relax_or_teleport():
    with pytest.raises(rl.Infeasible):
        rl.build_schedule('chunk', np.ones((4, 9)), state(lambda path: .0049))


def test_archived_preview_stays_961_samples():
    q, _, _ = quintic(np.zeros((5, 9)))
    assert q.shape == (961, 9)


def test_single_history_is_actual_quarter_deltas_not_four_copies():
    a = np.zeros(9); a[0] = .5
    out = rl.build_schedule('single', a, state())
    history = out.subsegment_actions_si[:, 0]
    assert history.sum() == pytest.approx(.5 * SCALES[0], abs=1e-6)
    assert history[1] > history[0]
    assert np.allclose(history, np.diff(out.q[[0, 60, 120, 180, 240], 0]))


def test_nonfinite_soft_limits_cannot_evade_limit_checks():
    s = state()
    limits = s.limits.copy(); limits[8] = np.nan
    invalid = rl.JointState(s.reference, s.fk, s.base_rotation, limits, s.certified_clearance)
    with pytest.raises(ValueError, match='finite'):
        rl.build_schedule('single', np.zeros(9), invalid)


def test_continuous_ball_limits_keep_original_unbounded_semantics():
    s = state()
    limits = s.limits.copy(); limits[6:] = [-np.inf, np.inf]
    continuous = rl.JointState(s.reference, s.fk, s.base_rotation, limits, s.certified_clearance)
    action = np.zeros((4, 9)); action[:, 6] = .25
    out = rl.build_schedule('chunk', action, continuous)
    assert np.isfinite(out.q).all()
    assert out.q[-1, 6] > 0
    assert out.analytic_peak_acceleration_ratio <= 1


@pytest.mark.parametrize('row', [[np.inf, np.inf], [-np.inf, -np.inf], [1., -1.]])
def test_invalid_limit_interval_rejected(row):
    s = state()
    limits = s.limits.copy(); limits[8] = row
    invalid = rl.JointState(s.reference, s.fk, s.base_rotation, limits, s.certified_clearance)
    with pytest.raises(ValueError):
        rl.build_schedule('single', np.zeros(9), invalid)
