import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
    CapsuleState, PrimitiveCode, PrimitiveRequest, make_local_primitive_action_cfg,
)


def test_shared_config_has_frozen_design_values():
    cfg = make_local_primitive_action_cfg()
    assert cfg.capsule_mass_kg == pytest.approx(0.0057349997)
    assert cfg.motion_durations_s == (5.5, 4.5, 3.5, 8.0)
    assert cfg.timeout_durations_s == (8.0, 7.0, 6.0, 9.5)
    assert cfg.xy_force_limit_n == pytest.approx(0.5 * cfg.weight_n)
    assert cfg.downward_preload_n == pytest.approx(0.15 * cfg.weight_n)


def test_request_normalizes_direction_without_aliasing():
    direction = np.array([3.0, 4.0])
    request = PrimitiveRequest(PrimitiveCode.UPRIGHT_TO_TILT, direction)
    direction[:] = 0.0
    np.testing.assert_allclose(request.direction_xy, [0.6, 0.8])
    with pytest.raises(ValueError):
        PrimitiveRequest(0, np.zeros(2))


def test_capsule_state_detects_nonfinite_values():
    state = CapsuleState(np.zeros(3), [1, 0, 0, 0], np.zeros(3), np.zeros(3))
    assert state.is_finite
    bad = CapsuleState([np.nan, 0, 0], [1, 0, 0, 0], np.zeros(3), np.zeros(3))
    assert not bad.is_finite
