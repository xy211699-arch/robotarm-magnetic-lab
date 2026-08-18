import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
    CapsuleState, PrimitiveId, PrimitiveRequest, make_local_primitive_controller_cfg,
)


def test_shared_config_has_frozen_design_values():
    cfg = make_local_primitive_controller_cfg()
    assert cfg.capsule_mass_kg == pytest.approx(0.0057349997)
    assert cfg.motion_duration_s == (5.5, 4.5, 3.5, 8.0)
    assert cfg.hard_timeout_s == (8.0, 7.0, 6.0, 9.5)
    assert cfg.xy_force_limit_n == pytest.approx(1.0 * cfg.weight_n)
    assert cfg.downward_preload_n == pytest.approx(0.15 * cfg.weight_n)


def test_request_validates_primitive_and_azimuth():
    request = PrimitiveRequest(PrimitiveId.UPRIGHT_TO_30_DEG, np.pi / 2)
    assert request.primitive_id is PrimitiveId.UPRIGHT_TO_30_DEG
    assert request.azimuth_rad == pytest.approx(np.pi / 2)
    with pytest.raises(ValueError):
        PrimitiveRequest(0, np.nan)


def test_capsule_state_detects_nonfinite_values():
    state = CapsuleState(0.0, np.zeros(3), [1, 0, 0, 0], np.zeros(3), np.zeros(3))
    assert state.is_finite
    bad = CapsuleState(0.0, [np.nan, 0, 0], [1, 0, 0, 0], np.zeros(3), np.zeros(3))
    assert not bad.is_finite
