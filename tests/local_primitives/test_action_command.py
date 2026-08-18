import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.local_primitive_action import (
    PrimitiveCommandDecoder, make_local_primitive_action_cfg,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
    PrimitiveId, simulation_profile_sha256,
)


def test_start_is_rising_edge_and_zero_direction_defaults_to_world_x():
    decoder = PrimitiveCommandDecoder()
    request = decoder.decode([1.0, 2.0, 0.0, 0.0])
    assert request.primitive_id is PrimitiveId.UPRIGHT_TO_30_DEG
    assert request.azimuth_rad == pytest.approx(0.0)
    assert decoder.decode([1.0, 2.0, 0.0, 0.0]) is None
    assert decoder.decode([0.0, 2.0, 0.0, 0.0]) is None
    assert decoder.decode([1.0, 2.0, 0.0, 1.0]).azimuth_rad == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("code", [-1.0, 4.0, 1.5])
def test_invalid_primitive_code_is_rejected(code):
    decoder = PrimitiveCommandDecoder()
    with pytest.raises(ValueError):
        decoder.decode([1.0, code, 1.0, 0.0])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_action_is_rejected(value):
    with pytest.raises(ValueError):
        PrimitiveCommandDecoder().decode([1.0, value, 1.0, 0.0])


def test_action_exposes_exact_tracked_profile_digest():
    cfg = make_local_primitive_action_cfg()
    assert cfg.profile_sha256 == simulation_profile_sha256()
    assert cfg.controller_cfg.profile_sha256 == cfg.profile_sha256
