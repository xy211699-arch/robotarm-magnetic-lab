from __future__ import annotations

import math

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface import (
    IdealSurfaceAction,
    IdealSurfaceConfig,
    SurfaceFlags,
    compute_action_mask,
)


def test_frozen_action_ids_are_contiguous_and_unique():
    assert [(item.name, int(item)) for item in IdealSurfaceAction] == [
        ("HOLD", 0),
        ("START_TILT_000", 1),
        ("START_TILT_045", 2),
        ("START_TILT_090", 3),
        ("START_TILT_135", 4),
        ("START_TILT_180", 5),
        ("START_TILT_225", 6),
        ("START_TILT_270", 7),
        ("START_TILT_315", 8),
        ("TILT_MORE", 9),
        ("RISE", 10),
        ("PRECESS_POS", 11),
        ("PRECESS_NEG", 12),
        ("ROLL_POS", 13),
        ("ROLL_NEG", 14),
    ]


def test_default_config_matches_ideal_surface_v2():
    cfg = IdealSurfaceConfig()
    assert cfg.schema_version == "ideal_surface_v2"
    assert cfg.action_duration_s == 1.0
    assert cfg.tilt_step_rad == pytest.approx(math.radians(15.0))
    assert cfg.precession_step_rad == pytest.approx(math.radians(15.0))
    assert cfg.roll_arc_length_m == pytest.approx(0.010)
    assert cfg.upright_enter_rad == pytest.approx(math.radians(5.0))
    assert cfg.upright_exit_rad == pytest.approx(math.radians(8.0))


def enabled(mask):
    return set(np.flatnonzero(np.asarray(mask, dtype=bool)).tolist())


def test_mask_is_minimal_and_state_dependent():
    cfg = IdealSurfaceConfig()
    upright = compute_action_mask(
        SurfaceFlags(upright=True, side_contact=False), cfg
    )
    assert enabled(upright) == {0, 1, 2, 3, 4, 5, 6, 7, 8}

    tilted = compute_action_mask(
        SurfaceFlags(upright=False, side_contact=False), cfg
    )
    assert enabled(tilted) == {0, 9, 10, 11, 12}

    side = compute_action_mask(
        SurfaceFlags(upright=False, side_contact=True), cfg
    )
    assert enabled(side) == {0, 9, 10, 11, 12, 13, 14}

    limited = compute_action_mask(
        SurfaceFlags(
            upright=False,
            side_contact=True,
            contact_limited=True,
        ),
        cfg,
    )
    assert enabled(limited) == {0, 10, 11, 12, 13, 14}


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="upright"):
        IdealSurfaceConfig(upright_enter_rad=0.2, upright_exit_rad=0.1)
    with pytest.raises(ValueError, match="action_duration"):
        IdealSurfaceConfig(action_duration_s=0.0)
