"""Pure normalized-force contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force import (
    force_world_from_action,
    normalize_force_direction,
    validate_force_weight_ratio,
)


def test_half_weight_force_uses_live_mass():
    force = force_world_from_action(np.array([1.0, 0.0, 0.0]), 0.0057, 0.5)
    np.testing.assert_allclose(force, [0.5 * 0.0057 * 9.81, 0.0, 0.0])


def test_diagonal_is_norm_limited():
    direction = normalize_force_direction(np.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)
    np.testing.assert_allclose(direction, [2.0**-0.5, 2.0**-0.5, 0.0])


def test_components_are_clipped_before_norm_limit():
    direction = normalize_force_direction(np.array([8.0, 0.5, 0.0]))
    np.testing.assert_allclose(direction, np.array([1.0, 0.5, 0.0]) / np.sqrt(1.25))


@pytest.mark.parametrize("ratio", [0.0, -0.1, 2.01, np.nan, np.inf])
def test_ratio_outside_contract_is_rejected(ratio):
    with pytest.raises(ValueError):
        validate_force_weight_ratio(ratio)


@pytest.mark.parametrize("ratio", [1.0e-9, 0.5, 2.0])
def test_ratio_inside_contract_is_accepted(ratio):
    assert validate_force_weight_ratio(ratio) == ratio


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan, np.inf])
def test_invalid_live_mass_is_rejected(mass):
    with pytest.raises(ValueError, match="mass"):
        force_world_from_action([1.0, 0.0, 0.0], mass, 0.5)


def test_force_function_does_not_modify_input():
    action = np.array([1.0, 1.0, 0.0])
    before = action.copy()
    force_world_from_action(action, 0.0057, 0.5)
    np.testing.assert_array_equal(action, before)
