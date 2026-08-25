from pathlib import Path

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    CONTROL_HZ,
    PHYSICS_HZ,
    PHYSICS_STEPS_PER_CONTROL,
    ParameterizedForceConfig,
    ParameterizedForceMode,
    parameterized_endpoint_forces,
    parameterized_force_ratio,
)


ROOT = Path(__file__).resolve().parents[2]


def test_six_mode_enum_is_exact():
    assert [(item.name, int(item)) for item in ParameterizedForceMode] == [
        ("HOLD", 0),
        ("MOVE_POS", 1),
        ("MOVE_NEG", 2),
        ("VIEW_POS", 3),
        ("VIEW_NEG", 4),
        ("UP", 5),
    ]


def test_clocks_and_strength_mapping_are_frozen():
    assert PHYSICS_HZ == 240
    assert CONTROL_HZ == 10
    assert PHYSICS_STEPS_PER_CONTROL == 24
    cfg = ParameterizedForceConfig()
    assert [parameterized_force_ratio(ParameterizedForceMode.MOVE_POS, alpha, cfg) for alpha in (0, 0.5, 1)] == pytest.approx([0.70, 0.95, 1.20])
    assert [parameterized_force_ratio(ParameterizedForceMode.VIEW_POS, alpha, cfg) for alpha in (0, 0.5, 1)] == pytest.approx([0.30, 0.60, 0.90])
    assert [parameterized_force_ratio(ParameterizedForceMode.UP, alpha, cfg) for alpha in (0, 0.5, 1)] == pytest.approx([0.70, 0.85, 1.00])


def test_force_distribution_and_directions():
    mass = 0.005735
    axis = np.asarray([1.0, 0.0, 0.0])
    move = parameterized_endpoint_forces(ParameterizedForceMode.MOVE_POS, 0.5, mass_kg=mass, camera_axis_world=axis)
    np.testing.assert_allclose(move.camera_force_world, move.other_force_world)
    assert np.dot(move.direction_world, axis) == pytest.approx(0.0)
    assert move.direction_world[2] == pytest.approx(0.0)
    assert np.linalg.norm(move.camera_force_world + move.other_force_world) == pytest.approx(0.95 * mass * 9.81)

    move_neg = parameterized_endpoint_forces(ParameterizedForceMode.MOVE_NEG, 0.5, mass_kg=mass, camera_axis_world=axis)
    np.testing.assert_allclose(move_neg.direction_world, -move.direction_world)

    view = parameterized_endpoint_forces(ParameterizedForceMode.VIEW_POS, 0.5, mass_kg=mass, camera_axis_world=axis)
    assert np.linalg.norm(view.camera_force_world) == pytest.approx(0.60 * mass * 9.81)
    np.testing.assert_allclose(view.other_force_world, np.zeros(3))

    up = parameterized_endpoint_forces(ParameterizedForceMode.UP, 0.5, mass_kg=mass, camera_axis_world=axis)
    np.testing.assert_allclose(up.camera_force_world, [0.0, 0.0, 0.85 * mass * 9.81])
    np.testing.assert_allclose(up.other_force_world, np.zeros(3))

    hold = parameterized_endpoint_forces(ParameterizedForceMode.HOLD, 1.0, mass_kg=mass, camera_axis_world=axis)
    np.testing.assert_allclose(hold.camera_force_world, np.zeros(3))
    np.testing.assert_allclose(hold.other_force_world, np.zeros(3))


def test_full_control_step_force_activity_contract():
    mass = 0.005735
    axis = np.asarray([1.0, 0.0, 0.0])
    active = [
        parameterized_endpoint_forces(ParameterizedForceMode.MOVE_POS, 0.5, mass_kg=mass, camera_axis_world=axis)
        for _ in range(PHYSICS_STEPS_PER_CONTROL)
    ]
    hold = [
        parameterized_endpoint_forces(ParameterizedForceMode.HOLD, 0.5, mass_kg=mass, camera_axis_world=axis)
        for _ in range(PHYSICS_STEPS_PER_CONTROL)
    ]
    assert len(active) == len(hold) == 24
    assert all(np.linalg.norm(item.camera_force_world) > 0.0 for item in active)
    assert all(np.linalg.norm(item.camera_force_world) == 0.0 for item in hold)


def test_live_launcher_records_boundary_and_leakage_audit_fields():
    source = (ROOT / "scripts/parameterized_force/teleop_table_10hz.py").read_text(encoding="utf-8")
    for token in (
        '"physics_substeps"',
        '"physics_step_indices"',
        '"force_active_substeps"',
        '"start_sim_time_s"',
        '"end_sim_time_s"',
        '"end_rgb_frame_id"',
        '"com_displacement_norm_m"',
        '"axis_change_deg_unoriented"',
        '"actor_observation_keys"',
        '"finite_state"',
    ):
        assert token in source
