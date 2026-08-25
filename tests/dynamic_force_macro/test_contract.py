import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force_macro import (
    DynamicForceMacroActionId,
    DynamicForceMacroConfig,
    NumericalContractError,
    camera_sphere_centers_local,
    equivalent_com_wrench,
    lateral_direction_world,
    phase_for_substep,
    point_forces_for_action,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    PHYSICS_STEPS_PER_CONTROL,
    ParameterizedForceConfig,
    ParameterizedForceMode,
    parameterized_endpoint_forces,
    parameterized_force_ratio,
)


def test_action_ids_are_frozen():
    assert [int(value) for value in DynamicForceMacroActionId] == list(range(14))
    assert {
        name: int(getattr(DynamicForceMacroActionId, name))
        for name in ("HOLD", "MOVE_POS", "MOVE_NEG", "VIEW_POS", "VIEW_NEG", "UP")
    } == {"HOLD": 0, "MOVE_POS": 1, "MOVE_NEG": 2, "VIEW_POS": 3, "VIEW_NEG": 4, "UP": 5}


def test_move_low_total_force_is_four_tenths_weight_split_equally():
    points = point_forces_for_action(
        DynamicForceMacroActionId.MOVE_POS,
        mass_kg=0.005735,
        lateral_direction_world=np.array([0.0, 1.0, 0.0]),
        camera_center_world=np.array([0.0, 0.0, -0.006]),
        other_center_world=np.array([0.0, 0.0, 0.006]),
        config=DynamicForceMacroConfig(),
    )
    assert len(points) == 2
    np.testing.assert_allclose(np.linalg.norm(points[0].force_world), 0.20 * 0.005735 * 9.81)
    np.testing.assert_allclose(np.linalg.norm(points[1].force_world), 0.20 * 0.005735 * 9.81)
    force, torque = equivalent_com_wrench(points, np.zeros(3))
    np.testing.assert_allclose(force, sum((p.force_world for p in points), np.zeros(3)))
    np.testing.assert_allclose(torque, sum((np.cross(p.position_world, p.force_world) for p in points), np.zeros(3)))


@pytest.mark.parametrize(
    "action,ratio,point_count,sign",
    [
        (DynamicForceMacroActionId.MOVE_POS, 0.40, 2, 1.0),
        (DynamicForceMacroActionId.MOVE_NEG, 0.40, 2, -1.0),
        (DynamicForceMacroActionId.MOVE_POS_MEDIUM, 0.50, 2, 1.0),
        (DynamicForceMacroActionId.MOVE_NEG_MEDIUM, 0.50, 2, -1.0),
        (DynamicForceMacroActionId.MOVE_POS_HIGH, 0.60, 2, 1.0),
        (DynamicForceMacroActionId.MOVE_NEG_HIGH, 0.60, 2, -1.0),
        (DynamicForceMacroActionId.VIEW_POS, 0.25, 1, 1.0),
        (DynamicForceMacroActionId.VIEW_NEG, 0.25, 1, -1.0),
        (DynamicForceMacroActionId.VIEW_POS_MEDIUM, 0.35, 1, 1.0),
        (DynamicForceMacroActionId.VIEW_NEG_MEDIUM, 0.35, 1, -1.0),
        (DynamicForceMacroActionId.VIEW_POS_HIGH, 0.45, 1, 1.0),
        (DynamicForceMacroActionId.VIEW_NEG_HIGH, 0.45, 1, -1.0),
    ],
)
def test_move_and_view_action_ids_encode_three_force_levels(action, ratio, point_count, sign):
    mass = 0.005735
    points = point_forces_for_action(
        action,
        mass_kg=mass,
        lateral_direction_world=np.array([0.0, 1.0, 0.0]),
        camera_center_world=np.array([0.0, 0.0, -0.006]),
        other_center_world=np.array([0.0, 0.0, 0.006]),
        config=DynamicForceMacroConfig(),
    )
    assert len(points) == point_count
    expected_per_point = ratio * mass * 9.81 / point_count
    for point in points:
        np.testing.assert_allclose(point.force_world, [0.0, sign * expected_per_point, 0.0])


def test_move_and_view_phase_boundaries_are_exact():
    assert not phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 47).force_active
    assert phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 48).force_active
    assert phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 191).force_active
    assert not phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 192).force_active


def test_up_is_active_on_final_substep():
    assert phase_for_substep(DynamicForceMacroActionId.UP, 239).force_active


def test_parameterized_force_contract_is_exactly_240_over_10_hz():
    assert PHYSICS_STEPS_PER_CONTROL == 24
    cfg = ParameterizedForceConfig()
    assert [parameterized_force_ratio(ParameterizedForceMode.MOVE_POS, alpha, cfg) for alpha in (0, 0.5, 1)] == pytest.approx([0.70, 0.95, 1.20])
    assert [parameterized_force_ratio(ParameterizedForceMode.VIEW_POS, alpha, cfg) for alpha in (0, 0.5, 1)] == pytest.approx([0.30, 0.60, 0.90])
    assert [parameterized_force_ratio(ParameterizedForceMode.UP, alpha, cfg) for alpha in (0, 0.5, 1)] == pytest.approx([0.70, 0.85, 1.00])


def test_parameterized_endpoint_distribution_matches_contract():
    mass = 0.005735
    axis = np.asarray([1.0, 0.0, 0.0])
    move = parameterized_endpoint_forces(
        ParameterizedForceMode.MOVE_POS, 0.5, mass_kg=mass, camera_axis_world=axis
    )
    assert np.linalg.norm(move.camera_force_world) == pytest.approx(0.5 * 0.95 * mass * 9.81)
    np.testing.assert_allclose(move.camera_force_world, move.other_force_world)
    view = parameterized_endpoint_forces(
        ParameterizedForceMode.VIEW_POS, 0.5, mass_kg=mass, camera_axis_world=axis
    )
    assert np.linalg.norm(view.camera_force_world) == pytest.approx(0.60 * mass * 9.81)
    np.testing.assert_allclose(view.other_force_world, np.zeros(3))
    up = parameterized_endpoint_forces(
        ParameterizedForceMode.UP, 0.5, mass_kg=mass, camera_axis_world=axis
    )
    np.testing.assert_allclose(up.camera_force_world, [0.0, 0.0, 0.85 * mass * 9.81])
    np.testing.assert_allclose(up.other_force_world, np.zeros(3))


@pytest.mark.parametrize(
    "camera_offset,expected_camera,expected_other",
    [
        ([0.0, 0.0, -0.0127], [0.0, 0.0, -0.006], [0.0, 0.0, 0.006]),
        ([0.0, 0.0, 0.0127], [0.0, 0.0, 0.006], [0.0, 0.0, -0.006]),
    ],
)
def test_camera_sphere_center_is_selected_from_actual_camera_mount(
    camera_offset, expected_camera, expected_other
):
    camera, other = camera_sphere_centers_local(camera_offset, 0.012)
    np.testing.assert_allclose(camera, expected_camera)
    np.testing.assert_allclose(other, expected_other)


def test_camera_sphere_center_rejects_ambiguous_mount():
    with pytest.raises(NumericalContractError):
        camera_sphere_centers_local([0.0, 0.0, 0.0], 0.012)


def test_up_is_one_world_up_force_at_exact_camera_center():
    camera = np.array([0.11, -0.22, 0.33], dtype=np.float64)
    other = np.array([0.11, -0.22, 0.342], dtype=np.float64)
    points = point_forces_for_action(
        DynamicForceMacroActionId.UP,
        mass_kg=0.005735,
        lateral_direction_world=np.array([0.0, 1.0, 0.0]),
        camera_center_world=camera,
        other_center_world=other,
        config=DynamicForceMacroConfig(up_force_ratio=0.85),
    )
    assert len(points) == 1
    assert points[0].endpoint == "camera"
    np.testing.assert_allclose(points[0].position_world, camera)
    np.testing.assert_allclose(points[0].force_world, [0.0, 0.0, 0.85 * 0.005735 * 9.81])


def test_lateral_direction_rejects_vertical_axis():
    with pytest.raises(NumericalContractError):
        lateral_direction_world([0.0, 0.0, 1.0])
