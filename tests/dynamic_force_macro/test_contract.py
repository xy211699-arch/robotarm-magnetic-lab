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


def test_action_ids_are_frozen():
    assert [int(value) for value in DynamicForceMacroActionId] == list(range(6))


def test_move_total_force_is_nine_tenths_weight_split_equally():
    points = point_forces_for_action(
        DynamicForceMacroActionId.MOVE_POS,
        mass_kg=0.005735,
        lateral_direction_world=np.array([0.0, 1.0, 0.0]),
        camera_center_world=np.array([0.0, 0.0, -0.006]),
        other_center_world=np.array([0.0, 0.0, 0.006]),
        config=DynamicForceMacroConfig(),
    )
    assert len(points) == 2
    np.testing.assert_allclose(np.linalg.norm(points[0].force_world), 0.45 * 0.005735 * 9.81)
    np.testing.assert_allclose(np.linalg.norm(points[1].force_world), 0.45 * 0.005735 * 9.81)
    force, torque = equivalent_com_wrench(points, np.zeros(3))
    np.testing.assert_allclose(force, sum((p.force_world for p in points), np.zeros(3)))
    np.testing.assert_allclose(torque, sum((np.cross(p.position_world, p.force_world) for p in points), np.zeros(3)))


def test_move_and_view_phase_boundaries_are_exact():
    assert not phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 47).force_active
    assert phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 48).force_active
    assert phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 191).force_active
    assert not phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 192).force_active


def test_up_is_active_on_final_substep():
    assert phase_for_substep(DynamicForceMacroActionId.UP, 239).force_active


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
