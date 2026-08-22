import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force_macro import (
    DynamicForceMacroActionId,
    DynamicForceMacroConfig,
    NumericalContractError,
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
    "camera,other",
    [
        ([-0.006, 0.0, 0.0], [0.006, 0.0, 0.0]),
        ([-0.004, 0.0, 0.004], [0.004, 0.0, -0.004]),
        ([0.003, -0.004, -0.002], [-0.003, 0.004, 0.002]),
    ],
)
def test_up_is_camera_lift_other_end_down_pure_couple(camera, other):
    camera = np.asarray(camera, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    points = point_forces_for_action(
        DynamicForceMacroActionId.UP,
        mass_kg=0.005735,
        lateral_direction_world=np.array([0.0, 1.0, 0.0]),
        camera_center_world=camera,
        other_center_world=other,
        config=DynamicForceMacroConfig(up_force_ratio=0.85),
    )
    assert [point.endpoint for point in points] == ["camera", "other"]
    np.testing.assert_allclose(points[0].force_world, -points[1].force_world)
    assert points[0].force_world[2] >= 0.0
    assert points[1].force_world[2] <= 0.0
    force, torque = equivalent_com_wrench(points, 0.5 * (camera + other))
    np.testing.assert_allclose(force, np.zeros(3), atol=1.0e-12)
    camera_axis = (camera - other) / np.linalg.norm(camera - other)
    camera_vertical_acceleration_sign = float(np.dot(np.cross(torque, camera_axis), np.array([0.0, 0.0, 1.0])))
    assert camera_vertical_acceleration_sign >= 0.0


def test_up_camera_down_uses_deterministic_nonzero_tipping_couple():
    points = point_forces_for_action(
        DynamicForceMacroActionId.UP,
        mass_kg=0.005735,
        lateral_direction_world=np.array([0.0, 1.0, 0.0]),
        camera_center_world=np.array([0.0, 0.0, -0.006]),
        other_center_world=np.array([0.0, 0.0, 0.006]),
        config=DynamicForceMacroConfig(up_force_ratio=0.85),
    )
    assert np.linalg.norm(points[0].force_world) > 0.0
    np.testing.assert_allclose(points[0].force_world, -points[1].force_world)


def test_lateral_direction_rejects_vertical_axis():
    with pytest.raises(NumericalContractError):
        lateral_direction_world([0.0, 0.0, 1.0])
