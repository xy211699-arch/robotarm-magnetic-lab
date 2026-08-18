import math

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.geometry import (
    camera_frame,
    capsule_axis_world,
    classify_contact_region,
    freeze_support_material_point,
    grid_direction_world,
    reconstruct_material_point,
    support_tangent_error,
    view_target_axis,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.contact_history import (
    ContactRegion,
    ContactSample,
    SideContactHistory,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.types import (
    CapsuleState,
    ElevenActionId,
)


def _state(quaternion=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.02)):
    return CapsuleState(position, quaternion, np.zeros(3), np.zeros(3))


def _axis_angle_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return np.asarray([math.cos(angle / 2.0), *(math.sin(angle / 2.0) * axis)])


def test_camera_frame_and_nine_grid_directions_are_relative_and_orthonormal():
    optical, up, right = camera_frame(_state())
    np.testing.assert_allclose([np.linalg.norm(optical), np.linalg.norm(up), np.linalg.norm(right)], 1.0)
    np.testing.assert_allclose([optical @ up, optical @ right, up @ right], 0.0, atol=1.0e-12)
    expected = {
        ElevenActionId.VIEW_UP: up,
        ElevenActionId.VIEW_UP_RIGHT: (up + right) / math.sqrt(2.0),
        ElevenActionId.VIEW_RIGHT: right,
        ElevenActionId.VIEW_DOWN_RIGHT: (-up + right) / math.sqrt(2.0),
        ElevenActionId.VIEW_DOWN: -up,
        ElevenActionId.VIEW_DOWN_LEFT: (-up - right) / math.sqrt(2.0),
        ElevenActionId.VIEW_LEFT: -right,
        ElevenActionId.VIEW_UP_LEFT: (up - right) / math.sqrt(2.0),
    }
    for action, direction in expected.items():
        np.testing.assert_allclose(grid_direction_world(action, up, right), direction, atol=1.0e-12)
        target = view_target_axis(optical, direction, angle_rad=math.radians(15.0))
        angle = math.acos(np.clip(float(optical @ target), -1.0, 1.0))
        assert math.degrees(angle) == pytest.approx(15.0, abs=1.0e-10)


@pytest.mark.parametrize(
    "quaternion",
    [
        (1.0, 0.0, 0.0, 0.0),
        tuple(_axis_angle_quaternion([0.0, 1.0, 0.0], math.radians(60.0))),
        tuple(_axis_angle_quaternion([0.0, 1.0, 0.0], math.radians(90.0))),
        tuple(_axis_angle_quaternion([0.3, 0.7, 0.2], 1.13)),
    ],
)
def test_support_material_offset_stays_local_while_world_point_tracks_pose(quaternion):
    normal = np.asarray([0.0, 0.0, 1.0])
    start = _state(quaternion)
    frozen = freeze_support_material_point(start, normal, radius_m=0.0065, half_length_m=0.006)
    start_world, _ = reconstruct_material_point(start, frozen.local_offset_m)

    moved = _state(
        _axis_angle_quaternion([1.0, 0.0, 0.0], 0.4),
        position=(0.02, -0.01, 0.03),
    )
    moved_world, _ = reconstruct_material_point(moved, frozen.local_offset_m)
    assert frozen.local_offset_m.shape == (3,)
    assert np.linalg.norm(moved_world - start_world) > 0.01
    np.testing.assert_allclose(frozen.anchor_world_m, start_world)


def test_support_tangent_error_contains_no_normal_component():
    normal = np.asarray([0.2, -0.3, 0.93])
    normal /= np.linalg.norm(normal)
    error = support_tangent_error(
        anchor_world_m=np.asarray([1.0, 2.0, 3.0]),
        support_world_m=np.asarray([0.2, -0.4, 0.7]),
        normal_world=normal,
    )
    assert float(error @ normal) == pytest.approx(0.0, abs=1.0e-12)


def test_contact_regions_use_only_axial_coordinate_and_history_uses_physics_substeps():
    assert classify_contact_region(0.0060001, 0.006) is ContactRegion.CAMERA_HEMISPHERE
    assert classify_contact_region(-0.0060001, 0.006) is ContactRegion.NONCAMERA_HEMISPHERE
    assert classify_contact_region(0.006, 0.006) is ContactRegion.SIDEWALL
    assert classify_contact_region(-0.006, 0.006) is ContactRegion.SIDEWALL

    history = SideContactHistory(capacity_substeps=12)
    for substep in range(20):
        sigma = 0.0 if substep == 8 else (0.007 if substep == 10 else -0.007)
        history.append(
            ContactSample(
                physics_substep=substep,
                point_world=np.asarray([substep, 0.0, 0.0]),
                normal_world=np.asarray([0.0, 0.0, 1.0]),
                axial_coordinate_m=sigma,
                impulse_n_s=0.0 if substep % 2 else 999.0,
            )
        )

    assert not history.had_sidewall_contact(current_substep=20, last_n_substeps=12)
    camera = history.camera_constraints(current_substep=20, last_n_substeps=12)
    assert [sample.physics_substep for sample in camera] == [10]
    assert all(sample.region is ContactRegion.CAMERA_HEMISPHERE for sample in camera)
    assert len(history) <= 12


def test_capsule_directed_axis_is_local_negative_z():
    np.testing.assert_allclose(capsule_axis_world(_state()), [0.0, 0.0, -1.0])

