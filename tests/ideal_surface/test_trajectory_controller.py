from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface import (
    ContactAssessment,
    ControllerSnapshot,
    IdealActionStatus,
    IdealSurfaceAction,
    IdealSurfaceConfig,
    IdealSurfaceController,
    SurfaceFlags,
    SurfaceNavigationMesh,
    quintic,
)


class _Plane:
    vertices_world = np.asarray([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], float)
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], int)


def snapshot(theta_deg=0.0, phi_deg=0.0, side_contact=False):
    theta, phi = math.radians(theta_deg), math.radians(phi_deg)
    # phi=0 follows projected camera image-up (+Y); positive phi follows e2=-X.
    direction = np.asarray([-math.sin(phi), math.cos(phi), 0.0])
    axis = math.sin(theta) * direction + np.asarray([0.0, 0.0, math.cos(theta)])
    image_up = math.cos(theta) * direction - np.asarray([0.0, 0.0, math.sin(theta)])
    cfg = IdealSurfaceConfig()
    height = cfg.capsule_radius_m + cfg.capsule_cylinder_half_length_m * abs(axis[2])
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface import (
        orientation_from_axis_and_image_up,
        quaternion_wxyz_to_matrix,
    )
    quaternion = orientation_from_axis_and_image_up(axis, image_up)
    realized_image_up = quaternion_wxyz_to_matrix(quaternion)[:, 1]
    return ControllerSnapshot(
        sim_time_s=0.0,
        position_world=np.asarray([0.0, 0.0, height]),
        quaternion_for_sim=quaternion,
        axis_world=axis,
        image_up_world=realized_image_up,
        surface_point_world=np.zeros(3),
        surface_normal_world=np.asarray([0, 0, 1]),
        surface_triangle_id=0,
        theta_rad=theta,
        phi_rad=phi,
        flags=SurfaceFlags(upright=theta_deg <= 5.0, side_contact=side_contact),
    )


def controller(initial=None, assessor=None):
    value = IdealSurfaceController(
        SurfaceNavigationMesh.from_reference(_Plane(), inward_sign=1),
        cfg=IdealSurfaceConfig(),
        pose_assessor=assessor,
    )
    value.reset(initial or snapshot())
    return value


def run_to_done(value):
    output = None
    for _ in range(240):
        output = value.step(1 / 240)
    assert output is not None and output.result is not None
    return output.result


@pytest.mark.parametrize("tau, expected", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_quintic_endpoints(tau, expected):
    assert quintic(tau) == pytest.approx(expected)


def test_start_tilt_maps_each_id_to_one_unique_axis():
    value = controller()
    outputs = []
    for action_id in range(1, 9):
        value.reset(snapshot())
        assert value.submit(action_id, value.snapshot, request_id=action_id)
        outputs.append(run_to_done(value).final_axis_world)
    assert len({tuple(np.round(axis, 8)) for axis in outputs}) == 8


def test_start_tilt_rotates_center_about_fixed_support_anchor():
    before = snapshot()
    value = controller(before)
    value.submit(IdealSurfaceAction.START_TILT_000, before, 101)
    result = run_to_done(value)
    anchor = np.asarray([0.0, 0.0, 0.0])
    expected_center = (
        anchor
        + IdealSurfaceConfig().capsule_cylinder_half_length_m * result.final_axis_world
        + np.asarray([0.0, 0.0, IdealSurfaceConfig().capsule_radius_m])
    )
    np.testing.assert_allclose(result.final_position_world, expected_center, atol=1.0e-9)
    assert math.degrees(result.final_tilt_rad) == pytest.approx(15.0, abs=0.2)


def test_precession_keeps_tilt_and_changes_azimuth_by_fifteen_degrees():
    value = controller(snapshot(theta_deg=45, phi_deg=0))
    value.submit(IdealSurfaceAction.PRECESS_POS, value.snapshot, 1)
    result = run_to_done(value)
    assert result.final_tilt_rad == pytest.approx(math.radians(45), abs=math.radians(0.2))
    assert result.final_azimuth_rad == pytest.approx(math.radians(15), abs=math.radians(0.2))


def test_contact_limited_motion_holds_until_one_second_boundary():
    calls = 0

    def block_after(pose, active_triangle, cfg):
        nonlocal calls
        calls += 1
        blocked = calls >= 145
        return ContactAssessment(
            support_valid=True, side_contact=False, contact_limited=blocked,
            boundary_limited=False, hard_failure=False,
            maximum_penetration_m=0.0001 if blocked else 0.0,
            support_point_world=np.zeros(3), support_normal_world=np.asarray([0, 0, 1]),
            active_triangle=active_triangle, barrel_clearances_m=np.ones(2),
            barrel_axial_parameters=np.asarray([-1.0, 1.0]),
        )

    value = controller(snapshot(theta_deg=15), assessor=block_after)
    value.submit(IdealSurfaceAction.TILT_MORE, value.snapshot, 4)
    outputs = [value.step(1 / 240) for _ in range(240)]
    assert outputs[-1].result.status is IdealActionStatus.DONE
    assert outputs[-1].result.contact_limited
    assert outputs[-1].result.duration_s == pytest.approx(1.0)
    np.testing.assert_allclose(outputs[150].position_world, outputs[-1].position_world)


def test_positive_roll_obeys_right_hand_no_slip_sign():
    before = snapshot(theta_deg=90, phi_deg=0, side_contact=True)
    value = controller(before)
    value.submit(IdealSurfaceAction.ROLL_POS, before, 9)
    result = run_to_done(value)
    expected = -0.010 * np.cross(before.surface_normal_world, before.axis_tangent_world)
    np.testing.assert_allclose(result.final_position_world - before.position_world, expected, atol=1e-4)


def test_roll_follows_curved_surface_without_latching_at_first_normal_correction():
    before = snapshot(theta_deg=90, phi_deg=0, side_contact=True)

    def curved_clearance(pose, active_triangle, cfg):
        required_height = cfg.capsule_radius_m + 0.2 * abs(float(pose.center_world[0]))
        penetration = max(0.0, required_height - float(pose.center_world[2]))
        return ContactAssessment(
            support_valid=True,
            side_contact=True,
            contact_limited=penetration > cfg.planned_penetration_radius_fraction * cfg.capsule_radius_m,
            boundary_limited=False,
            hard_failure=penetration > cfg.hard_penetration_radius_fraction * cfg.capsule_radius_m,
            maximum_penetration_m=penetration,
            support_point_world=np.asarray([pose.center_world[0], pose.center_world[1], 0.0]),
            support_normal_world=np.asarray([0.0, 0.0, 1.0]),
            active_triangle=active_triangle,
            barrel_clearances_m=np.zeros(5),
            barrel_axial_parameters=np.linspace(-0.5, 0.5, 5),
        )

    value = controller(before, assessor=curved_clearance)
    value.submit(IdealSurfaceAction.ROLL_POS, before, 91)
    result = run_to_done(value)
    assert result.status is IdealActionStatus.DONE
    assert not result.contact_limited
    assert result.final_position_world[0] - before.position_world[0] == pytest.approx(0.010, abs=1e-4)
    assert result.final_position_world[2] > before.position_world[2]


@pytest.mark.parametrize("theta_deg, expected_deg", [(75.0, 90.0), (90.0, 105.0)])
def test_rise_rotates_about_opposite_non_camera_support_anchor(theta_deg, expected_deg):
    before = snapshot(theta_deg=theta_deg, phi_deg=0, side_contact=theta_deg == 90.0)
    value = controller(before)
    value.submit(IdealSurfaceAction.RISE, before, 105)
    result = run_to_done(value)
    assert math.degrees(result.final_tilt_rad) == pytest.approx(expected_deg, abs=0.2)
    cfg = IdealSurfaceConfig()
    # Camera optical axis is local -Z, therefore the opposite end is +axis.
    opposite_anchor = (
        before.position_world
        + cfg.capsule_cylinder_half_length_m * before.axis_world
        - cfg.capsule_radius_m * before.surface_normal_world
    )
    expected_center = (
        opposite_anchor
        - cfg.capsule_cylinder_half_length_m * result.final_axis_world
        + cfg.capsule_radius_m * before.surface_normal_world
    )
    np.testing.assert_allclose(result.final_position_world, expected_center, atol=1.0e-9)


def test_logical_upright_accepts_both_axis_poles_after_stability_window():
    before = snapshot(theta_deg=165, phi_deg=0, side_contact=False)
    value = controller(before)
    value.submit(IdealSurfaceAction.RISE, before, 106)
    result = run_to_done(value)
    assert math.degrees(result.final_tilt_rad) == pytest.approx(180.0, abs=0.2)
    assert result.status is IdealActionStatus.DONE
    assert value.snapshot.flags.upright


def test_inverted_upright_tilts_to_side_about_non_camera_end():
    before = snapshot(theta_deg=180, phi_deg=0, side_contact=False)
    before = replace(before, flags=SurfaceFlags(upright=True, side_contact=False))
    value = controller(before)
    value.submit(IdealSurfaceAction.START_TILT_000, before, 107)
    result = run_to_done(value)
    assert math.degrees(result.final_tilt_rad) == pytest.approx(165.0, abs=0.2)
    assert not result.contact_limited
    value.acknowledge_result()
    for request_id in range(108, 113):
        value.submit(IdealSurfaceAction.TILT_MORE, value.snapshot, request_id)
        result = run_to_done(value)
        assert not result.contact_limited
        value.acknowledge_result()
    assert math.degrees(value.snapshot.theta_rad) == pytest.approx(90.0, abs=0.2)
    assert value.snapshot.flags.side_contact


def test_roll_first_substep_is_continuous_from_corrected_start_pose():
    before = snapshot(theta_deg=90, phi_deg=0, side_contact=True)
    before = replace(before, position_world=before.position_world + np.asarray([0.0, 0.0, 0.001]))
    value = controller(before)
    value.submit(IdealSurfaceAction.ROLL_POS, before, 102)
    output = value.step(1 / 240)
    assert np.linalg.norm(output.position_world - before.position_world) < 1.0e-6


def test_rise_first_substep_is_continuous_after_corrected_roll_pose():
    before = snapshot(theta_deg=90, phi_deg=0, side_contact=True)
    before = replace(before, position_world=before.position_world + np.asarray([0.0, 0.0, 0.001]))
    value = controller(before)
    value.submit(IdealSurfaceAction.RISE, before, 104)
    output = value.step(1 / 240)
    assert np.linalg.norm(output.position_world - before.position_world) < 1.0e-6


def test_open_boundary_latches_boundary_flag_without_contact_flag():
    before = snapshot(theta_deg=90, phi_deg=0, side_contact=True)
    surface = np.asarray([1.999, 0.0, 0.0])
    before = replace(
        before,
        position_world=surface + np.asarray([0.0, 0.0, IdealSurfaceConfig().capsule_radius_m]),
        surface_point_world=surface,
        surface_triangle_id=0,
    )
    value = controller(before)
    value.submit(IdealSurfaceAction.ROLL_POS, before, 90)
    result = run_to_done(value)
    assert result.boundary_limited
    assert not result.contact_limited


def test_invalid_masked_action_is_one_second_no_effect():
    value = controller(snapshot())
    assert value.submit(IdealSurfaceAction.ROLL_POS, value.snapshot, 20)
    result = run_to_done(value)
    assert result.no_effect
    np.testing.assert_allclose(result.final_position_world, snapshot().position_world)


def test_request_ids_are_deduplicated_and_result_requires_acknowledgement():
    value = controller()
    assert value.submit(IdealSurfaceAction.HOLD, value.snapshot, 5)
    run_to_done(value)
    assert not value.ready
    assert not value.submit(IdealSurfaceAction.HOLD, value.snapshot, 6)
    value.acknowledge_result()
    assert value.ready
    assert not value.submit(IdealSurfaceAction.HOLD, value.snapshot, 5)


def test_hard_failure_enters_terminal_fault_and_reset_recovers():
    def fail(pose, active_triangle, cfg):
        return ContactAssessment(
            support_valid=False, side_contact=False, contact_limited=True,
            boundary_limited=False, hard_failure=True, maximum_penetration_m=0.01,
            support_point_world=np.zeros(3), support_normal_world=np.asarray([0, 0, 1]),
            active_triangle=active_triangle, barrel_clearances_m=-np.ones(2),
            barrel_axial_parameters=np.asarray([-1.0, 1.0]),
        )

    value = controller(snapshot(theta_deg=15), assessor=fail)
    value.submit(IdealSurfaceAction.TILT_MORE, value.snapshot, 1)
    output = value.step(1 / 240)
    assert output.result.status is IdealActionStatus.HARD_FAILURE
    assert not value.ready
    value.reset(snapshot())
    assert value.ready


def test_rejected_future_hard_penetration_clips_to_last_safe_pose():
    def reject_after_twenty_degrees(pose, active_triangle, cfg):
        tilt = math.acos(float(np.clip(pose.axis_world[2], -1.0, 1.0)))
        hard = tilt > math.radians(20.0)
        return ContactAssessment(
            support_valid=not hard,
            side_contact=False,
            contact_limited=hard,
            boundary_limited=False,
            hard_failure=hard,
            maximum_penetration_m=0.001 if hard else 0.0,
            support_point_world=np.zeros(3),
            support_normal_world=np.asarray([0, 0, 1]),
            active_triangle=active_triangle,
            barrel_clearances_m=np.ones(2),
            barrel_axial_parameters=np.asarray([-0.5, 0.5]),
        )

    before = snapshot(theta_deg=15)
    value = controller(before, assessor=reject_after_twenty_degrees)
    value.submit(IdealSurfaceAction.TILT_MORE, before, 103)
    result = run_to_done(value)
    assert result.status is IdealActionStatus.DONE
    assert result.contact_limited
    assert math.degrees(result.final_tilt_rad) <= 20.0
    assert result.maximum_penetration_m == 0.0
