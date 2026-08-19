from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotarm_magnetic_lab.magnetics import FiniteMagnetSystem, load_config
from virtual_magnet import load_profile
from virtual_magnet.pose_inverse import (
    PoseInverseState,
    integrate_pose_increment,
    numerical_pose_jacobian,
    solve_pose_increment,
)


def _quat(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()


class LinearPoseWrench:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float64)

    def __call__(self, position, rotation):
        coordinates = np.concatenate((position, Rotation.from_matrix(rotation).as_rotvec()))
        return self.matrix @ coordinates


def test_central_pose_jacobian_uses_si_translation_and_local_rotation():
    matrix = np.diag([2.0, 3.0, 4.0, 0.5, 0.75, 1.25])
    model = LinearPoseWrench(matrix)
    position = np.array([0.01, -0.02, 0.08])
    rotation = np.eye(3)
    wrench, jacobian = numerical_pose_jacobian(model, position, rotation, 1.0e-5, 1.0e-5)
    np.testing.assert_allclose(wrench, model(position, rotation), atol=1.0e-12)
    np.testing.assert_allclose(jacobian, matrix, rtol=1.0e-6, atol=1.0e-9)


def test_weighted_damped_solve_reduces_reachable_residual():
    model = LinearPoseWrench(np.diag([1.0, 2.0, 3.0, 0.5, 0.7, 0.9]))
    state = PoseInverseState(
        position=np.array([0.0, 0.0, 0.08]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        capsule_magnet_position=np.zeros(3),
        capsule_magnet_rotation=np.eye(3),
        nominal_position=np.array([0.0, 0.0, 0.08]),
        nominal_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    desired = model(np.array([0.0005, -0.0004, 0.0803]), Rotation.from_rotvec([0.005, -0.004, 0.003]).as_matrix())
    before = np.linalg.norm(desired - model(state.position, np.eye(3)))
    result = solve_pose_increment(
        model,
        state,
        desired,
        weights=np.ones(6),
        translation_step_m=1.0e-5,
        rotation_step_rad=1.0e-5,
        damping=1.0e-8,
        relative_regularization=0.0,
        translation_trust_m=0.002,
        rotation_trust_rad=0.02,
        minimum_separation_m=0.04,
        maximum_separation_m=0.14,
        maximum_relative_angle_rad=np.pi,
        condition_limit=1.0e12,
    )
    after = np.linalg.norm(desired - model(result.position, Rotation.from_quat(result.quaternion_xyzw).as_matrix()))
    assert after < before * 1.0e-3
    assert not result.solver_saturated


def test_trust_region_and_quaternion_sign_are_bounded():
    position = np.array([0.0, 0.0, 0.08])
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    next_position, next_quaternion, clipped = integrate_pose_increment(
        position,
        quaternion,
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        translation_trust_m=0.001,
        rotation_trust_rad=0.02,
    )
    assert np.linalg.norm(next_position - position) == pytest.approx(0.001)
    assert Rotation.from_quat(next_quaternion).magnitude() == pytest.approx(0.02)
    assert np.dot(next_quaternion, quaternion) >= 0.0
    assert np.linalg.norm(next_quaternion) == pytest.approx(1.0)
    assert clipped


def test_singular_model_holds_last_finite_pose():
    model = LinearPoseWrench(np.zeros((6, 6)))
    state = PoseInverseState(
        position=np.array([0.0, 0.0, 0.08]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        capsule_magnet_position=np.zeros(3),
        capsule_magnet_rotation=np.eye(3),
        nominal_position=np.array([0.0, 0.0, 0.08]),
        nominal_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    result = solve_pose_increment(
        model,
        state,
        np.ones(6),
        weights=np.ones(6),
        translation_step_m=1.0e-4,
        rotation_step_rad=1.0e-3,
        damping=1.0e-4,
        relative_regularization=0.01,
        translation_trust_m=0.001,
        rotation_trust_rad=0.02,
        minimum_separation_m=0.04,
        maximum_separation_m=0.14,
        maximum_relative_angle_rad=np.pi,
        condition_limit=1.0e8,
    )
    np.testing.assert_allclose(result.position, state.position)
    np.testing.assert_allclose(result.quaternion_xyzw, state.quaternion_xyzw)
    assert result.solver_saturated
    assert np.isfinite(result.condition_number) or np.isinf(result.condition_number)


def test_finite_model_iteration_reduces_nearby_reachable_wrench_residual():
    profile = load_profile()
    finite = FiniteMagnetSystem(load_config())
    capsule_position = np.zeros(3)
    capsule_rotation = np.eye(3)

    def model(position, rotation):
        force, torque = finite.force_torque_si(position, rotation, capsule_position, capsule_rotation)
        return np.concatenate((force, torque))

    desired_position = np.array([0.003, -0.002, 0.18])
    desired_rotation = Rotation.from_rotvec([0.02, -0.015, 0.01]).as_matrix()
    desired = model(desired_position, desired_rotation)
    state = PoseInverseState(
        position=np.array([0.0, 0.0, 0.18]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        capsule_magnet_position=capsule_position,
        capsule_magnet_rotation=capsule_rotation,
        nominal_position=np.array([0.0, 0.0, 0.18]),
        nominal_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    start_residual = np.linalg.norm(desired - model(state.position, np.eye(3)))
    residual = start_residual
    for _ in range(8):
        result = solve_pose_increment(
            model,
            state,
            desired,
            weights=np.asarray(profile.force_weights + profile.torque_weights),
            translation_step_m=profile.translation_fd_step_m,
            rotation_step_rad=profile.rotation_fd_step_rad,
            damping=profile.inverse_damping,
            relative_regularization=0.0,
            translation_trust_m=profile.translation_trust_m,
            rotation_trust_rad=profile.rotation_trust_rad,
            minimum_separation_m=profile.minimum_separation_m,
            maximum_separation_m=profile.maximum_separation_m,
            maximum_relative_angle_rad=profile.maximum_relative_angle_rad,
            condition_limit=profile.condition_limit,
        )
        state = PoseInverseState(
            position=result.position,
            quaternion_xyzw=result.quaternion_xyzw,
            capsule_magnet_position=capsule_position,
            capsule_magnet_rotation=capsule_rotation,
            nominal_position=state.nominal_position,
            nominal_quaternion_xyzw=state.nominal_quaternion_xyzw,
        )
        residual = np.linalg.norm(desired - model(state.position, Rotation.from_quat(state.quaternion_xyzw).as_matrix()))
    assert np.isfinite(residual)
    assert residual < start_residual * 0.35
