from __future__ import annotations

import numpy as np
import pytest
import torch

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force_macro import (
    PointForce,
    equivalent_com_wrench,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    ParameterizedForceMode,
    parameterized_endpoint_forces,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.vectorized_parameterized_force import (
    batched_equivalent_com_wrench,
    batched_parameterized_endpoint_forces,
)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_batched_modes_match_scalar_controller(alpha):
    modes = torch.arange(6, dtype=torch.int64)
    masses = torch.full((6,), 0.005735, dtype=torch.float64)
    axes = torch.tensor([[1.0, 0.0, 0.0]] * 6, dtype=torch.float64)
    batch = batched_parameterized_endpoint_forces(
        modes, torch.full((6,), alpha, dtype=torch.float64), masses, axes
    )
    for row, mode in enumerate(ParameterizedForceMode):
        scalar = parameterized_endpoint_forces(
            mode,
            alpha,
            mass_kg=float(masses[row]),
            camera_axis_world=axes[row].numpy(),
        )
        np.testing.assert_allclose(
            batch.camera_forces_world[row].numpy(), scalar.camera_force_world, atol=1e-15
        )
        np.testing.assert_allclose(
            batch.other_forces_world[row].numpy(), scalar.other_force_world, atol=1e-15
        )
        np.testing.assert_allclose(
            batch.directions_world[row].numpy(), scalar.direction_world, atol=1e-15
        )
        assert batch.force_ratios[row].item() == pytest.approx(scalar.force_ratio)


def test_vertical_active_axis_reports_exact_environment_rows():
    with pytest.raises(ValueError, match=r"undefined lateral direction.*\[1\]"):
        batched_parameterized_endpoint_forces(
            torch.tensor([0, 1]),
            torch.tensor([0.0, 0.5]),
            torch.tensor([0.005735, 0.005735]),
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        )


@pytest.mark.parametrize(
    "modes,alpha,masses,axes,match",
    [
        ([0, 6], [0.0, 0.0], [1.0, 1.0], [[1, 0, 0], [1, 0, 0]], "mode IDs"),
        ([0], [1.1], [1.0], [[1, 0, 0]], "alpha"),
        ([0], [0.0], [0.0], [[1, 0, 0]], "masses"),
        ([0], [0.0], [1.0], [[0, 0, 0]], "camera axes"),
    ],
)
def test_invalid_batches_are_rejected(modes, alpha, masses, axes, match):
    with pytest.raises(ValueError, match=match):
        batched_parameterized_endpoint_forces(
            torch.tensor(modes),
            torch.tensor(alpha),
            torch.tensor(masses),
            torch.tensor(axes),
        )


def test_batch_rows_and_device_are_preserved():
    with pytest.raises(ValueError, match="rows must match"):
        batched_parameterized_endpoint_forces(
            torch.tensor([1, 2]),
            torch.tensor([0.5]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([[1.0, 0.0, 0.0]] * 2),
        )


def test_batched_equivalent_wrench_matches_scalar_cross_products():
    modes = torch.tensor([1, 3, 5], dtype=torch.int64)
    alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    masses = torch.full((3,), 0.005735, dtype=torch.float64)
    axes = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
    command = batched_parameterized_endpoint_forces(modes, alpha, masses, axes)
    camera = torch.tensor([[0.01, 0.0, 0.0]] * 3, dtype=torch.float64)
    other = torch.tensor([[-0.01, 0.0, 0.0]] * 3, dtype=torch.float64)
    com = torch.tensor([[0.0, 0.003, 0.0]] * 3, dtype=torch.float64)
    force, torque = batched_equivalent_com_wrench(
        command.camera_forces_world,
        command.other_forces_world,
        camera,
        other,
        com,
    )
    for row in range(3):
        points = []
        if np.linalg.norm(command.camera_forces_world[row].numpy()) > 0:
            points.append(
                PointForce(
                    "camera",
                    camera[row].numpy(),
                    command.camera_forces_world[row].numpy(),
                )
            )
        if np.linalg.norm(command.other_forces_world[row].numpy()) > 0:
            points.append(
                PointForce(
                    "other",
                    other[row].numpy(),
                    command.other_forces_world[row].numpy(),
                )
            )
        expected_force, expected_torque = equivalent_com_wrench(tuple(points), com[row].numpy())
        np.testing.assert_allclose(force[row].numpy(), expected_force, atol=1e-15)
        np.testing.assert_allclose(torque[row].numpy(), expected_torque, atol=1e-15)
