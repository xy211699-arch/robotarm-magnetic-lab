from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface import (
    LocalFrame,
    Spherocylinder,
    SurfaceNavigationMesh,
    quintic,
)


@dataclass(frozen=True)
class FakeReference:
    vertices_world: np.ndarray
    triangles: np.ndarray


def unit_square_mesh() -> SurfaceNavigationMesh:
    reference = FakeReference(
        vertices_world=np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
        ),
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int),
    )
    return SurfaceNavigationMesh.from_reference(reference, inward_sign=1)


def disconnected_parallel_sheets(gap_m: float) -> SurfaceNavigationMesh:
    first = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    second = first + np.asarray([0.0, 0.0, gap_m])
    reference = FakeReference(
        vertices_world=np.concatenate((first, second)),
        triangles=np.asarray(
            [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=int
        ),
    )
    return SurfaceNavigationMesh.from_reference(reference, inward_sign=1)


def test_local_search_never_jumps_to_disconnected_nearby_sheet():
    mesh = disconnected_parallel_sheets(gap_m=0.001)
    hit = mesh.advance(
        triangle_id=0,
        point_world=np.array([0.25, 0.25, 0.0]),
        tangent_delta_world=np.array([0.2, 0.0, 0.0]),
        recovery_radius_m=0.02,
    )
    assert hit.component_id == mesh.component_ids[0]
    assert hit.triangle_id in {0, 1}
    assert hit.point_world[2] == pytest.approx(0.0)


def test_exact_closest_hit_uses_distance_then_triangle_id_tie_break():
    mesh = unit_square_mesh()
    hit = mesh.closest_hit(np.asarray([0.5, 0.5, 0.25]))
    assert hit.triangle_id == 0
    np.testing.assert_allclose(hit.point_world, [0.5, 0.5, 0.0])


def test_open_edge_is_reported_as_boundary_not_surface_loss():
    mesh = unit_square_mesh()
    hit = mesh.advance(
        triangle_id=1,
        point_world=np.array([0.9, 0.5, 0.0]),
        tangent_delta_world=np.array([0.2, 0.0, 0.0]),
        recovery_radius_m=0.1,
    )
    assert hit.boundary_limited
    assert hit.point_world[0] == pytest.approx(1.0)


def test_bent_strip_builds_adjacency_and_oriented_normals():
    reference = FakeReference(
        vertices_world=np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=int),
    )
    mesh = SurfaceNavigationMesh.from_reference(reference, inward_sign=-1)
    assert 1 in mesh.adjacency[0]
    assert 2 in mesh.adjacency[1]
    np.testing.assert_allclose(mesh.normals[0], [0, 0, -1])
    np.testing.assert_allclose(mesh.normals[3], [1, 0, 0])


def test_direction_bins_use_image_up_reference():
    frame = LocalFrame(
        point_world=np.zeros(3),
        normal_world=np.array([0.0, 0.0, 1.0]),
        image_up_tangent_world=np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(frame.direction(math.radians(0)), [1, 0, 0])
    np.testing.assert_allclose(frame.direction(math.radians(90)), [0, 1, 0], atol=1e-12)


def test_spherocylinder_support_height_changes_with_tilt():
    capsule = Spherocylinder(radius_m=0.005, cylinder_half_length_m=0.0075)
    upright = capsule.support_distance(np.array([0, 0, 1]), np.array([0, 0, 1]))
    side = capsule.support_distance(np.array([1, 0, 0]), np.array([0, 0, 1]))
    assert upright == pytest.approx(0.0125)
    assert side == pytest.approx(0.005)
    assert capsule.effective_roll_radius() == pytest.approx(0.005)


@pytest.mark.parametrize("tau, expected", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_quintic_reference_values(tau, expected):
    assert quintic(tau) == pytest.approx(expected)
