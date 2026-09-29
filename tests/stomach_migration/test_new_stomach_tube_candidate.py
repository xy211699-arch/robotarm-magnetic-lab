"""Synthetic geometry checks for the two-ended candidate coverage mask."""

from types import SimpleNamespace

import numpy as np
import pytest

from robotarm_magnetic_lab.coverage.new_stomach_tubes import (
    candidate_tube_mask,
    infer_inward_direction,
)


def _cylinder(origin, direction, radius=0.01, rings=7, sectors=64):
    vertices = []
    for step in range(rings):
        center = np.asarray(origin) + direction * (step * 0.005)
        for angle in np.linspace(0, 2 * np.pi, sectors, endpoint=False):
            vertices.append(center + [0, radius * np.cos(angle), radius * np.sin(angle)])
    triangles = []
    for step in range(rings - 1):
        for sector in range(sectors):
            current = step * sectors + sector
            following = step * sectors + (sector + 1) % sectors
            upper = current + sectors
            upper_following = following + sectors
            triangles.extend(((current, following, upper), (following, upper_following, upper)))
    return np.asarray(vertices), np.asarray(triangles, dtype=np.int64)


def test_inward_axis_is_inferred_from_wall_support_not_opening_sign():
    plus = np.array([[0.005, 0.01, 0.0]] * 100)
    minus = np.array([[-0.005, 0.01, 0.0]] * 5)
    direction, support = infer_inward_direction(
        np.concatenate((plus, minus)),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
    )
    assert direction.tolist() == [1.0, 0.0, 0.0]
    assert support == (100, 5)
    with pytest.raises(ValueError, match="ambiguous"):
        balanced_minus = np.array([[-0.005, 0.01, 0.0]] * 30)
        infer_inward_direction(np.concatenate((plus[:30], balanced_minus)),
                               np.zeros(3), np.array([1.0, 0.0, 0.0]))


def test_cylindrical_tube_mask_keeps_body_and_produces_disjoint_ends():
    left, lf = _cylinder([0.0, 0.0, 0.0], np.array([1.0, 0.0, 0.0]))
    right, rf = _cylinder([0.1, 0.0, 0.0], np.array([-1.0, 0.0, 0.0]))
    body = np.array([[0.04, 0.04, -0.01], [0.06, 0.04, -0.01], [0.05, 0.04, 0.01]])
    vertices = np.concatenate((left, right, body))
    faces = np.concatenate((
        lf,
        rf + len(left),
        np.array([[len(left) + len(right), len(left) + len(right) + 1,
                   len(left) + len(right) + 2]]),
    ))
    reference = SimpleNamespace(vertices_world=vertices, triangles=faces)
    result = candidate_tube_mask(
        reference,
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert np.allclose(result.inward_directions, [[1, 0, 0], [-1, 0, 0]])
    assert len(result.per_tube_face_indices[0]) == 5 * 2 * 64
    assert len(result.per_tube_face_indices[1]) == 5 * 2 * 64
    assert result.per_tube_component_counts == (1, 1)
    assert not np.intersect1d(*result.per_tube_face_indices).size
    assert len(result.reachable_face_indices) == 2 * 64 * 2 + 1
    assert result.excluded_area_m2 > 0
    assert result.reachable_area_m2 > 0
