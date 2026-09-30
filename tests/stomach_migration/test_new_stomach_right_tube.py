"""Checks for the complete right-side tubular-appendage candidate."""

from types import SimpleNamespace

import numpy as np
import pytest

from robotarm_magnetic_lab.coverage.new_stomach_tubes import candidate_right_tube_mask


def _connected_strip():
    vertices = np.array(
        [[x, y, 0.0] for x in np.linspace(0.0, 0.04, 5) for y in (0.0, 0.01)]
    )
    triangles = []
    for column in range(4):
        base = 2 * column
        triangles.extend(
            ((base, base + 1, base + 2), (base + 1, base + 3, base + 2))
        )
    return SimpleNamespace(
        vertices_world=vertices,
        triangles=np.asarray(triangles, dtype=np.int64),
    )


def test_neck_cut_selects_the_entire_connected_far_side():
    reference = _connected_strip()
    result = candidate_right_tube_mask(
        reference, cut_x_from_min_m=0.02, max_y_from_min_m=0.02
    )
    centers = reference.vertices_world[reference.triangles].mean(axis=1)
    assert set(result.face_indices.tolist()) == set(np.flatnonzero(centers[:, 0] >= 0.02))
    assert result.connected_components == 1
    assert result.area_m2 > 0
    assert np.isclose(result.bounds_from_min_m[0, 0], centers[result.face_indices, 0].min())


def test_neck_cut_rejects_out_of_region_or_disconnected_mesh():
    reference = _connected_strip()
    with pytest.raises(ValueError, match="annotated Y"):
        candidate_right_tube_mask(
            reference, cut_x_from_min_m=0.02, max_y_from_min_m=0.004
        )
    vertices = np.vstack(
        (reference.vertices_world, [[0.05, 0, 0], [0.06, 0, 0], [0.05, 0.01, 0]])
    )
    faces = np.vstack((reference.triangles, [[10, 11, 12]]))
    detached = SimpleNamespace(vertices_world=vertices, triangles=faces)
    with pytest.raises(ValueError, match="disconnected"):
        candidate_right_tube_mask(
            detached, cut_x_from_min_m=0.02, max_y_from_min_m=0.02
        )
