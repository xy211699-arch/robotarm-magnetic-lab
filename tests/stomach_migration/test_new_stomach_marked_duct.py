"""Tests for the user-marked small-duct candidate mask."""

from types import SimpleNamespace

import numpy as np
from scipy.sparse.csgraph import dijkstra

from robotarm_magnetic_lab.coverage.new_stomach_marked_duct import (
    MarkedDuctSeed,
    _unique_edge_graph,
    candidate_marked_duct_mask,
)


def test_shared_mesh_edge_has_its_physical_length_once():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    graph = _unique_edge_graph(vertices, triangles)
    assert np.isclose(graph[0, 2], np.sqrt(2))
    assert np.isclose(dijkstra(graph, directed=False, indices=0)[2], np.sqrt(2))


def test_marked_duct_follows_cylinder_from_matched_open_boundary():
    rings, sectors, radius = 4, 32, 0.003
    vertices = []
    for ring in range(rings):
        y = 0.060 + ring * 0.005
        for angle in np.linspace(0, 2 * np.pi, sectors, endpoint=False):
            vertices.append(
                [0.185 + radius * np.cos(angle), y, 0.068 + radius * np.sin(angle)]
            )
    triangles = []
    for ring in range(rings - 1):
        for sector in range(sectors):
            a = ring * sectors + sector
            b = ring * sectors + (sector + 1) % sectors
            triangles.extend(((a, b, a + sectors), (b, b + sectors, a + sectors)))
    # Detached reference triangle sets the same relative-coordinate origin as
    # the accepted new-stomach scene; it must remain outside the duct mask.
    vertices.extend(([0, 0, 0], [0.001, 0, 0], [0, 0.001, 0]))
    triangles.append((rings * sectors, rings * sectors + 1, rings * sectors + 2))
    reference = SimpleNamespace(
        vertices_world=np.asarray(vertices),
        triangles=np.asarray(triangles, dtype=np.int64),
    )
    result = candidate_marked_duct_mask(
        reference,
        seeds=(MarkedDuctSeed("test", (0.185, 0.060, 0.068), radius, 0.010),),
    )
    assert result.connected_components == 1
    assert len(result.face_indices) > 0
    assert (len(triangles) - 1) not in result.face_indices
    assert result.area_m2 > 0
