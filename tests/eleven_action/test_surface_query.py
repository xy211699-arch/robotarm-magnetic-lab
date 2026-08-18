import numpy as np

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.surface_query import (
    FlatSurfaceQuery,
    StomachSurfaceQuery,
    TriangleMeshSurfaceQuery,
)


def _nonuniform_mesh():
    # Triangle 0 is nearest the query.  Triangles 1 and 2 share one of its
    # vertices; triangle 3 is disconnected and must not enter the one-ring.
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.4], [0.0, 3.0, 0.6],
            [9.0, 9.0, 0.0], [10.0, 9.0, 0.0], [9.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [1, 3, 2], [0, 2, 4], [5, 6, 7]], dtype=np.int64)
    return vertices, triangles


def test_nearest_triangle_one_ring_and_area_weighted_inward_normal():
    vertices, triangles = _nonuniform_mesh()
    query = TriangleMeshSurfaceQuery(vertices, triangles, inward_sign=1)
    hit = query.query(np.asarray([0.15, 0.15, 0.1]))

    assert hit.triangle_id == 0
    assert hit.one_ring_triangle_ids == (0, 1, 2)
    raw = np.cross(
        vertices[triangles[hit.one_ring_triangle_ids, 1]]
        - vertices[triangles[hit.one_ring_triangle_ids, 0]],
        vertices[triangles[hit.one_ring_triangle_ids, 2]]
        - vertices[triangles[hit.one_ring_triangle_ids, 0]],
    ).sum(axis=0)
    expected = raw / np.linalg.norm(raw)
    np.testing.assert_allclose(hit.normal_world, expected, atol=1.0e-12)
    assert len(hit.geometry_digest) == 64


def test_flat_and_stomach_adapters_are_identical_for_identical_geometry():
    vertices = np.asarray([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]])
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    flat = FlatSurfaceQuery(vertices=vertices, triangles=triangles)
    stomach = StomachSurfaceQuery(vertices=vertices, triangles=triangles, inward_sign=1)
    point = np.asarray([0.37, -0.42, 0.15])
    flat_hit, stomach_hit = flat.query(point), stomach.query(point)

    np.testing.assert_allclose(flat_hit.point_world, stomach_hit.point_world)
    np.testing.assert_allclose(flat_hit.normal_world, stomach_hit.normal_world)
    assert flat_hit.triangle_id == stomach_hit.triangle_id
    assert flat_hit.one_ring_triangle_ids == stomach_hit.one_ring_triangle_ids
    assert flat_hit.geometry_digest == stomach_hit.geometry_digest


def test_generated_flat_plane_has_world_positive_z_normal():
    hit = FlatSurfaceQuery.regular_plane(half_extent_m=0.5, cells_per_side=4).query(
        np.asarray([0.13, -0.21, 0.07])
    )
    np.testing.assert_allclose(hit.normal_world, [0.0, 0.0, 1.0], atol=1.0e-12)

