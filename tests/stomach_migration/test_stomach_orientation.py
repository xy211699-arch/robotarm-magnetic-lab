import numpy as np

from robotarm_magnetic_lab.geometry.stomach_orientation import (
    extract_boundary_components,
    quaternion_xyzw_to_matrix,
    select_tube_openings,
    solve_horizontal_axes,
)


def _disk(center, normal, radius, segments, vertex_offset):
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(normal @ helper)) > 0.8:
        helper = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, helper)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    ring = np.asarray(
        [
            np.asarray(center)
            + radius * (np.cos(theta) * tangent + np.sin(theta) * bitangent)
            for theta in np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        ]
    )
    vertices = np.vstack([np.asarray(center), ring])
    triangles = np.asarray(
        [
            (
                vertex_offset,
                vertex_offset + 1 + index,
                vertex_offset + 1 + ((index + 1) % segments),
            )
            for index in range(segments)
        ],
        dtype=np.int64,
    )
    return vertices, triangles


def test_selects_two_dominant_planar_openings_and_makes_both_axes_horizontal():
    parts = [
        _disk((2.0, 0.0, 0.0), (1.0, 0.2, 0.4), 0.2, 12, 0),
        _disk((0.0, 2.0, 0.0), (-0.3, 1.0, 0.7), 0.2, 12, 13),
        _disk((0.0, 0.0, 0.2), (0.0, 0.0, 1.0), 0.03, 5, 26),
    ]
    vertices = np.vstack([part[0] for part in parts])
    triangles = np.vstack([part[1] for part in parts])

    components = extract_boundary_components(vertices, triangles)
    openings = select_tube_openings(vertices, components, mesh_centroid=np.zeros(3))

    assert [opening.vertex_count for opening in openings] == [12, 12]
    quaternion = solve_horizontal_axes(
        openings[0].axis,
        openings[1].axis,
        body_center=np.array([-1.0, -1.0, -0.5]),
    )
    rotation = quaternion_xyzw_to_matrix(quaternion)
    for opening in openings:
        transformed = rotation @ opening.axis
        elevation_deg = np.degrees(np.arcsin(abs(transformed[2])))
        assert elevation_deg <= 1.0e-8
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-12)


def test_nearly_parallel_axes_have_a_finite_deterministic_solution():
    axis_a = np.array([1.0, 0.0, 0.0])
    axis_b = np.array([1.0, 1.0e-12, 0.0])
    body_center = np.array([0.0, 0.2, -1.0])

    first = solve_horizontal_axes(axis_a, axis_b, body_center)
    second = solve_horizontal_axes(axis_a, axis_b, body_center)

    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, atol=0.0)
    rotation = quaternion_xyzw_to_matrix(first)
    assert abs(float((rotation @ axis_a)[2])) <= 1.0e-12
    assert abs(float((rotation @ axis_b)[2])) <= 1.0e-10
    assert float((rotation @ body_center)[2]) < 0.0
