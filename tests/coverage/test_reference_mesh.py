"""Reference-surface preprocessing tests."""

from __future__ import annotations

import numpy as np
import pytest

from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh


def _triangle(path: str, points: list[list[float]], transform=None) -> MeshInput:
    return MeshInput(
        prim_path=path,
        vertices=np.asarray(points, dtype=np.float64),
        face_vertex_counts=np.asarray([3], dtype=np.int64),
        face_vertex_indices=np.asarray([0, 1, 2], dtype=np.int64),
        world_transform=np.eye(4) if transform is None else np.asarray(transform, dtype=np.float64),
    )


def test_explicit_selection_transform_weld_and_incident_triangles():
    transform = np.eye(4)
    transform[3, :3] = [1.0, 2.0, 3.0]
    first = _triangle("/InnerA", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], transform)
    second = _triangle(
        "/InnerB",
        [[0.0 + 5.0e-7, 0, 0], [0, -1, 0], [-1, 0, 0]],
        transform,
    )
    excluded = _triangle("/CollisionProxy", [[10, 0, 0], [11, 0, 0], [10, 1, 0]])

    result = preprocess_reference_mesh(
        [excluded, second, first],
        selected_prim_paths=["/InnerA", "/InnerB"],
        weld_tolerance_m=1.0e-6,
    )

    assert result.selected_prim_paths == ("/InnerA", "/InnerB")
    assert result.vertices_world.shape == (5, 3)
    assert result.triangles.shape == (2, 3)
    shared = [index for index, incident in enumerate(result.incident_triangles) if incident == (0, 1)]
    assert len(shared) == 1
    assert np.allclose(result.vertices_world[shared[0]], [1.0, 2.0, 3.0], atol=1.0e-6)
    assert not np.any(result.vertices_world[:, 0] >= 10.0)


def test_result_is_deterministic_for_input_order():
    a = _triangle("/A", [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    b = _triangle("/B", [[0, 0, 0], [0, -1, 0], [-1, 0, 0]])
    one = preprocess_reference_mesh([a, b], ["/B", "/A"])
    two = preprocess_reference_mesh([b, a], ["/A", "/B"])
    np.testing.assert_array_equal(one.vertices_world, two.vertices_world)
    np.testing.assert_array_equal(one.triangles, two.triangles)
    assert one.incident_triangles == two.incident_triangles
    assert one.geometry_sha256 == two.geometry_sha256


@pytest.mark.parametrize(
    ("mesh", "message"),
    [
        (_triangle("/Bad", [[0, 0, 0], [float("nan"), 0, 0], [0, 1, 0]]), "nonfinite"),
        (
            MeshInput("/Bad", np.zeros((3, 3)), np.asarray([3]), np.asarray([0, 1, 9]), np.eye(4)),
            "index",
        ),
        (
            MeshInput("/Quad", np.zeros((4, 3)), np.asarray([4]), np.asarray([0, 1, 2, 3]), np.eye(4)),
            "triangular",
        ),
    ],
)
def test_invalid_geometry_is_rejected(mesh: MeshInput, message: str):
    with pytest.raises(ValueError, match=message):
        preprocess_reference_mesh([mesh], [mesh.prim_path])


def test_empty_or_unknown_selection_is_rejected():
    mesh = _triangle("/Inner", [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="empty"):
        preprocess_reference_mesh([mesh], [])
    with pytest.raises(ValueError, match="not found"):
        preprocess_reference_mesh([mesh], ["/Missing"])
