from __future__ import annotations

import math

import numpy as np
import torch

from robotarm_magnetic_lab.coverage.batched_visibility import (
    batched_candidate_mask,
    build_incident_face_table,
    visible_from_batched_first_hits,
)
from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh
from robotarm_magnetic_lab.coverage.visibility import (
    camera_facing_first_hits,
    candidate_vertices,
    visible_from_first_hits,
)


def _two_triangle_reference():
    vertices = np.asarray(
        [
            [-0.01, -0.01, 0.03],
            [0.01, -0.01, 0.03],
            [0.01, 0.01, 0.03],
            [-0.01, 0.01, 0.03],
        ],
        dtype=np.float64,
    )
    return preprocess_reference_mesh(
        [
            MeshInput(
                "/Inner",
                vertices,
                np.asarray([3, 3]),
                np.asarray([0, 1, 2, 0, 2, 3]),
                np.eye(4),
            )
        ],
        ["/Inner"],
    )


def test_batched_candidates_equal_scalar_for_each_camera():
    angle = math.radians(60.0)
    vertices_np = np.asarray(
        [
            [0.0, 0.0, 0.07],
            [0.07 * math.sin(angle), 0.0, 0.07 * math.cos(angle)],
            [0.0, 0.0, 0.070001],
            [0.01, 0.0, 0.03],
        ],
        dtype=np.float64,
    )
    vertices = torch.tensor(vertices_np, dtype=torch.float64)
    centers = torch.tensor([[0, 0, 0], [0.01, 0, 0]], dtype=torch.float64)
    axes = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float64)
    mask, distances = batched_candidate_mask(vertices, centers, axes)
    for row in range(2):
        scalar_ids, scalar_distances = candidate_vertices(
            vertices_np, centers[row].numpy(), axes[row].numpy()
        )
        assert torch.nonzero(mask[row], as_tuple=False).reshape(-1).tolist() == scalar_ids.tolist()
        np.testing.assert_allclose(
            distances[row, scalar_ids].numpy(), scalar_distances, atol=1.0e-12
        )


def test_incident_distance_and_lumen_normal_gates_match_scalar():
    reference = _two_triangle_reference()
    table = build_incident_face_table(reference.incident_triangles, device="cpu")
    candidates = torch.ones((2, 4), dtype=torch.bool)
    target_distances = torch.full((2, 4), 0.03, dtype=torch.float64)
    hit_distances = torch.tensor(
        [[0.0301, 0.0301001, 0.03, float("inf")], [0.03, 0.03, 0.03, 0.03]],
        dtype=torch.float64,
    )
    hit_faces = torch.tensor([[0, 0, 1, -1], [1, 1, 0, 0]], dtype=torch.int64)
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]], dtype=torch.float64)
    targets = torch.tensor(reference.vertices_world, dtype=torch.float64)
    result = visible_from_batched_first_hits(
        reference,
        centers,
        targets,
        candidates,
        target_distances,
        hit_distances,
        hit_faces,
        table,
        normal_sign=-1,
    )
    for row in range(2):
        ids = np.arange(4, dtype=np.int64)
        incident_ok = visible_from_first_hits(
            ids,
            target_distances[row].numpy(),
            hit_distances[row].numpy(),
            hit_faces[row].numpy(),
            reference.incident_triangles,
        )
        facing = camera_facing_first_hits(
            centers[row].numpy(),
            reference.vertices_world,
            hit_faces[row].numpy(),
            reference,
            normal_sign=-1,
        )
        np.testing.assert_array_equal(result[row].numpy(), incident_ok & facing)


def test_incident_table_has_minus_one_padding():
    table = build_incident_face_table(((0, 2), (1,), ()), device="cpu")
    assert table.dtype == torch.int64
    assert table.tolist() == [[0, 2], [1, -1], [-1, -1]]
