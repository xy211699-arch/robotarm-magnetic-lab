from __future__ import annotations

import json

import numpy as np
import pytest

from robotarm_magnetic_lab.coverage.area_weights import target_vertex_area_weights
from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh
from robotarm_magnetic_lab.coverage.unreachable_region import (
    UnreachableSeed,
    build_unreachable_mask,
    load_unreachable_mask,
    seeds_from_record,
    unreachable_region_record,
)


def _strip_reference():
    # Four 10 mm-wide squares triangulated into a connected strip.
    points = []
    for x in range(5):
        points.extend(((0.01 * x, 0.0, 0.0), (0.01 * x, 0.01, 0.0)))
    faces = []
    for x in range(4):
        a, b, c, d = 2 * x, 2 * x + 1, 2 * x + 2, 2 * x + 3
        faces.extend(((a, c, b), (c, d, b)))
    mesh = MeshInput(
        "/Stomach",
        np.asarray(points, dtype=np.float64),
        np.full(len(faces), 3, dtype=np.int64),
        np.asarray(faces, dtype=np.int64).reshape(-1),
        np.eye(4),
    )
    return preprocess_reference_mesh([mesh], ["/Stomach"])


def test_multi_seed_union_and_reachable_weights_recompute_shared_boundaries():
    reference = _strip_reference()
    seeds = (
        UnreachableSeed(0, reference.vertices_world[0], 0.01),
        UnreachableSeed(7, reference.vertices_world[-1], 0.01),
    )
    mask, per_seed = build_unreachable_mask(reference, seeds)
    assert len(per_seed) == 2
    assert len(mask.excluded_triangle_indices) < len(reference.triangles)
    assert set(mask.excluded_triangle_indices) == set(per_seed[0]) | set(per_seed[1])
    raw = target_vertex_area_weights(reference)
    reachable = target_vertex_area_weights(reference, mask.reachable_triangle_indices)
    assert np.isclose(raw.sum(), mask.excluded_area_m2 + reachable.sum())
    assert np.isclose(reachable.sum(), mask.reachable_area_m2)
    # Boundary vertices retain only contributions from reachable faces.
    assert np.any((reachable > 0.0) & (reachable < raw))


def test_frozen_record_round_trip_recomputes_and_rejects_tampering(tmp_path):
    reference = _strip_reference()
    seeds = (UnreachableSeed(0, np.asarray([0.0, 0.0, 0.0]), 0.01),)
    record = unreachable_region_record(
        reference=reference,
        seeds=seeds,
        reason="physical inlet region outside the reachable lumen",
        operator="unit-test",
    )
    path = tmp_path / "unreachable_region_v1.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    loaded, mask = load_unreachable_mask(path, reference)
    assert loaded["status"] == "frozen"
    assert mask.config_sha256 == record["config_sha256"]
    assert 0.0 < mask.excluded_area_fraction < 1.0
    restored = seeds_from_record(loaded)
    assert len(restored) == 1
    assert restored[0].triangle_index == seeds[0].triangle_index
    np.testing.assert_allclose(restored[0].point_world_m, seeds[0].point_world_m)
    assert restored[0].radius_m == seeds[0].radius_m

    record["reason"] = "controller happened to fail"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_unreachable_mask(path, reference)


def test_empty_and_complete_exclusions_are_rejected():
    reference = _strip_reference()
    with pytest.raises(ValueError, match="at least one"):
        build_unreachable_mask(reference, ())
    with pytest.raises(ValueError, match="complete stomach target"):
        build_unreachable_mask(
            reference,
            (UnreachableSeed(0, np.zeros(3), 0.08),),
        )
