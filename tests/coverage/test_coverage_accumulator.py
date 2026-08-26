"""Monotonic recorded-frame coverage accumulation tests."""

from __future__ import annotations

import numpy as np

from robotarm_magnetic_lab.coverage.accumulator import CoverageAccumulator
from robotarm_magnetic_lab.coverage.area_weights import target_vertex_area_weights
from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh


def test_unique_frame_updates_once_and_mask_is_monotonic():
    accumulator = CoverageAccumulator(vertex_count=5)
    first = accumulator.update(frame_id=10, visible_vertex_indices=[0, 2])
    duplicate = accumulator.update(frame_id=10, visible_vertex_indices=[1, 3])
    second = accumulator.update(frame_id=11, visible_vertex_indices=[2, 4])

    assert first.updated and first.newly_covered_count == 2
    assert not duplicate.updated and duplicate.newly_covered_count == 0
    assert second.updated and second.newly_covered_count == 1
    assert accumulator.mask.tolist() == [True, False, True, False, True]
    assert second.cumulative_count == 3
    assert second.coverage_fraction == 0.6
    assert accumulator.recorded_frame_count == 2


def test_reset_clears_mask_frame_ids_and_counters():
    accumulator = CoverageAccumulator(vertex_count=3)
    accumulator.update("frame-a", [0, 1])
    accumulator.reset()
    assert accumulator.mask.tolist() == [False, False, False]
    assert accumulator.recorded_frame_count == 0
    repeated_after_reset = accumulator.update("frame-a", [2])
    assert repeated_after_reset.updated
    assert accumulator.mask.tolist() == [False, False, True]


def test_nonuniform_area_weights_control_fraction_and_conserve_triangle_area():
    # Face 0 has area 0.5 m2 and face 1 has area 2.0 m2.
    mesh = MeshInput(
        "/Target",
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
            ]
        ),
        np.asarray([3, 3]),
        np.asarray([0, 1, 2, 3, 4, 5]),
        np.eye(4),
    )
    reference = preprocess_reference_mesh([mesh], ["/Target"])
    weights = target_vertex_area_weights(reference)
    assert np.isclose(weights.sum(), 2.5)
    accumulator = CoverageAccumulator(len(weights), vertex_weights=weights)
    update = accumulator.update(1, reference.triangles[0])
    assert np.isclose(update.visible_area_m2, 0.5)
    assert np.isclose(update.cumulative_area_m2, 0.5)
    assert np.isclose(update.total_area_m2, 2.5)
    assert np.isclose(update.coverage_fraction, 0.2)
    assert not np.isclose(update.coverage_fraction, update.cumulative_count / len(weights))
