"""Monotonic recorded-frame coverage accumulation tests."""

from __future__ import annotations

from robotarm_magnetic_lab.coverage.accumulator import CoverageAccumulator


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
