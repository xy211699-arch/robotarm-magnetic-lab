"""Pure integration invariants for the P0 evaluator runtime."""

from __future__ import annotations

import numpy as np

from robotarm_magnetic_lab.coverage.runtime import (
    RecordedFrameClock,
    assert_coverage_consistency,
)


def test_recorded_frame_clock_emits_once_per_unique_timestamp():
    clock = RecordedFrameClock(update_period_s=1.0)
    assert clock.observe(0.0) == 0
    assert clock.observe(0.0) is None
    assert clock.observe(1.0) == 1
    assert clock.observe(1.0000001) is None
    assert clock.observe(2.0) == 2
    clock.reset()
    assert clock.observe(0.0) == 0


def test_recorded_frame_clock_rejects_an_unannounced_sensor_reset():
    clock = RecordedFrameClock(update_period_s=1.0)
    assert clock.observe(3.0) == 3
    with np.testing.assert_raises(ValueError):
        clock.observe(0.0)


def test_mask_color_record_export_consistency():
    mask = np.asarray([False, True, True, False, True])
    summary = assert_coverage_consistency(
        mask=mask,
        record={
            "cumulative_count": 3,
            "vertex_count": 5,
            "coverage_fraction": 0.6,
        },
        export_metadata={"coverage_percent_text": "60.000%"},
    )
    assert summary == {"vertex_count": 5, "covered_count": 3, "coverage_fraction": 0.6}


def test_consistency_rejects_stale_record():
    with np.testing.assert_raises(ValueError):
        assert_coverage_consistency(
            np.asarray([True, False]),
            {"cumulative_count": 0, "vertex_count": 2, "coverage_fraction": 0.0},
            {"coverage_percent_text": "0.000%"},
        )
