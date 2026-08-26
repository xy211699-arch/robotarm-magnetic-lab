"""Runtime helpers for the privileged P0 coverage evaluator.

The pure pieces in this module define the recorded-frame and consistency
boundaries.  Simulator-specific orchestration remains in the launcher so none
of these values can enter policy observations or the atomic executor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from robotarm_magnetic_lab.ui.coverage_view import COVERED_COLOR, coverage_colors


@dataclass
class RecordedFrameClock:
    """Turn sensor timestamps into unique stable frame IDs."""

    update_period_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.update_period_s) or self.update_period_s <= 0.0:
            raise ValueError("update_period_s must be finite and positive")
        self._last_id: int | None = None

    def observe(self, timestamp_s: float) -> int | None:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("sensor timestamp must be finite and nonnegative")
        frame_id = int(round(timestamp / self.update_period_s))
        if frame_id == self._last_id:
            return None
        if self._last_id is not None and frame_id < self._last_id:
            raise ValueError("sensor frame IDs must be monotonic until reset")
        self._last_id = frame_id
        return frame_id

    def reset(self) -> None:
        self._last_id = None


def assert_coverage_consistency(
    mask: np.ndarray,
    record: Mapping[str, Any],
    export_metadata: Mapping[str, Any],
    vertex_weights: np.ndarray | None = None,
) -> dict[str, int | float]:
    """Require exact mask/3D-color/record/2D-export agreement."""
    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    count = int(values.sum())
    total = int(len(values))
    if total <= 0:
        raise ValueError("coverage mask cannot be empty")
    weighted = vertex_weights is not None
    if not weighted:
        weights = np.ones(total, dtype=np.float64)
    else:
        weights = np.asarray(vertex_weights, dtype=np.float64).reshape(-1)
    if len(weights) != total or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("coverage weights are invalid")
    total_area = float(weights.sum())
    if total_area <= 0.0:
        raise ValueError("coverage weights have zero total area")
    covered_area = float(weights[values].sum())
    fraction = covered_area / total_area
    colors = coverage_colors(values)
    color_count = int(np.count_nonzero(np.all(colors == COVERED_COLOR, axis=1)))
    if color_count != count:
        raise ValueError("3D coverage colors disagree with mask")
    if int(record["vertex_count"]) != total or int(record["cumulative_count"]) != count:
        raise ValueError("frame record disagrees with mask")
    if not math.isclose(float(record["coverage_fraction"]), fraction, abs_tol=1.0e-12):
        raise ValueError("frame coverage fraction disagrees with mask")
    if "cumulative_area_m2" in record and not math.isclose(
        float(record["cumulative_area_m2"]), covered_area, abs_tol=1.0e-12
    ):
        raise ValueError("frame cumulative area disagrees with mask")
    expected_text = f"{fraction * 100.0:.3f}%"
    if str(export_metadata["coverage_percent_text"]) != expected_text:
        raise ValueError("2D export coverage text disagrees with mask")
    result = {
        "vertex_count": total,
        "covered_count": count,
        "coverage_fraction": fraction,
    }
    if weighted:
        result.update({"covered_area_m2": covered_area, "total_area_m2": total_area})
    return result
