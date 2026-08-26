"""Coverage color and deterministic projection tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robotarm_magnetic_lab.ui.coverage_view import (
    CAPSULE_COLOR,
    COVERED_COLOR,
    CURRENT_VISIBLE_COLOR,
    TRAJECTORY_COLOR,
    UNCOVERED_COLOR,
    ProjectionConfig,
    coverage_colors,
    export_coverage_projection,
)


def test_exact_color_mapping_and_counts():
    mask = np.asarray([False, True, False, True])
    colors = coverage_colors(mask)
    np.testing.assert_array_equal(colors[~mask], np.tile(UNCOVERED_COLOR, (2, 1)))
    np.testing.assert_array_equal(colors[mask], np.tile(COVERED_COLOR, (2, 1)))
    assert CAPSULE_COLOR.tolist() == [40, 40, 40]
    assert TRAJECTORY_COLOR.tolist() == [0, 0, 0]


def test_current_visible_vertices_override_cumulative_history_color():
    cumulative = np.asarray([False, True, True, False])
    current = np.asarray([False, False, True, True])
    colors = coverage_colors(cumulative, current)
    np.testing.assert_array_equal(colors[0], UNCOVERED_COLOR)
    np.testing.assert_array_equal(colors[1], COVERED_COLOR)
    np.testing.assert_array_equal(colors[2], CURRENT_VISIBLE_COLOR)
    np.testing.assert_array_equal(colors[3], CURRENT_VISIBLE_COLOR)


def test_projection_is_deterministic_and_records_orientation(tmp_path: Path):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0]], dtype=float)
    mask = np.asarray([False, True, False, True])
    trajectory = np.asarray([[0.1, 0.2, 0], [0.8, 1.7, 0]])
    cfg = ProjectionConfig(width_px=320, height_px=240, horizontal_axis=0, vertical_axis=1, flip_vertical=True)
    first = export_coverage_projection(
        tmp_path / "first.png",
        vertices,
        mask,
        capsule_position_world=np.asarray([0.8, 1.7, 0]),
        trajectory_world=trajectory,
        coverage_fraction=0.5,
        elapsed_time_s=12.25,
        config=cfg,
    )
    second = export_coverage_projection(
        tmp_path / "second.png",
        vertices,
        mask,
        capsule_position_world=np.asarray([0.8, 1.7, 0]),
        trajectory_world=trajectory,
        coverage_fraction=0.5,
        elapsed_time_s=12.25,
        config=cfg,
    )
    assert (tmp_path / "first.png").read_bytes() == (tmp_path / "second.png").read_bytes()
    assert first == second
    assert first["image_size_px"] == [320, 240]
    assert first["projection_axes"] == {"horizontal": "X", "vertical": "Y", "flip_vertical": True}
    assert first["coverage_percent_text"] == "50.000%"
    json.dumps(first)
