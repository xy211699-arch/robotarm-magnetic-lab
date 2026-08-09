#!/usr/bin/env python3
"""Validate P0 coverage geometry and the approved CUDA first-hit adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))

from robotarm_magnetic_lab.coverage.accumulator import CoverageAccumulator  # noqa: E402
from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh  # noqa: E402
from robotarm_magnetic_lab.coverage.visibility import (  # noqa: E402
    ScalarFirstHitRaycaster,
    WarpFirstHitRaycaster,
    visible_from_first_hits,
)


def _fixture():
    points = np.asarray(
        [
            [-0.01, -0.01, 0.02],
            [0.01, -0.01, 0.02],
            [0.0, 0.01, 0.02],
            [-0.01, -0.01, 0.03],
            [0.01, -0.01, 0.03],
            [0.0, 0.01, 0.03],
        ],
        dtype=np.float64,
    )
    source = MeshInput("/Fixture", points, np.asarray([3, 3]), np.arange(6), np.eye(4))
    return preprocess_reference_mesh([source], ["/Fixture"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("all", "gpu", "scalar"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kit_args", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    reference = _fixture()
    origin = np.zeros(3)
    targets = np.asarray([[0.0, 0.0, 0.03], [0.05, 0.0, 0.03]])
    scalar_distance, scalar_face = ScalarFirstHitRaycaster(reference).query(origin, targets)
    result = {
        "scalar_distance_m": scalar_distance.tolist(),
        "scalar_face_id": scalar_face.tolist(),
    }
    if args.check in ("all", "gpu"):
        gpu_distance, gpu_face = WarpFirstHitRaycaster(reference, args.device).query(origin, targets)
        np.testing.assert_allclose(gpu_distance, scalar_distance, atol=1.0e-5)
        np.testing.assert_array_equal(gpu_face, scalar_face)
        result.update({"gpu_distance_m": gpu_distance.tolist(), "gpu_face_id": gpu_face.tolist()})
    visible = visible_from_first_hits(
        np.asarray([3]), np.asarray([0.03]), np.asarray([0.02]), np.asarray([0]), reference.incident_triangles
    )
    assert not bool(visible[0])
    accumulator = CoverageAccumulator(len(reference.vertices_world))
    assert accumulator.update(1, [0]).updated
    assert not accumulator.update(1, [1]).updated
    print("COVERAGE_GEOMETRY_PASS " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
