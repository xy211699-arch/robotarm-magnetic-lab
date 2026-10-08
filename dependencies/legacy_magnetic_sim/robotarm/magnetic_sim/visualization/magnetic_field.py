"""USD line-vector visualization for magnetic flux density."""

import numpy as np
from pxr import Gf, UsdGeom


def create_or_update_vectors(stage, path, origins, vectors, arrow_length, arrow_width):
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    vectors = np.asarray(vectors, dtype=float).reshape(-1, 3)
    magnitudes = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(magnitudes[:, None], 1.0e-12)
    tips = origins + float(arrow_length) * directions

    points = []
    for origin, tip in zip(origins, tips):
        points.extend((Gf.Vec3f(*origin), Gf.Vec3f(*tip)))

    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curves.CreateCurveVertexCountsAttr([2] * len(origins))
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([float(arrow_width)] * len(points))
    curves.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.6, 1.0)])
    return magnitudes


def create_or_update_streamlines(stage, path, lines, line_width):
    """Create linear USD curves from a collection of 3-D point sequences."""
    valid_lines = [np.asarray(line, dtype=float).reshape(-1, 3) for line in lines]
    valid_lines = [line for line in valid_lines if len(line) >= 2]
    points = [Gf.Vec3f(*point) for line in valid_lines for point in line]

    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curves.CreateCurveVertexCountsAttr([len(line) for line in valid_lines])
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([float(line_width)] * len(points))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    curves.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.55, 1.0)])
    return len(valid_lines), len(points)


def draw_streamlines_debug(draw_interface, lines, width_px):
    """Draw streamlines as persistent viewport line segments."""
    starts = []
    ends = []
    for line in lines:
        points = np.asarray(line, dtype=float).reshape(-1, 3)
        starts.extend(map(tuple, points[:-1]))
        ends.extend(map(tuple, points[1:]))
    draw_interface.clear_lines()
    if starts:
        draw_interface.draw_lines(
            starts,
            ends,
            [(0.05, 0.65, 1.0, 1.0)] * len(starts),
            [int(width_px)] * len(starts),
        )
    return int(draw_interface.get_num_lines())
