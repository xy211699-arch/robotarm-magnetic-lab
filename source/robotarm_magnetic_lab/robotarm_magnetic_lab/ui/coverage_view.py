"""Isolated P0 coverage colors, Kit point-cloud view, and 2D export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


UNCOVERED_COLOR = np.asarray([220, 35, 35], dtype=np.uint8)
COVERED_COLOR = np.asarray([30, 190, 70], dtype=np.uint8)
CAPSULE_COLOR = np.asarray([40, 40, 40], dtype=np.uint8)
TRAJECTORY_COLOR = np.asarray([0, 0, 0], dtype=np.uint8)
AXIS_NAMES = ("X", "Y", "Z")


def coverage_colors(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    colors = np.tile(UNCOVERED_COLOR, (len(values), 1))
    colors[values] = COVERED_COLOR
    return colors


@dataclass(frozen=True)
class ProjectionConfig:
    width_px: int = 960
    height_px: int = 720
    horizontal_axis: int = 0
    vertical_axis: int = 1
    flip_vertical: bool = False
    padding_px: int = 40

    def __post_init__(self) -> None:
        if self.width_px <= 2 * self.padding_px or self.height_px <= 2 * self.padding_px:
            raise ValueError("projection image is too small for its padding")
        if self.horizontal_axis not in (0, 1, 2) or self.vertical_axis not in (0, 1, 2):
            raise ValueError("projection axes must be X, Y, or Z")
        if self.horizontal_axis == self.vertical_axis:
            raise ValueError("projection axes must be distinct")


def _projection_transform(vertices: np.ndarray, config: ProjectionConfig):
    selected = vertices[:, [config.horizontal_axis, config.vertical_axis]]
    lower = selected.min(axis=0)
    upper = selected.max(axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    usable = np.asarray(
        [config.width_px - 2 * config.padding_px, config.height_px - 2 * config.padding_px],
        dtype=np.float64,
    )

    def transform(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        projected = (values[:, [config.horizontal_axis, config.vertical_axis]] - lower) / span
        if not config.flip_vertical:
            projected[:, 1] = 1.0 - projected[:, 1]  # image Y always grows downward
        return projected * usable + config.padding_px

    return transform, lower, upper


def export_coverage_projection(
    output_path: Path,
    vertices_world: np.ndarray,
    mask: np.ndarray,
    capsule_position_world: np.ndarray,
    trajectory_world: np.ndarray,
    coverage_fraction: float,
    elapsed_time_s: float,
    config: ProjectionConfig = ProjectionConfig(),
) -> dict[str, Any]:
    """Export the deterministic red/green coverage projection and metadata."""
    from PIL import Image, ImageDraw, ImageFont

    vertices = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if len(vertices) != len(values):
        raise ValueError("coverage mask length must equal vertex count")
    trajectory = np.asarray(trajectory_world, dtype=np.float64).reshape(-1, 3)
    capsule = np.asarray(capsule_position_world, dtype=np.float64).reshape(1, 3)
    transform, lower, upper = _projection_transform(vertices, config)
    vertex_pixels = np.rint(transform(vertices)).astype(int)
    image = Image.new("RGB", (config.width_px, config.height_px), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    colors = coverage_colors(values)
    for pixel, color in zip(vertex_pixels, colors, strict=True):
        x, y = int(pixel[0]), int(pixel[1])
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=tuple(int(v) for v in color))
    if len(trajectory) >= 2:
        trajectory_pixels = [tuple(int(v) for v in point) for point in np.rint(transform(trajectory))]
        draw.line(trajectory_pixels, fill=tuple(int(v) for v in TRAJECTORY_COLOR), width=2)
    capsule_pixel = np.rint(transform(capsule))[0].astype(int)
    draw.ellipse(
        (
            int(capsule_pixel[0] - 4),
            int(capsule_pixel[1] - 4),
            int(capsule_pixel[0] + 4),
            int(capsule_pixel[1] + 4),
        ),
        fill=tuple(int(v) for v in CAPSULE_COLOR),
    )
    font = ImageFont.load_default()
    coverage_text = f"{float(coverage_fraction) * 100.0:.3f}%"
    draw.rectangle((8, 8, 265, 55), fill=(245, 245, 245))
    draw.text((12, 12), f"Coverage: {coverage_text}", fill=(0, 0, 0), font=font)
    draw.text((12, 30), f"Elapsed: {float(elapsed_time_s):.3f} s", fill=(0, 0, 0), font=font)
    draw.rectangle((config.width_px - 180, 12, config.width_px - 168, 24), fill=tuple(UNCOVERED_COLOR))
    draw.text((config.width_px - 162, 12), "uncovered", fill=(0, 0, 0), font=font)
    draw.rectangle((config.width_px - 180, 30, config.width_px - 168, 42), fill=tuple(COVERED_COLOR))
    draw.text((config.width_px - 162, 30), "covered", fill=(0, 0, 0), font=font)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return {
        "image_size_px": [config.width_px, config.height_px],
        "projection_axes": {
            "horizontal": AXIS_NAMES[config.horizontal_axis],
            "vertical": AXIS_NAMES[config.vertical_axis],
            "flip_vertical": config.flip_vertical,
        },
        "world_bounds_on_projection_axes_m": [lower.tolist(), upper.tolist()],
        "padding_px": config.padding_px,
        "coverage_percent_text": coverage_text,
    }


class KitCoveragePointCloudView:
    """Guide-purpose USD debug geometry shown in a dedicated Kit viewport."""

    def __init__(self, vertices_world: np.ndarray, root_path: str = "/World/P0CoverageDebug") -> None:
        import omni.usd
        from omni.kit.viewport.utility import create_viewport_window
        from pxr import Gf, Sdf, UsdGeom, Vt

        self._UsdGeom = UsdGeom
        self._Vt = Vt
        self._Gf = Gf
        self._root_path = root_path
        self._vertices = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, root_path)
        root.GetPrim().SetMetadata("comment", "TASK-001 guide-only coverage debug; excluded from physics and rays")
        self._points = UsdGeom.Points.Define(stage, root_path + "/Surface")
        self._points.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        self._points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self._vertices.astype(np.float32)))
        self._points.GetWidthsAttr().Set([0.0007] * len(self._vertices))
        self._marker = UsdGeom.Points.Define(stage, root_path + "/Capsule")
        self._marker.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        self._marker.GetWidthsAttr().Set([0.006])
        self._trajectory = UsdGeom.BasisCurves.Define(stage, root_path + "/Trajectory")
        self._trajectory.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        self._trajectory.GetTypeAttr().Set(UsdGeom.Tokens.linear)
        self._trajectory.GetBasisAttr().Set(UsdGeom.Tokens.bezier)
        self._trajectory.GetWidthsAttr().Set([0.001])
        camera_path = root_path + "/Camera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        center = self._vertices.mean(axis=0)
        span = float(np.linalg.norm(self._vertices.max(axis=0) - self._vertices.min(axis=0)))
        eye = center + np.asarray([0.0, -1.4 * span, 0.8 * span])
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0, 0, 1))
        camera.GetPrim().GetAttribute("xformOpOrder").Clear()
        camera.AddTransformOp().Set(view.GetInverse())
        self._window = create_viewport_window(
            name="P0 Stomach Coverage",
            width=960,
            height=720,
            camera_path=Sdf.Path(camera_path),
        )
        self.update(np.zeros(len(self._vertices), dtype=bool), center, np.empty((0, 3)))

    def update(self, mask: np.ndarray, capsule_position_world: np.ndarray, trajectory_world: np.ndarray) -> None:
        colors = coverage_colors(mask).astype(np.float32) / 255.0
        self._points.GetDisplayColorAttr().Set(self._Vt.Vec3fArray.FromNumpy(colors))
        marker = np.asarray(capsule_position_world, dtype=np.float32).reshape(1, 3)
        self._marker.GetPointsAttr().Set(self._Vt.Vec3fArray.FromNumpy(marker))
        self._marker.GetDisplayColorAttr().Set(
            self._Vt.Vec3fArray.FromNumpy((CAPSULE_COLOR[None, :] / 255.0).astype(np.float32))
        )
        trajectory = np.asarray(trajectory_world, dtype=np.float32).reshape(-1, 3)
        self._trajectory.GetPointsAttr().Set(self._Vt.Vec3fArray.FromNumpy(trajectory))
        self._trajectory.GetCurveVertexCountsAttr().Set([len(trajectory)] if len(trajectory) else [])
        self._trajectory.GetDisplayColorAttr().Set(
            self._Vt.Vec3fArray.FromNumpy((TRAJECTORY_COLOR[None, :] / 255.0).astype(np.float32))
        )

    def close(self) -> None:
        if self._window is not None:
            self._window.destroy()
            self._window = None
