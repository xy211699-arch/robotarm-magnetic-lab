# Local small-view collection: generated from the saved V8 engine.
# v8: pragmatic image framing and image-count based lesion collection.
# RUN INSIDE Isaac Sim with runpy. Existing orbit_v6/targets.json is reused.
# Use "5 Collect ULCERATED: 50 images" or the corresponding RAISED button.
# Distance is the ACTUAL optical-axis first visible stomach hit, in mm.
# Valid range remains 30-60 mm; no snapping to planned range values.
# Marked center must be comfortably inside image; whole ROI need not fit a circle.
# Source region stays in the raw frame with a small crop margin; peripheral ROI
# wall landmarks need not all remain unobstructed from every side view.
# Actual marked target must be visible. Geometry, 5 mm clearance, frozen-render
# and center depth/instance guards are retained. Duplicates are not counted.
# Per-class collection is bounded: actual shortfalls are recorded, never padded.
# These are seed-supervised candidates pending class/visibility review, not
# a declaration that the final 200-image dataset or train/test splits are done.
# v7 full standalone file. Run INSIDE Isaac Sim with runpy.
# Existing orbit_v6/targets.json calibration is reused, not overwritten on launch.
# v7 replaces sphere-only framing with projection of sampled source surfaces.
# Each of 12 azimuths tries several ranges/cone angles/offsets, logs every attempt.
# 30-60 mm optical first-hit range, >=5 mm clearance and occlusion checks remain.
# New geometry functions have CPU checks; real GUI/GPU pilot still required.
# RUN: import runpy; orbit = runpy.run_path("/home/multirobo/gastric-single-frame-20261117/gastric_orbit_v7.py")
# Gastric v6: region calibration and automatic ORBIT PILOT, inside Isaac Sim.
# The retained v5 engine was validated by the user. New UI/orbit code has CPU
# checks only; validate a small pilot in the real GUI before scaling up.
# RUN: import runpy; orbit = runpy.run_path("/home/multirobo/gastric-single-frame-20261117/gastric_orbit_v6.py")
# This opens a saved-image calibration panel; it does NOT start bulk capture.
# 1. Load latest valid v4 sample. Set X/Y to lesion center and adjust R.
# 2. Recenter if the lesion is cropped; then mark the COMPLETE visible lesion.
# 3. Save Raised or Ulcerated. Verify the saved three-dimensional region.
# 4. Test a 12-view orbit. All class labels remain pending human review.
# Raw exported image.png never contains the calibration circle or UI text.
# Snapshots still reference external materials/textures: keep source assets.
# This pilot intentionally does not generate final training splits or claim
# that a circular ROI provides an exact segmentation of the lesion.
# v5: isolate the measurement camera from interactive viewport initialization.
# The preview shows the exported PNG only AFTER validation and file export.
# The existing output_v4 directory is retained to avoid moving output again.
# Run the saved file via runpy inside Isaac Sim. No manual config edits needed.
# Gastric capture v4: run this file with runpy INSIDE Isaac Sim Script Editor.
# Automatic unique camera paths and reversible session masks for old helpers.
# Script-source path is inferred from __file__ when run via runpy.
# Output: /home/multirobo/gastric-single-frame-20261117/output_v4/<capture-id>/
# Static single-frame protocol only; still requires real GUI validation.
# Gastric single-frame capture v3 — run INSIDE Isaac Sim Script Editor.
# Local test scope: CPU math/parser/guard checks, NOT a real GPU validation.
# v3 intentionally supports only frozen-state capture. It uses
# step_async(wait_for_render=True, delta_time=0.0), keeps native frame ID null,
# and verifies pose/scene/time plus center-depth/object identity.
# Save this exact file at CONFIG['script_source_path'] before running it.
# Original source and v2 diagnostics are preserved in the prior package.
import asyncio
import builtins
import hashlib
import itertools
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import traceback
import uuid

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdShade


CONFIG = {
    # This file must be saved here before pasting its complete contents in Kit.
    "script_source_path": "/home/multirobo/gastric-single-frame-20261117/capture_one_frame_v5.py",
    "camera_world_matrix_replay": None,
    "replay_meters_per_unit": None,
    "replay_geometry_sha256": None,
    "save_scene_snapshot": True,
    "output_root": "/home/multirobo/gastric-single-frame-20261117/output_v4",
    # 自动选择不唯一时，在这里填写实际可见胃壁 Mesh 的完整 Prim 路径。
    "stomach_mesh_path": None,
    "helper_root": "/GastricSingleFrameCaptureV4",
    # 奇数尺寸，使一个像素的中心对应光轴。
    "resolution": [801, 801],
    "horizontal_fov_deg": 90.0,
    "horizontal_aperture_mm": 20.0,
    "near_m": 0.0002,
    "far_m": 0.30,
    "minimum_clearance_m": 0.005,
    "preferred_distance_m": 0.030,
    "led_intensity": 4.0,
    "led_radius_m": 0.001,
    "led_offset_camera_m": [0.0, 0.0, -0.001],
    "led_temperature_K": 4800.0,
    "warmup_steps": 3,
    "rt_subframes": 8,
    "render_timeout_s": 60.0,
    "depth_tolerance_m": 0.0005,
}
class _SkipExportPreview(Exception):
    pass


REGISTRY = "_gastric_single_frame_capture_v1"


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-14:
        raise ValueError("无效方向向量")
    return vector / length


def json_value(value):
    if isinstance(value, np.void) and value.dtype.names:
        return {name: json_value(value[name]) for name in value.dtype.names}
    if isinstance(value, np.ndarray) and value.dtype.names:
        return [json_value(v) for v in value.reshape(-1)]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


class Triangles:
    """所有坐标均为米；射线求交不进行背面剔除。"""

    def __init__(self, vertices, faces):
        self.tri = np.asarray(vertices, dtype=np.float64)[faces]
        self.a = self.tri[:, 0]
        self.e1 = self.tri[:, 1] - self.a
        self.e2 = self.tri[:, 2] - self.a

    def hits(self, origin, direction):
        origin = np.asarray(origin, dtype=np.float64)
        direction = unit(direction)
        p = np.cross(np.broadcast_to(direction, self.e2.shape), self.e2)
        determinant = np.einsum("ij,ij->i", self.e1, p)
        scale = np.linalg.norm(self.e1, axis=1) * np.linalg.norm(
            self.e2, axis=1
        )
        usable = np.abs(determinant) > 1e-12 * np.maximum(scale, 1e-20)
        inverse = np.zeros_like(determinant)
        inverse[usable] = 1.0 / determinant[usable]

        offset = origin - self.a
        u = np.einsum("ij,ij->i", offset, p) * inverse
        q = np.cross(offset, self.e1)
        v = (q @ direction) * inverse
        distance = np.einsum("ij,ij->i", self.e2, q) * inverse
        accepted = (
            usable
            & (u >= -1e-9)
            & (v >= -1e-9)
            & (u + v <= 1.0 + 1e-9)
            & (distance > 1e-8)
        )
        values = np.sort(distance[accepted])
        if not len(values):
            return values
        # 合并三角形公共边产生的重复交点，避免破坏奇偶判定。
        return values[np.r_[True, np.diff(values) > 1e-7]]

    def clearance(self, point):
        """点到三角形表面的最小距离，非到顶点的近似距离。"""
        point = np.asarray(point)
        a, b, c = self.tri[:, 0], self.tri[:, 1], self.tri[:, 2]
        ab, ac, ap = b - a, c - a, point - a
        aa = np.einsum("ij,ij->i", ab, ab)
        cc = np.einsum("ij,ij->i", ac, ac)
        cross = np.einsum("ij,ij->i", ab, ac)
        pa = np.einsum("ij,ij->i", ap, ab)
        pc = np.einsum("ij,ij->i", ap, ac)
        denominator = aa * cc - cross * cross
        safe = np.where(denominator > 1e-30, denominator, 1.0)
        u = (cc * pa - cross * pc) / safe
        v = (aa * pc - cross * pa) / safe
        inside = (
            (denominator > 1e-30) & (u >= 0) & (v >= 0) & (u + v <= 1)
        )
        delta = ap - u[:, None] * ab - v[:, None] * ac
        best = np.where(
            inside, np.einsum("ij,ij->i", delta, delta), np.inf
        )
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            squared = np.einsum("ij,ij->i", edge, edge)
            fraction = np.clip(
                np.einsum("ij,ij->i", point - start, edge)
                / np.maximum(squared, 1e-30),
                0,
                1,
            )
            delta = point - start - fraction[:, None] * edge
            best = np.minimum(
                best, np.einsum("ij,ij->i", delta, delta)
            )
        return math.sqrt(float(best.min()))

    def winding(self, point):
        angle = 0.0
        for begin in range(0, len(self.tri), 16384):
            vectors = self.tri[begin:begin + 16384] - point
            a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
            la = np.linalg.norm(a, axis=1)
            lb = np.linalg.norm(b, axis=1)
            lc = np.linalg.norm(c, axis=1)
            numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
            denominator = (
                la * lb * lc
                + np.einsum("ij,ij->i", a, b) * lc
                + np.einsum("ij,ij->i", b, c) * la
                + np.einsum("ij,ij->i", c, a) * lb
            )
            angle += float((2 * np.arctan2(numerator, denominator)).sum())
        return angle / (4 * math.pi)


def math_check():
    vertices = np.array(list(itertools.product((-0.01, 0.01), repeat=3)))
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    query = Triangles(vertices, faces)
    assert np.allclose(query.hits([0, 0, 0], [0, 0, -1]), [0.01])
    assert np.allclose(
        query.hits([0, 0, 0.03], [0, 0, -1]), [0.02, 0.04]
    )
    assert abs(query.clearance([0.006, 0, 0]) - 0.004) < 1e-10
    assert abs(abs(query.winding([0, 0, 0])) - 1) < 1e-8
    assert abs(query.winding([0.1, 0, 0])) < 1e-8
    mirrored = Triangles(vertices * [-1, 1, 1], faces)
    assert np.allclose(mirrored.hits([0, 0, 0], [0, 0, -1]), [0.01])
    return "analytic_cube_range_clearance_winding_and_mirror_passed"


def visible(prim, time_code):
    imageable = UsdGeom.Imageable(prim)
    return (
        imageable
        and imageable.ComputeVisibility(time_code) != UsdGeom.Tokens.invisible
        and imageable.ComputePurpose()
        not in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy)
    )


def pick_mesh(stage, time_code, config):
    meshes = [
        p for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        if p.IsA(UsdGeom.Mesh) and visible(p, time_code)
    ]
    explicit = config["stomach_mesh_path"]
    if explicit:
        selected = [p for p in meshes if str(p.GetPath()) == explicit]
    else:
        known = {
            "/NewStomach/geometry/mesh",
            "/new_stomachfinishied/geometry/mesh",
            "/World/envs/env_0/Stomach/geometry/mesh",
        }
        selected = [p for p in meshes if str(p.GetPath()) in known]
        if not selected:
            selected = [
                p for p in meshes
                if "stomach" in str(p.GetPath()).lower()
                and "collision" not in str(p.GetPath()).lower()
            ]
    if len(selected) != 1:
        raise RuntimeError(
            "无法唯一选择胃壁 Mesh。请设置 CONFIG['stomach_mesh_path']。"
            "可见 Mesh：" + str([str(p.GetPath()) for p in meshes])
        )
    return selected[0]


def mesh_snapshot(prim, time_code, meters_per_unit):
    mesh = UsdGeom.Mesh(prim)
    if mesh.GetSubdivisionSchemeAttr().Get() != UsdGeom.Tokens.none:
        raise RuntimeError(
            str(prim.GetPath()) + " 使用细分曲面；不能用控制网格冒充渲染表面"
        )
    if any("Skel" in str(s) for s in prim.GetAppliedSchemas()):
        raise RuntimeError("本脚本不支持蒙皮变形网格")
    points = np.array(mesh.GetPointsAttr().Get(time_code), dtype=np.float64)
    counts = np.array(
        mesh.GetFaceVertexCountsAttr().Get(time_code), dtype=np.int64
    )
    indices = np.array(
        mesh.GetFaceVertexIndicesAttr().Get(time_code), dtype=np.int64
    )
    if (
        points.ndim != 2 or points.shape[1] != 3
        or not np.isfinite(points).all() or not len(counts)
    ):
        raise RuntimeError("网格点或拓扑无效")
    if not np.all(counts == 3) or len(indices) != 3 * len(counts):
        raise RuntimeError("本脚本要求实际三角网格，不会修改源拓扑")
    faces = indices.reshape(-1, 3)
    if faces.min() < 0 or faces.max() >= len(points):
        raise RuntimeError("三角形索引越界")
    holes = mesh.GetHoleIndicesAttr().Get(time_code)
    if holes is not None and len(holes):
        faces = faces[
            ~np.isin(np.arange(len(faces)), np.asarray(holes))
        ]

    matrix = np.array(
        UsdGeom.XformCache(time_code).GetLocalToWorldTransform(prim),
        dtype=np.float64,
    )
    # USD/Gf 使用行向量约定；先应用完整世界变换，再换算成米。
    world = (
        points @ matrix[:3, :3] + matrix[3, :3]
    ) * meters_per_unit
    digest = hashlib.sha256(world.tobytes() + faces.tobytes()).hexdigest()
    details = {
        "mesh_path": str(prim.GetPath()),
        "vertex_count": len(points),
        "triangle_count": len(faces),
        "world_matrix_usd_row_vector": matrix.tolist(),
        "geometry_sha256": digest,
        "double_sided": bool(mesh.GetDoubleSidedAttr().Get()),
        "orientation": str(mesh.GetOrientationAttr().Get()),
    }
    return world, faces, details


def material_info(prim, time_code):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    result = {
        "material_path": str(material.GetPath()) if material else None,
        "asset_inputs": [],
    }
    if material:
        for shader_prim in Usd.PrimRange(material.GetPrim()):
            if not shader_prim.IsA(UsdShade.Shader):
                continue
            for item in UsdShade.Shader(shader_prim).GetInputs():
                value = item.Get(time_code)
                if isinstance(value, Sdf.AssetPath):
                    result["asset_inputs"].append({
                        "attribute": str(item.GetAttr().GetPath()),
                        "authored": value.path,
                        "resolved": value.resolvedPath,
                        "local_file_exists": bool(
                            value.resolvedPath
                            and Path(value.resolvedPath).is_file()
                        ),
                    })
    return result


def choose_pose(vertices, faces, config):
    # 仅操作 NumPy 快照；通过 asyncio.to_thread 调用。
    math_result = math_check()
    query = Triangles(vertices, faces)
    low, high = vertices.min(0), vertices.max(0)
    fractions = np.array(list(itertools.product(
        (0.20, 0.35, 0.50, 0.65, 0.80), repeat=3
    )))
    candidates = low + fractions * (high - low)
    # 抽样顶点只用于候选排序，不用于内部或净空验收。
    sample = vertices[::max(1, len(vertices) // 4096)]
    ranking = sorted(
        candidates,
        key=lambda p: -float(np.linalg.norm(sample - p, axis=1).min()),
    )
    directions = [
        unit(d) for d in itertools.product((-1, 0, 1), repeat=3)
        if d != (0, 0, 0)
    ]
    attempts = []
    for point in ranking[:32]:
        clearance = query.clearance(point)
        if clearance < config["minimum_clearance_m"]:
            continue
        winding = query.winding(point)
        entry = {
            "position_world_m": point.tolist(),
            "clearance_mm": clearance * 1000,
            "winding": winding,
            "accepted": False,
        }
        attempts.append(entry)
        if not 0.90 <= abs(winding) <= 1.10:
            continue
        probes = [query.hits(point, d) for d in directions]
        odd_count = sum(len(hits) % 2 == 1 for hits in probes)
        entry["odd_parity_directions"] = odd_count
        if odd_count != len(directions):
            continue
        options = [
            (abs(float(hits[0]) - config["preferred_distance_m"]),
             direction, float(hits[0]))
            for direction, hits in zip(directions, probes)
            if len(hits)
            and config["minimum_clearance_m"] < hits[0]
            < config["far_m"] * 0.9
        ]
        if options:
            entry["accepted"] = True
            _, direction, distance = min(options, key=lambda item: item[0])
            return query, point, direction, {
                "math_self_check": math_result,
                "candidate_tests": attempts,
                "selected_distance_m": distance,
                "method": (
                    "exact triangle clearance + generalized winding "
                    "+ 26/26 odd-parity two-sided rays"
                ),
                "limitation": (
                    "胃模型有解剖开口；这是保守的几何内部检查，"
                    "不把开放曲面宣称为已证明封闭体积"
                ),
            }
    raise RuntimeError(
        "没有候选位置通过净空、绕数和26方向奇偶检查；"
        "停止，不创建未经验证的腔内相机"
    )


def camera_pose(stage, path, time_code, meters_per_unit):
    matrix = UsdGeom.XformCache(time_code).GetLocalToWorldTransform(
        stage.GetPrimAtPath(path)
    )
    position = np.array(matrix.ExtractTranslation()) * meters_per_unit
    direction = unit(matrix.TransformDir(Gf.Vec3d(0, 0, -1)))
    quaternion = matrix.ExtractRotationQuat()
    wxyz = [
        float(quaternion.GetReal()),
        *[float(v) for v in quaternion.GetImaginary()],
    ]
    return position, direction, wxyz, np.array(matrix)


def box_near(origin, direction, low, high):
    near, far = 0.0, float("inf")
    for axis in range(3):
        if abs(direction[axis]) < 1e-14:
            if origin[axis] < low[axis] or origin[axis] > high[axis]:
                return None
        else:
            a = (low[axis] - origin[axis]) / direction[axis]
            b = (high[axis] - origin[axis]) / direction[axis]
            near = max(near, min(a, b))
            far = min(far, max(a, b))
            if near > far:
                return None
    return near if far >= 0 else None


def possible_occluders(
    stage, target_path, helper_root, time_code, units, origin, axis, distance
):
    """筛选光轴前方其他可见几何；不依赖碰撞是否启用。"""
    bbox_cache = UsdGeom.BBoxCache(
        time_code, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    snapshots, uncertain = [], []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path == target_path or path.startswith(helper_root + "/"):
            continue
        if not (
            prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer)
        ) or not visible(prim, time_code):
            continue
        bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            continue
        near = box_near(
            origin, axis,
            np.array(bounds.GetMin()) * units,
            np.array(bounds.GetMax()) * units,
        )
        if near is None or near >= distance:
            continue
        if prim.IsA(UsdGeom.Mesh):
            try:
                vertices, faces, details = mesh_snapshot(
                    prim, time_code, units
                )
                snapshots.append((path, vertices, faces, details))
            except Exception as exc:
                uncertain.append({
                    "path": path, "reason": str(exc),
                    "bounding_box_near_m": near,
                })
        else:
            # 对未实现精确求交的曲线/解析几何/实例保守拒绝，
            # 不越过可能的遮挡物给出墙后距离。
            uncertain.append({
                "path": path,
                "reason": "non_triangle_geometry_may_occlude_axis",
                "bounding_box_near_m": near,
            })
    return snapshots, uncertain


def check_other_meshes(snapshots, origin, axis, wall_distance):
    blockers = []
    for path, vertices, faces, details in snapshots:
        hits = Triangles(vertices, faces).hits(origin, axis)
        if len(hits) and float(hits[0]) < wall_distance - 1e-7:
            blockers.append({
                "path": path,
                "first_hit_distance_m": float(hits[0]),
                "geometry_sha256": details["geometry_sha256"],
            })
    return blockers


def buffer_data(value):
    return value.get("data", value) if isinstance(value, dict) else value


def metadata_schema(value):
    """Describe the actual result, without inventing missing runtime fields."""
    if isinstance(value, dict):
        return {str(k): metadata_schema(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "dtype": str(value.dtype),
                "shape": list(value.shape), "fields": list(value.dtype.names or ())}
    return {"type": type(value).__name__, "value": json_value(value)}


def render_frame_identifier(metadata):
    """Parse known native frame counters; unknown schemas remain invalid."""
    candidates = []

    def visit(value, path="IsaacReadTimes"):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + "." + str(key))
        elif isinstance(value, (np.ndarray, np.void)) and value.dtype.names:
            for name in value.dtype.names:
                visit(value[name], path + "." + name)
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            visit(value[0], path)
        else:
            leaf = path.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
            # Only explicit frame-counter fields are accepted, not simulation
            # time, app update counts, capture_id, or arbitrary integer fields.
            if leaf in ("renderingFrameNumber", "swhFrameNumber", "frameNumber"):
                try:
                    a = np.asarray(value).reshape(-1)
                    if a.size != 1 or a.dtype.kind not in "iu":
                        return
                    number = int(a[0])
                    if number >= 0:
                        candidates.append({"source": path, "value": number})
                except (ValueError, TypeError, OverflowError):
                    pass

    visit(metadata)
    for name in ("renderingFrameNumber", "swhFrameNumber", "frameNumber"):
        selected = [c for c in candidates
                    if c["source"].rsplit(".", 1)[-1].rsplit(":", 1)[-1] == name]
        if selected:
            if len({v["value"] for v in selected}) != 1:
                return None
            return selected[0]
    return None


def value_digest(value):
    h = hashlib.sha256()
    if value is not None and not isinstance(value, (str, bytes, Sdf.AssetPath)):
        try:
            a = np.asarray(value)
            if a.dtype.kind in "biufc" and a.dtype != object:
                h.update(str((a.dtype.str, a.shape)).encode())
                h.update(np.ascontiguousarray(a).tobytes())
                return h.hexdigest()
        except (ValueError, TypeError):
            pass
    h.update(str(value).encode("utf-8"))
    return h.hexdigest()


def render_scene_signature(stage, camera_path, time_code):
    """Composed geometry/material/light/capture-camera state after warmup.

    UI perspective cameras and render-graph initialization are not image-scene
    edits. Visible geometry, materials, lights and the measured camera are.
    Keep per-property digests so an actual change has an actionable path.
    """
    signature = {}
    transform_dependencies = set()
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        is_render_internal = any(path == root or path.startswith(root + "/")
                                 for root in ("/Render", "/Replicator"))
        if is_render_internal:
            continue
        is_geometry = prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer)
        is_material = prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader)
        is_light = prim.HasAPI(UsdLux.LightAPI)
        if not (is_geometry or is_material or is_light or path == camera_path):
            continue
        signature[path + "|type"] = value_digest(prim.GetTypeName())
        for attr in prim.GetAttributes():
            signature[str(attr.GetPath())] = value_digest(attr.Get(time_code))
        for rel in prim.GetRelationships():
            signature[str(rel.GetPath())] = value_digest(tuple(map(str, rel.GetTargets())))
        xform = UsdGeom.Xformable(prim)
        if xform:
            signature[path + "|world_transform"] = value_digest(
                UsdGeom.XformCache(time_code).GetLocalToWorldTransform(prim))
        if is_geometry:
            signature[path + "|computed_visibility"] = value_digest(
                UsdGeom.Imageable(prim).ComputeVisibility(time_code))
        parent = prim.GetParent()
        while parent and not parent.IsPseudoRoot():
            transform_dependencies.add(str(parent.GetPath()))
            parent = parent.GetParent()
    return signature, transform_dependencies


def signature_changes(before, after):
    return sorted(k for k in before.keys() | after.keys()
                  if before.get(k) != after.get(k))


def relevant_notice_paths(paths, signature, transform_dependencies):
    relevant = []
    prims = {str(Sdf.Path(k.split("|", 1)[0]).GetPrimPath()) for k in signature}
    for path in paths:
        p = Sdf.Path(path)
        prim_path = str(p.GetPrimPath())
        if path in signature or prim_path in prims:
            relevant.append(path)
        elif prim_path in transform_dependencies and (
                not p.IsPropertyPath() or p.name.startswith("xformOp")
                or p.name in ("visibility", "purpose")):
            relevant.append(path)
    return sorted(set(relevant))



def allocate_helper_path(base, exists, token_factory):
    """New path per capture; never overwrite a prim, including inactive ones."""
    base = base.split("_run_", 1)[0]
    if not base.startswith("/") or base == "/":
        raise ValueError("Helper base must be a non-root absolute prim path")
    for _ in range(16):
        candidate = base + "_run_" + token_factory()
        if not exists(candidate):
            return candidate
    raise RuntimeError("Could not allocate an unused capture path")


def legacy_helper_name(name):
    return bool(re.fullmatch(
        r"GastricSingleFrameCapture(?:V[234])?(?:_run_[0-9a-f]+)?", name))


def mask_legacy_capture_helpers(stage, layer):
    """Mask recognized old camera+LED helpers in our reversible session layer.

    Do not edit their owning root/reference layer or remove original prims.
    Names alone do not establish ownership: require our camera/LED structure
    and reject any subtree containing mesh/other geometric model prims.
    """
    result = {"temporarily_masked": [], "skipped_unrecognized": []}
    for prim in list(stage.GetPseudoRoot().GetChildren()):
        if not legacy_helper_name(prim.GetName()):
            continue
        path = str(prim.GetPath())
        cam = stage.GetPrimAtPath(path + "/Camera")
        led = stage.GetPrimAtPath(path + "/Camera/LED")
        expected = bool(cam and cam.IsA(UsdGeom.Camera)
                        and led and led.HasAPI(UsdLux.LightAPI))
        contains_geometry = any(
            p.IsA(UsdGeom.Gprim) or p.IsA(UsdGeom.PointInstancer)
            for p in Usd.PrimRange(prim))
        if not expected or contains_geometry:
            result["skipped_unrecognized"].append(path)
            continue
        owner_layers = [spec.layer.identifier for spec in prim.GetPrimStack()]
        # Directly author in the owned anonymous layer: the current UI edit
        # target cannot redirect this opinion into the original model file.
        override = Sdf.CreatePrimInLayer(layer, Sdf.Path(path))
        override.active = False
        result["temporarily_masked"].append(
            {"path": path, "owner_layers_untouched": owner_layers})
    return result


def release_capture(state):
    stage, layer = state.get("stage"), state.get("layer")
    product = state.get("product")
    if stage is not None and layer is not None and product is not None:
        with Usd.EditContext(stage, layer):
            for annotator in state.get("annotators", {}).values():
                try:
                    annotator.detach([product.path])
                except Exception as exc:
                    print("[GASTRIC] detach warning:", exc)
            try:
                product.destroy()
            except Exception as exc:
                print("[GASTRIC] render product cleanup warning:", exc)
    state["product"] = None
    state["annotators"] = {}


def remove_owned(state):
    release_capture(state)
    if state.get("window") is not None:
        state["window"].destroy()
        state["window"] = None
    stage, layer = state.get("stage"), state.get("layer")
    if stage is not None and layer is not None:
        session = stage.GetSessionLayer()
        session.subLayerPaths = [
            path for path in session.subLayerPaths
            if path != layer.identifier
        ]
    state["layer"] = None


def gastric_capture_cleanup():
    state = getattr(builtins, REGISTRY, None)
    if not state:
        print("[GASTRIC] no owned capture nodes")
        return
    task = state.get("task")
    if task is not None and not task.done():
        state["cleanup_requested"] = True
        task.cancel()
        print("[GASTRIC] cancellation requested; cleanup pending")
    else:
        remove_owned(state)
        print("[GASTRIC] removed owned resources; original stage not saved/reloaded")


async def capture_one(state, config):
    capture_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    camera_path = config["helper_root"] + "/Camera"
    record = {
        "sample_id": capture_id + "_0001",
        "scene_id": None,
        "capture_id": capture_id,
        "trajectory_id": None,
        "scene_root_layer": None,
        "stomach_mesh_path": None,
        "camera_prim_path": camera_path,
        "camera_position_world_m": None,
        "camera_orientation_world": None,
        "quaternion_order": "wxyz",
        "camera_coordinate_convention": (
            "USD/OpenGL: local +X right, +Y up, optical forward local -Z"
        ),
        "optical_axis_world": None,
        "image_resolution": config["resolution"],
        "projection_parameters": None,
        "render_frame_id": None,
        "simulation_time_s": None,
        "hit_mesh_path": None,
        "hit_position_world_m": None,
        "distance_mm": None,
        "valid": False,
        "invalid_reasons": [],
        "image_path": None,
        "configuration": config,
        "status": "preparing",
        "script_revision": "v9-local-five-ten-degree-photos",
        "warnings": [],
        "verification_scope": "one frozen-state pinhole capture with render completion barrier; NOT moving-camera capture",
        "synchronization": {
            "method": "replicator_step_async_wait_for_render_static_state",
            "native_render_frame_id_available": False,
            "native_render_frame_id": None,
            "frame_id_note": "capture_id is a local transaction ID, not a native renderer frame number",
            "wait_for_render": True,
            "delta_time_s": 0.0,
            "render_completed": False,
        },
    }
    output = None
    notice = None
    timeline = None
    was_playing = False

    try:
        output = Path(config["output_root"]) / capture_id
        output.mkdir(parents=True, exist_ok=False)
        source_path = Path(config["script_source_path"])
        if source_path.is_file():
            shutil.copyfile(source_path, output / "capture_script.py")
            record["script_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            record["script_source_note"] = "copied saved source; run this exact saved file for provenance"
        else:
            record["warnings"].append("saved_script_source_missing")
        record["runtime"] = {"python": sys.version, "numpy": np.__version__}
        try:
            record["runtime"]["kit_version"] = omni.kit.app.get_app().get_version()
        except Exception as exc:
            record["runtime"]["kit_version_error"] = str(exc)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("当前没有 Stage；脚本不会自行打开场景")
        state["stage"] = stage
        requested_base = config["helper_root"].split("_run_", 1)[0]
        config["helper_root"] = allocate_helper_path(
            requested_base, lambda p: stage.GetPrimAtPath(p).IsValid(),
            lambda: uuid.uuid4().hex[:12])
        camera_path = config["helper_root"] + "/Camera"
        record["helper_root_requested_base"] = requested_base
        record["helper_root_actual"] = config["helper_root"]
        record["camera_prim_path"] = camera_path

        root = stage.GetRootLayer()
        record["scene_root_layer"] = root.realPath or root.identifier
        record["scene_id"] = hashlib.sha256(
            root.identifier.encode()
        ).hexdigest()[:16]
        record["layers_before"] = [
            {
                "identifier": layer.identifier,
                "real_path": layer.realPath,
                "dirty": layer.dirty,
            }
            for layer in stage.GetUsedLayers()
        ]

        units = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isfinite(units) or units <= 0:
            raise RuntimeError("场景单位无效")
        record["meters_per_unit"] = units
        up_axis = str(UsdGeom.GetStageUpAxis(stage))
        record["stage_up_axis"] = up_axis
        if config["replay_meters_per_unit"] is not None and not math.isclose(
                units, float(config["replay_meters_per_unit"]), rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Replay stage units differ from saved units")

        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()
        if was_playing:
            timeline.pause()  # 不调用 stop，不重置物理状态。
        await omni.kit.app.get_app().next_update_async()

        sim_time = float(timeline.get_current_time())
        time_code = Usd.TimeCode(sim_time * stage.GetTimeCodesPerSecond())
        record["simulation_time_s"] = sim_time
        record["usd_time_code"] = time_code.GetValue()

        prim = pick_mesh(stage, time_code, config)
        vertices, faces, geometry = mesh_snapshot(prim, time_code, units)
        target_path = str(prim.GetPath())
        if config["replay_geometry_sha256"] is not None and (
                geometry["geometry_sha256"] != config["replay_geometry_sha256"]):
            raise RuntimeError("Replay stomach world geometry differs from saved geometry")
        record["stomach_mesh_path"] = target_path
        record["geometry"] = geometry
        record["material"] = material_info(prim, time_code)
        record["mesh_prim_stack"] = [
            {"layer": spec.layer.identifier, "path": str(spec.path)}
            for spec in prim.GetPrimStack()
        ]
        print("[GASTRIC] stage:", record["scene_root_layer"])
        print("[GASTRIC] stomach_mesh:", target_path)
        print("[GASTRIC] checking cavity geometry asynchronously")

        if config["camera_world_matrix_replay"] is None:
            query, eye, direction, selection = await asyncio.to_thread(
                choose_pose, vertices, faces, config
            )
        else:
            query = Triangles(vertices, faces)
            replay_matrix = np.asarray(config["camera_world_matrix_replay"], dtype=float)
            if replay_matrix.shape != (4, 4) or not np.isfinite(replay_matrix).all():
                raise ValueError("Replay camera matrix must be finite 4x4")
            r = replay_matrix[:3, :3]
            if (not np.allclose(r @ r.T, np.eye(3), atol=1e-8)
                    or not np.isclose(np.linalg.det(r), 1.0, atol=1e-8)
                    or not np.allclose(replay_matrix[:3, 3], 0)
                    or not np.isclose(replay_matrix[3, 3], 1)):
                raise ValueError("Replay matrix must be a rigid USD row-vector transform")
            eye = replay_matrix[3, :3] * units
            direction = unit(np.array([0., 0., -1.]) @ r)
            clearance = await asyncio.to_thread(query.clearance, eye)
            winding = await asyncio.to_thread(query.winding, eye)
            if clearance < config["minimum_clearance_m"] or not 0.90 <= abs(winding) <= 1.10:
                raise RuntimeError("Replay pose failed conservative cavity/clearance check")
            forward_hits = await asyncio.to_thread(query.hits, eye, direction)
            if not len(forward_hits):
                raise RuntimeError("Replay pose has no forward stomach hit")
            selection = {"method": "saved_world_matrix_replay",
                         "selected_distance_m": float(forward_hits[0]),
                         "clearance_mm": clearance * 1000, "winding": winding}
        record["pose_selection"] = selection
        if omni.usd.get_context().get_stage() != stage:
            raise RuntimeError("准备期间 Stage 已改变")
        if abs(float(timeline.get_current_time()) - sim_time) > 1e-9:
            raise RuntimeError("准备期间仿真时间改变")
        if mesh_snapshot(prim, time_code, units)[2]["geometry_sha256"] != (
            geometry["geometry_sha256"]
        ):
            raise RuntimeError("准备期间胃壁几何改变")

        helper = config["helper_root"]
        if stage.GetPrimAtPath(helper).IsValid():
            raise RuntimeError("固定采集路径被其他对象占用，拒绝覆盖：" + helper)

        # Mask old reserved camera/LED roots before snapshot and warmup.
        # The masks and the new camera are owned by a single anonymous layer;
        # cleanup removes that layer and restores the previous composition.
        layer = Sdf.Layer.CreateAnonymous("gastric_single_frame_session.usda")
        state["layer"] = layer
        stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
        record["legacy_capture_helpers"] = mask_legacy_capture_helpers(stage, layer)
        print("[GASTRIC] masked old capture helpers:",
              [x["path"] for x in record["legacy_capture_helpers"]["temporarily_masked"]])

        # Save the base scene BEFORE adding this capture's camera/light.
        # The snapshot preserves the temporary disabling of old helper lights.
        if config["save_scene_snapshot"]:
            snapshot_path = output / "scene_snapshot.usdc"
            stage.Flatten().Export(str(snapshot_path))
            record["scene_snapshot_path"] = str(snapshot_path)
            record["scene_snapshot_sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        width, height = config["resolution"]
        if width % 2 != 1 or height % 2 != 1:
            raise RuntimeError("光轴中心像素检查要求奇数图像尺寸")

        up = np.array([0, 0, 1] if up_axis == "Z" else [0, 1, 0])
        if abs(float(up @ direction)) > 0.95:
            up = np.array([1, 0, 0])
        world_matrix = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*(eye / units)),
            Gf.Vec3d(*((eye + direction * 0.03) / units)),
            Gf.Vec3d(*[float(v) for v in up]),
        ).GetInverse()

        if config["camera_world_matrix_replay"] is not None:
            world_matrix = Gf.Matrix4d(*replay_matrix.reshape(-1).tolist())

        with Usd.EditContext(stage, layer):
            UsdGeom.Xform.Define(stage, helper)
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.AddTransformOp().Set(world_matrix)
            camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
            # USD 焦距/光圈长度的单位是“场景单位的十分之一”。
            aperture = (
                config["horizontal_aperture_mm"] * 0.001 / (0.1 * units)
            )
            focal = aperture / (
                2 * math.tan(math.radians(config["horizontal_fov_deg"] / 2))
            )
            camera.CreateHorizontalApertureAttr(aperture)
            camera.CreateVerticalApertureAttr(aperture * height / width)
            camera.CreateHorizontalApertureOffsetAttr(0.0)
            camera.CreateVerticalApertureOffsetAttr(0.0)
            camera.CreateFocalLengthAttr(focal)
            camera.CreateClippingRangeAttr(Gf.Vec2f(
                config["near_m"] / units, config["far_m"] / units
            ))
            camera.CreateFStopAttr(0.0)  # 关闭景深。
            camera.CreateFocusDistanceAttr(
                selection["selected_distance_m"] / units
            )

            light = UsdLux.SphereLight.Define(stage, camera_path + "/LED")
            light.AddTranslateOp().Set(Gf.Vec3d(*[
                value / units for value in config["led_offset_camera_m"]
            ]))
            light.CreateRadiusAttr(config["led_radius_m"] / units)
            light.CreateNormalizeAttr(True)
            light.CreateIntensityAttr(config["led_intensity"])
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(config["led_temperature_K"])
            light.CreateColorAttr(Gf.Vec3f(1.0))
            # 没有创建灯罩/胶囊等可遮挡镜头的额外几何。

        position, axis, orientation, matrix_before = camera_pose(
            stage, camera_path, time_code, units
        )
        record["camera_position_world_m"] = position.tolist()
        record["camera_orientation_world"] = orientation
        record["optical_axis_world"] = axis.tolist()
        record["camera_world_matrix_usd_row_vector"] = matrix_before.tolist()

        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        actual_focal = float(camera.GetFocalLengthAttr().Get())
        actual_ha = float(camera.GetHorizontalApertureAttr().Get())
        actual_va = float(camera.GetVerticalApertureAttr().Get())
        record["projection_parameters"] = {
            "projection": str(camera.GetProjectionAttr().Get()),
            "focal_length_usd": actual_focal,
            "horizontal_aperture_usd": actual_ha,
            "vertical_aperture_usd": actual_va,
            "focal_length_mm": actual_focal * units * 100,
            "horizontal_fov_deg": math.degrees(
                2 * math.atan(actual_ha / (2 * actual_focal))
            ),
            "vertical_fov_deg": math.degrees(
                2 * math.atan(actual_va / (2 * actual_focal))
            ),
            "clipping_range_m": [
                float(v) * units for v in camera.GetClippingRangeAttr().Get()
            ],
            "f_stop": float(camera.GetFStopAttr().Get()),
            "lens_model": "native pinhole; no fisheye/circular postprocessing",
        }

        # 使用已创建实际 Camera 的光心和局部 -Z 世界方向重新测量。
        hits = await asyncio.to_thread(query.hits, position, axis)
        wall_distance = float(hits[0]) if len(hits) else None
        if wall_distance is None:
            record["invalid_reasons"].append("no_forward_stomach_intersection")
        elif not config["near_m"] < wall_distance < config["far_m"]:
            record["invalid_reasons"].append("wall_outside_clipping_range")
        else:
            snapshots, uncertain = possible_occluders(
                stage, target_path, helper, time_code, units,
                position, axis, wall_distance,
            )
            blockers = await asyncio.to_thread(
                check_other_meshes, snapshots, position, axis, wall_distance
            )
            record["geometric_occlusion_check"] = {
                "blocking_meshes": blockers,
                "unverified_possible_occluders": uncertain,
            }
            if blockers:
                record["invalid_reasons"].append("real_mesh_before_stomach")
            if uncertain:
                record["invalid_reasons"].append(
                    "possible_occluder_cannot_be_verified"
                )

        limits = config.get("accepted_distance_range_mm")
        if limits is not None and (wall_distance is None or not
                float(limits[0]) <= wall_distance * 1000 <= float(limits[1])):
            record["invalid_reasons"].append("optical_axis_distance_outside_30_60_mm")

        # Textures remain external dependencies: verify/copy before moving PCs.
        dependencies = []
        dependency_paths = set()
        for item in record["material"]["asset_inputs"]:
            if item.get("resolved"):
                dependency_paths.add(item["resolved"])
        for used_layer in stage.GetUsedLayers():
            if used_layer.realPath:
                dependency_paths.add(used_layer.realPath)
        for dependency in sorted(dependency_paths):
            p = Path(dependency)
            dependencies.append({"path": dependency, "exists": p.is_file(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None})
        record["file_dependencies"] = dependencies
        scene_identity = {"snapshot_sha256": record.get("scene_snapshot_sha256"),
            "geometry_sha256": geometry["geometry_sha256"],
            "dependencies": [{"path": d["path"], "sha256": d["sha256"]} for d in dependencies]}
        record["scene_id"] = hashlib.sha256(json.dumps(
            scene_identity, sort_keys=True).encode()).hexdigest()[:16]

        # Do not bind this measurement camera to an interactive viewport.
        # Kit's viewport initialization/navigation wrote centerOfInterest and
        # xformOp:transform in the user's warmup trace. A saved-image display
        # is created only AFTER capture and export, without any camera binding.
        record["preview_mode"] = "exported_png_after_capture_no_interactive_camera_binding"

        import omni.replicator.core as rep
        from omni.replicator.core.functional import write_image

        status = rep.orchestrator.get_status()
        if "STARTED" in str(getattr(status, "name", status)).upper():
            raise RuntimeError(
                "发现其他正在运行的 Replicator 任务；拒绝干扰该任务"
            )

        with Usd.EditContext(stage, layer):
            product = rep.create.render_product(
                camera_path, (width, height), name="GastricSingleFrameProduct"
            )
            state["product"] = product
            annotators = {}
            state["annotators"] = annotators
            for name in (
                "rgb", "distance_to_camera", "instance_id_segmentation"
            ):
                annotator = rep.AnnotatorRegistry.get_annotator(name)
                annotators[name] = annotator
                annotator.attach([product.path])
            # IsaacReadTimes is unavailable in the user's local build.
            # The transaction below uses the documented static-render barrier,
            # not a fabricated renderer frame counter.

        record["render_product_path"] = str(product.path)
        changed_paths = []

        def changed(event, sender):
            for path in (
                list(event.GetResyncedPaths())
                + list(event.GetChangedInfoOnlyPaths())
            ):
                p = str(path.GetPrimPath())
                changed_paths.append(str(path))

        notice = Tf.Notice.Register(
            Usd.Notice.ObjectsChanged, changed, stage
        )
        print("[GASTRIC] camera:", camera_path)
        print("[GASTRIC] warming up; do not edit stage")

        for _ in range(config["warmup_steps"]):
            # Replicator 可能创建的图和属性也写入本脚本的 Session 子层。
            with Usd.EditContext(stage, layer):
                await asyncio.wait_for(
                    rep.orchestrator.step_async(
                        delta_time=0.0,
                        pause_timeline=True,
                        wait_for_render=True,
                        rt_subframes=config["rt_subframes"],
                    ),
                    timeout=config["render_timeout_s"],
                )

        # Baseline only after graph/material initialization; warmup notices
        # remain recorded, but they are not part of the measured capture.
        record["warmup_stage_changes"] = sorted(set(changed_paths))
        changed_paths.clear()
        scene_before, transform_dependencies = render_scene_signature(
            stage, camera_path, time_code)
        # Scene initialization is allowed during warmup, not during capture.
        # The originally measured camera, stomach and simulation time must
        # still match before submitting the measured render.
        _, _, _, before_capture_matrix = camera_pose(stage, camera_path, time_code, units)
        record["camera_world_matrix_after_warmup"] = before_capture_matrix.tolist()
        record["warmup_camera_change"] = {
            "max_matrix_difference": float(np.max(np.abs(matrix_before-before_capture_matrix))),
            "translation_difference_mm": float(np.linalg.norm(
                matrix_before[3,:3]-before_capture_matrix[3,:3])*units*1000),
            "rotation_matrix_difference_norm": float(np.linalg.norm(
                matrix_before[:3,:3]-before_capture_matrix[:3,:3])),
        }
        if not np.allclose(matrix_before, before_capture_matrix, rtol=0, atol=1e-12):
            raise RuntimeError("Camera changed during warmup")
        if mesh_snapshot(prim, time_code, units)[2]["geometry_sha256"] != geometry["geometry_sha256"]:
            raise RuntimeError("Stomach geometry changed during warmup")
        if abs(float(timeline.get_current_time()) - sim_time) > 1e-9:
            raise RuntimeError("Simulation time changed during warmup")
        if timeline.is_playing():
            raise RuntimeError("Static capture requires a paused timeline")
        record["synchronization"]["timeline_paused_before"] = True
        record["capture_begin_utc"] = datetime.now(timezone.utc).isoformat()
        with Usd.EditContext(stage, layer):
            await asyncio.wait_for(rep.orchestrator.step_async(
                delta_time=0.0, pause_timeline=True, wait_for_render=True,
                rt_subframes=config["rt_subframes"]),
                timeout=config["render_timeout_s"])

        # step_async returns only after rendering/annotation completion with
        # wait_for_render=True. Freeze state and copy the same product's three
        # buffers without yielding; inspect state both sides of this operation.
        record["synchronization"]["render_completed"] = True

        rgb = np.array(buffer_data(annotators["rgb"].get_data()), copy=True)
        depth = np.array(
            buffer_data(annotators["distance_to_camera"].get_data()),
            copy=True,
        )
        segmentation = annotators["instance_id_segmentation"].get_data()
        ids = np.array(buffer_data(segmentation), copy=True)
        record["render_timing_raw"] = None
        record["render_timing_schema"] = {"available": False,
            "reason": "IsaacReadTimes not requested in v3; static render barrier used"}
        record["capture_buffers_read_utc"] = datetime.now(timezone.utc).isoformat()
        scene_after, _ = render_scene_signature(stage, camera_path, time_code)
        record["scene_signature_changed_keys"] = signature_changes(scene_before, scene_after)
        record["capture_stage_changes"] = sorted(set(changed_paths))
        record["capture_relevant_stage_changes"] = relevant_notice_paths(
            changed_paths, scene_before, transform_dependencies)
        record["render_scene_signature_before"] = scene_before
        record["render_scene_signature_after"] = scene_after
        record["render_frame_id"] = None
        record["warnings"].append("native_render_frame_id_unavailable_static_barrier_used")
        record["synchronization"]["timeline_paused_after"] = not timeline.is_playing()
        if timeline.is_playing():
            record["invalid_reasons"].append("timeline_resumed_during_static_capture")

        _, _, _, matrix_after = camera_pose(
            stage, camera_path, time_code, units
        )
        if omni.usd.get_context().get_stage() != stage:
            record["invalid_reasons"].append("stage_changed")
        if not np.allclose(matrix_before, matrix_after, rtol=0, atol=1e-12):
            record["invalid_reasons"].append("camera_moved")
        if abs(float(timeline.get_current_time()) - sim_time) > 1e-9:
            record["invalid_reasons"].append("simulation_time_changed")
        if mesh_snapshot(prim, time_code, units)[2]["geometry_sha256"] != (
            geometry["geometry_sha256"]
        ):
            record["invalid_reasons"].append("stomach_geometry_changed")
        if (record["scene_signature_changed_keys"]
                or record["capture_relevant_stage_changes"]):
            record["unexpected_stage_changes"] = sorted(set(
                record["scene_signature_changed_keys"]
                + record["capture_relevant_stage_changes"]))
            record["invalid_reasons"].append("scene_changed_during_capture")

        if (
            rgb.shape not in ((height, width, 3), (height, width, 4))
            or rgb.dtype != np.uint8
        ):
            raise RuntimeError(
                "RGB 缓冲尺寸/类型异常：" + str((rgb.shape, str(rgb.dtype)))
            )
        if depth.shape[:2] != (height, width) or ids.shape[:2] != (
            height, width
        ):
            raise RuntimeError("RGB、深度、实例ID尺寸不一致")

        np.save(output / "depth_m.npy", depth, allow_pickle=False)
        np.save(output / "instance_ids.npy", ids, allow_pickle=False)
        record["instance_id_labels"] = json_value(
            segmentation.get("info", {}) if isinstance(segmentation, dict) else {})
        rgb = np.ascontiguousarray(rgb[..., :3])
        center_y, center_x = height // 2, width // 2
        instance_id = int(np.asarray(
            ids[center_y, center_x]
        ).reshape(-1)[0])
        renderer_distance = float(np.asarray(
            depth[center_y, center_x]
        ).reshape(-1)[0])

        labels = (
            segmentation.get("info", {}).get("idToLabels", {})
            if isinstance(segmentation, dict) else {}
        )
        label = labels.get(str(instance_id), labels.get(instance_id))
        if isinstance(label, dict):
            label = label.get("primPath", label.get("prim_path"))
        same_mesh = isinstance(label, str) and (
            label == target_path or label.startswith(target_path + "/")
        )
        record["renderer_occlusion_check"] = {
            "center_pixel_xy": [center_x, center_y],
            "instance_id": instance_id,
            "instance_path": json_value(label),
            "distance_to_camera_m": json_value(renderer_distance),
        }
        if not same_mesh:
            record["invalid_reasons"].append(
                "center_occluded_or_target_instance_path_unavailable"
            )
        if not math.isfinite(renderer_distance) or renderer_distance <= 0:
            record["invalid_reasons"].append("invalid_renderer_depth")
        elif wall_distance is not None:
            tolerance = max(
                config["depth_tolerance_m"], 0.005 * wall_distance
            )
            record["renderer_occlusion_check"]["tolerance_m"] = tolerance
            if abs(renderer_distance - wall_distance) > tolerance:
                record["invalid_reasons"].append(
                    "renderer_depth_disagrees_with_first_stomach_hit"
                )

        luminance = rgb.mean(axis=2)
        record["image_quality"] = {
            "mean_rgb_0_255": rgb.mean(axis=(0, 1)).tolist(),
            "dark_pixel_fraction": float((luminance < 5).mean()),
            "clipped_pixel_fraction": float(
                (rgb.max(axis=2) >= 250).mean()
            ),
            "texture_visual_inspection": "pending_user_review",
        }
        if record["image_quality"]["dark_pixel_fraction"] > 0.95:
            record["invalid_reasons"].append("image_almost_black")
        if record["image_quality"]["clipped_pixel_fraction"] > 0.95:
            record["invalid_reasons"].append("image_almost_fully_clipped")

        # 直接导出相机 RGB 数据，不截取界面。
        image_path = output / "image.png"
        await asyncio.to_thread(
            write_image, path=str(image_path), data=rgb
        )
        record["image_path"] = str(image_path)
        record["image_sha256"] = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        record["geometric_diagnostic_only"] = {
            "forward_distance_m": wall_distance,
            "forward_hit_position_world_m": (
                (position + axis * wall_distance).tolist() if wall_distance is not None else None),
            "not_a_valid_training_label_when_valid_false": True,
        }
        record["synchronization"].update({
            "camera_world_matrix_unchanged": bool(np.allclose(
                matrix_before, matrix_after, rtol=0, atol=1e-12)),
            "simulation_time_unchanged": abs(float(timeline.get_current_time()) - sim_time) <= 1e-9,
            "render_scene_signature_unchanged": not record["scene_signature_changed_keys"],
            "no_relevant_scene_notices": not record["capture_relevant_stage_changes"],
            "same_render_product": str(product.path),
            "buffers_read_without_await": True,
            "central_instance_matches_stomach": bool(same_mesh),
            "central_depth_agrees_with_geometric_first_hit": (
                wall_distance is not None and math.isfinite(renderer_distance)
                and renderer_distance > 0
                and abs(renderer_distance - wall_distance) <= max(
                    config["depth_tolerance_m"], 0.005 * wall_distance)),
        })
        record["valid"] = (
            wall_distance is not None and not record["invalid_reasons"]
            and record["synchronization"]["render_completed"]
        )
        if record["valid"]:
            record["hit_mesh_path"] = target_path
            record["hit_position_world_m"] = (
                position + axis * wall_distance
            ).tolist()
            record["distance_mm"] = wall_distance * 1000
        # 无效样本保留 null，不输出墙后距离作为标签。
        record["status"] = (
            "captured_valid" if record["valid"] else "captured_invalid"
        )

        # This is only a viewer for the exported PNG. It does not render a
        # second image, touch the camera transform, or add labels to image.png.
        try:
            if not config.get("show_export_preview", True):
                raise _SkipExportPreview()
            import omni.ui as ui
            window = ui.Window("Gastric | Exported Frame", width=700, height=740)
            state["window"] = window
            with window.frame:
                with ui.VStack():
                    text = ("VALID | distance %.6f mm" % record["distance_mm"]
                            if record["valid"] else "INVALID - inspect sample.json")
                    ui.Label(text, height=24)
                    ui.Image(str(image_path), width=680, height=680)
        except _SkipExportPreview:
            pass
        except Exception as exc:
            record["warnings"].append("exported_image_preview_unavailable")
            record["preview_error"] = str(exc)
            print("[GASTRIC] PNG saved; preview unavailable:", exc)

    except asyncio.CancelledError:
        record["status"] = "cancelled"
        record["invalid_reasons"].append("capture_cancelled")
    except Exception as exc:
        record["status"] = "failed"
        record["invalid_reasons"].append(str(exc))
        record["traceback"] = traceback.format_exc()
        print("[GASTRIC] ERROR:\n", record["traceback"])
    finally:
        if notice is not None:
            notice.Revoke()
        try:
            release_capture(state)
            if state.get("cleanup_requested"):
                remove_owned(state)
        except Exception as exc:
            record["cleanup_error"] = str(exc)

        if (
            timeline is not None and was_playing
            and omni.usd.get_context().get_stage() == state.get("stage")
        ):
            timeline.play()

        if output is not None:
            try:
                (output / "config.json").write_text(
                    json.dumps(config, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (output / "sample.json").write_text(
                    json.dumps(
                        json_value(record), ensure_ascii=False,
                        indent=2, allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                if record["camera_position_world_m"] is not None:
                    replay = dict(config)
                    replay["stomach_mesh_path"] = record["stomach_mesh_path"]
                    replay["camera_world_matrix_replay"] = record["camera_world_matrix_usd_row_vector"]
                    replay["replay_meters_per_unit"] = record["meters_per_unit"]
                    replay["replay_geometry_sha256"] = record["geometry"]["geometry_sha256"]
                    (output / "replay_config.json").write_text(
                        json.dumps(json_value(replay), ensure_ascii=False, indent=2), encoding="utf-8")
                manifest = {}
                for p in sorted(output.iterdir()):
                    if p.is_file():
                        manifest[p.name] = {"bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                (output / "SHA256SUMS.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                state["last_output"] = str(output)
                print("[GASTRIC] output:", output)
            except Exception:
                print("[GASTRIC] record write failed:\n", traceback.format_exc())

        print(
            "[GASTRIC] DONE",
            "status=", record["status"],
            "valid=", record["valid"],
            "distance_mm=", record["distance_mm"],
        )
        print("[GASTRIC] invalid_reasons:", record["invalid_reasons"])
        print("[GASTRIC] cleanup: gastric_capture_cleanup()")
    return record


def gastric_capture_run(config_override=None):
    previous = getattr(builtins, REGISTRY, None)
    if previous:
        task = previous.get("task")
        if task is not None and not task.done():
            print("[GASTRIC] one capture already running; no duplicate started")
            return
        remove_owned(previous)

    state = {
        "stage": None, "layer": None, "window": None,
        "product": None, "annotators": {}, "cleanup_requested": False,
    }
    setattr(builtins, REGISTRY, state)
    runtime_config = json.loads(json.dumps(
        CONFIG if config_override is None else config_override))
    # runpy supplies the actual executing file, even if the user saved it as
    # capture_one_frame.py. No manual script_source_path edit is needed.
    executing_file = globals().get("__file__")
    if executing_file and Path(executing_file).is_file():
        runtime_config["script_source_path"] = str(Path(executing_file).resolve())
    state["task"] = asyncio.ensure_future(capture_one(state, runtime_config))
    print("[GASTRIC] scheduled one async capture; original stage not saved/reloaded")


def gastric_capture_replay(config_path):
    """Replay a saved pose on the same verified world geometry and units."""
    saved = json.loads(Path(config_path).read_text(encoding="utf-8"))
    gastric_capture_run(saved)




# ---------------- v6: calibration and physically different viewpoints -------
ORBIT_REGISTRY = '_gastric_orbit_calibration_v6'
ORBIT_HOME = Path('/home/multirobo/gastric-single-frame-20261117/orbit_v6')
CLASS_MAP = {'normal': 0, 'raised': 1, 'ulcerated': 2}


def orbit_atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(json_value(value), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def orbit_pixel_ray(sample, xy):
    width, height = sample['image_resolution']
    x, y = map(float, xy)
    if not (0 <= x <= width-1 and 0 <= y <= height-1):
        raise ValueError('Selected pixel is outside the raw image')
    tan_h = math.tan(math.radians(sample['projection_parameters']['horizontal_fov_deg']) / 2)
    tan_v = math.tan(math.radians(sample['projection_parameters']['vertical_fov_deg']) / 2)
    local = np.array([(2*(x+.5)/width-1)*tan_h,
                      (1-2*(y+.5)/height)*tan_v, -1.])
    matrix = np.asarray(sample['camera_world_matrix_usd_row_vector'], float)
    origin = matrix[3, :3] * float(sample['meters_per_unit'])
    return origin, unit(local @ matrix[:3, :3])


def orbit_project(points, matrix, units, resolution, fov_deg):
    matrix = np.asarray(matrix, float)
    points = np.atleast_2d(np.asarray(points, float))
    local = (points - matrix[3, :3]*units) @ matrix[:3, :3].T
    depth = -local[:, 2]
    focal_px = resolution[0] / (2*math.tan(math.radians(fov_deg)/2))
    safe = np.where(depth > 1e-12, depth, np.nan)
    xy = np.column_stack([resolution[0]/2-.5 + focal_px*local[:, 0]/safe,
                          resolution[1]/2-.5 - focal_px*local[:, 1]/safe])
    return xy, depth


def orbit_row_rotation(a, b):
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1+1e-9:
        axis = unit(np.cross(a, [1., 0., 0.]) if abs(a[0]) < .9
                    else np.cross(a, [0., 1., 0.]))
        return 2*np.outer(axis, axis)-np.eye(3)
    x, y, z = v
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return (np.eye(3)+skew+skew@skew/max(1+c, 1e-12)).T


def orbit_camera_matrix(eye, target, units, resolution, fov_deg, target_xy):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = unit(target-eye)
    up = np.array([0., 0., 1.])
    if abs(float(up @ forward)) > .95:
        up = np.array([0., 1., 0.])
    right = unit(np.cross(forward, up))
    up = unit(np.cross(right, forward))
    basis = np.stack([right, up, -forward])
    width, height = resolution
    focal = width/(2*math.tan(math.radians(fov_deg)/2))
    local_target = unit([(target_xy[0]-(width/2-.5))/focal,
                         -(target_xy[1]-(height/2-.5))/focal, -1.])
    matrix = np.eye(4)
    matrix[:3, :3] = orbit_row_rotation(local_target, [0., 0., -1.]) @ basis
    matrix[3, :3] = eye/units
    return matrix


def orbit_hit_detail(query, origin, direction):
    direction = unit(direction)
    p = np.cross(np.broadcast_to(direction, query.e2.shape), query.e2)
    det = np.einsum('ij,ij->i', query.e1, p)
    scale = np.linalg.norm(query.e1, axis=1)*np.linalg.norm(query.e2, axis=1)
    good = np.abs(det) > 1e-12*np.maximum(scale, 1e-20)
    inv = np.zeros_like(det)
    inv[good] = 1/det[good]
    delta = np.asarray(origin)-query.a
    u = np.einsum('ij,ij->i', delta, p)*inv
    q = np.cross(delta, query.e1)
    v = (q @ direction)*inv
    t = np.einsum('ij,ij->i', query.e2, q)*inv
    good &= (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (t > 1e-8)
    indices = np.flatnonzero(good)
    if not len(indices):
        raise ValueError('Selected ray does not hit the stomach')
    idx = int(indices[np.argmin(t[indices])])
    point = np.asarray(origin)+direction*float(t[idx])
    normal = unit(np.cross(query.e1[idx], query.e2[idx]))
    if float(normal @ (np.asarray(origin)-point)) < 0:
        normal = -normal
    return point, normal, idx, float(t[idx])


def orbit_circle_fits(center_xy, radius_px, resolution, fraction=.42):
    center = np.array([(resolution[0]-1)/2, (resolution[1]-1)/2])
    return (np.isfinite(center_xy).all() and math.isfinite(radius_px)
            and radius_px >= 0
            and np.linalg.norm(np.asarray(center_xy)-center)+radius_px
            <= min(resolution)*fraction)


def orbit_candidate(region, azimuth_deg, cone_deg, target_radius_mm,
                    image_offset_xy, units, resolution, fov_deg):
    target = np.asarray(region['center_world_m'], float)
    normal = unit(region['inward_normal_world'])
    tangent = unit(np.cross(normal, [0., 0., 1.]) if abs(normal[2]) < .9
                   else np.cross(normal, [0., 1., 0.]))
    bitangent = unit(np.cross(normal, tangent))
    az, cone = math.radians(azimuth_deg), math.radians(cone_deg)
    radial = (math.cos(cone)*normal + math.sin(cone)*
              (math.cos(az)*tangent+math.sin(az)*bitangent))
    eye = target + target_radius_mm*.001*radial
    center_xy = [(resolution[0]-1)/2+image_offset_xy[0],
                 (resolution[1]-1)/2+image_offset_xy[1]]
    matrix = orbit_camera_matrix(eye, target, units, resolution, fov_deg, center_xy)
    # A deliberately conservative enclosing sphere of the user-selected region.
    # This is only a geometric framing check, never a semantic mask guarantee.
    _, depths = orbit_project([target], matrix, units, resolution, fov_deg)
    radius_m = float(region['visible_region_enclosing_radius_m'])
    focal = resolution[0]/(2*math.tan(math.radians(fov_deg)/2))
    center_offset = np.asarray(center_xy)-[(resolution[0]-1)/2, (resolution[1]-1)/2]
    perspective_factor = math.sqrt(1+float(center_offset@center_offset)/focal**2)
    screen_radius = focal*radius_m*perspective_factor/max(float(depths[0])-radius_m, 1e-12)
    return {'eye_m': eye.tolist(), 'matrix': matrix.tolist(),
            'azimuth_deg': float(azimuth_deg)%360, 'cone_deg': float(cone_deg),
            'camera_to_marked_center_mm': float(target_radius_mm),
            'target_pixel_xy': center_xy, 'conservative_radius_px': screen_radius,
            'framing_ok': bool(depths[0] > radius_m and
                              orbit_circle_fits(center_xy, screen_radius, resolution))}


class GastricOrbitPanel:
    def __init__(self):
        import omni.ui as ui
        self.ui = ui
        self.task = None
        self.sample = None
        self.query = None
        self.query_hash = None
        self.geometry = None
        self.targets = {}
        self.stop_requested = False
        self.preview_index = 0
        self.sample_dir = None
        ORBIT_HOME.mkdir(parents=True, exist_ok=True)
        saved = ORBIT_HOME/'targets.json'
        if saved.is_file():
            self.targets = json.loads(saved.read_text(encoding='utf-8')).get('regions', {})
        self.window = ui.Window('Gastric | Small Views: 50 photos', width=980, height=825)
        with self.window.frame:
            with ui.VStack(spacing=5):
                self.status = ui.Label('Load a valid saved capture to start.', height=32,
                                       word_wrap=True)
                with ui.HStack(height=26, spacing=4):
                    ui.Button('1 Load latest valid', clicked_fn=lambda: self.start(self.load_latest))
                    ui.Button('2 Recenter on cross', clicked_fn=lambda: self.start(self.recenter))
                    ui.Button('Stop after current frame', clicked_fn=self.stop)
                with ui.HStack(spacing=8):
                    self.image = ui.Image('', width=600, height=600)
                    with ui.VStack(width=340, spacing=6):
                        ui.Label('Move X/Y cross to the lesion center.', height=24)
                        ui.Label('Then enlarge R to cover the WHOLE lesion.', height=26)
                        self.x = ui.SimpleIntModel(400)
                        self.y = ui.SimpleIntModel(400)
                        self.radius = ui.SimpleIntModel(60)
                        for title, model, minimum, maximum in (
                                ('X (left/right)', self.x, 0, 800),
                                ('Y (top/down)', self.y, 0, 800),
                                ('R (region radius)', self.radius, 4, 350)):
                            ui.Label(title, height=20)
                            ui.IntSlider(model, min=minimum, max=maximum, height=26)
                            model.add_value_changed_fn(lambda m: self.draw_overlay())
                        ui.Label('If cropped: move cross onto lesion, Recenter,', height=22)
                        ui.Label('then mark its complete extent in the new image.', height=22)
                        with ui.HStack(height=28, spacing=3):
                            ui.Button('Look left', clicked_fn=lambda: self.start(lambda: self.look(-45, 0)))
                            ui.Button('Look right', clicked_fn=lambda: self.start(lambda: self.look(45, 0)))
                        with ui.HStack(height=28, spacing=3):
                            ui.Button('Look up', clicked_fn=lambda: self.start(lambda: self.look(0, 30)))
                            ui.Button('Look down', clicked_fn=lambda: self.start(lambda: self.look(0, -30)))
                        ui.Button('3 Save RAISED (zhong qi)', height=32,
                                  clicked_fn=lambda: self.start(lambda: self.save_region('raised')))
                        ui.Button('3 Save ULCERATED (kui lan)', height=32,
                                  clicked_fn=lambda: self.start(lambda: self.save_region('ulcerated')))
                        ui.Button('4 Collect RAISED: 50 small views', height=32,
                                  clicked_fn=lambda: self.start(lambda: self.pilot('raised')))
                        ui.Button('4 Collect ULCERATED: 50 small views', height=32,
                                  clicked_fn=lambda: self.start(lambda: self.pilot('ulcerated')))
                        ui.Label('X/Y/R only annotate a saved PNG.', height=22)
                        ui.Label('They never write overlays into image.png.', height=22)
                with ui.HStack(height=30, spacing=5):
                    ui.Button('5 Collect RAISED: 50 images',
                              clicked_fn=lambda: self.start(lambda: self.collect('raised', 50)))
                    ui.Button('5 Collect ULCERATED: 50 images',
                              clicked_fn=lambda: self.start(lambda: self.collect('ulcerated', 50)))
                ui.Label('Count target: 50 distinct usable images per lesion class; labels pending review.', height=24)
        self.start(self.load_latest)

    def message(self, text):
        self.status.text = text
        print('[ORBIT]', text)

    def start(self, factory):
        if self.task is not None and not self.task.done():
            self.message('Busy. Wait for current operation, or Stop after current frame.')
            return
        async def guarded():
            try:
                await factory()
            except asyncio.CancelledError:
                self.message('Operation cancelled; partial outputs are retained.')
                raise
            except Exception as exc:
                self.message('ERROR: '+str(exc))
                print(traceback.format_exc())
        self.task = asyncio.ensure_future(guarded())

    def stop(self):
        self.stop_requested = True
        self.message('Stop requested. Current static frame may finish; files are retained.')

    async def load_latest(self):
        self.message('Finding latest valid saved frame...')
        root = Path(CONFIG['output_root'])
        files = sorted(root.glob('*/sample.json'), reverse=True)
        for path in files:
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                if (record.get('valid') and (path.parent/'image.png').is_file()
                        and (path.parent/'depth_m.npy').is_file()):
                    self.set_sample(record, path.parent)
                    self.message('Loaded '+record['capture_id']+'. Mark a complete lesion with X/Y/R.')
                    return
            except (ValueError, OSError):
                continue
        raise RuntimeError('No valid saved frame in '+str(root))

    def set_sample(self, sample, directory):
        directory = Path(directory)
        image_path = directory/'image.png'
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != sample['image_sha256']:
            raise RuntimeError('Saved PNG hash differs from sample.json; select the original capture')
        if sample['image_resolution'] != [801, 801]:
            raise ValueError('Calibration UI currently requires the validated 801 x 801 capture')
        self.sample, self.sample_dir = sample, directory
        self.x.set_value(400)
        self.y.set_value(400)
        self.draw_overlay()

    def draw_overlay(self):
        if self.sample is None:
            return
        try:
            from PIL import Image, ImageDraw
            with Image.open(self.sample_dir/'image.png') as source:
                img = source.convert('RGB')
            draw = ImageDraw.Draw(img)
            x, y, r = self.x.as_int, self.y.as_int, self.radius.as_int
            draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 230, 0), width=3)
            draw.line((x-12, y, x+12, y), fill=(0, 255, 255), width=3)
            draw.line((x, y-12, x, y+12), fill=(0, 255, 255), width=3)
            draw.text((12, 12), 'CALIBRATION ONLY: X=%d Y=%d R=%d'%(x, y, r), fill='white')
            self.preview_index += 1
            preview = ORBIT_HOME/('annotation_preview_%d.png'%self.preview_index)
            img.save(preview)
            self.image.source_url = str(preview)
            # Keep only a few UI cache images. Never touch raw capture images.
            old = ORBIT_HOME/('annotation_preview_%d.png'%(self.preview_index-4))
            if old.is_file():
                old.unlink()
        except Exception as exc:
            self.message('Preview unavailable: '+str(exc))

    async def get_query(self):
        if self.sample is None:
            raise RuntimeError('Load a valid sample first')
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError('No open stage')
        if omni.timeline.get_timeline_interface().is_playing():
            raise RuntimeError('Pause the timeline before calibration/pilot')
        units = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isclose(units, self.sample['meters_per_unit'], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError('Current stage units differ from the loaded sample')
        time = omni.timeline.get_timeline_interface().get_current_time()
        prim = stage.GetPrimAtPath(self.sample['stomach_mesh_path'])
        vertices, faces, info = mesh_snapshot(prim, Usd.TimeCode(time*stage.GetTimeCodesPerSecond()), units)
        if info['geometry_sha256'] != self.sample['geometry']['geometry_sha256']:
            raise RuntimeError('Current stomach geometry differs from the loaded sample')
        if self.query_hash != info['geometry_sha256']:
            self.query = await asyncio.to_thread(Triangles, vertices, faces)
            self.query_hash = info['geometry_sha256']
        self.geometry = info
        return self.query

    async def surface_pixel(self, xy):
        x, y = map(lambda v: int(round(v)), xy)
        origin, ray = orbit_pixel_ray(self.sample, [x, y])
        hit = await asyncio.to_thread(orbit_hit_detail, self.query, origin, ray)
        depth = np.load(self.sample_dir/'depth_m.npy', mmap_mode='r', allow_pickle=False)
        ids = np.load(self.sample_dir/'instance_ids.npy', mmap_mode='r', allow_pickle=False)
        rendered = float(np.asarray(depth[y, x]).reshape(-1)[0])
        instance = int(np.asarray(ids[y, x]).reshape(-1)[0])
        labels = self.sample.get('instance_id_labels', {}).get('idToLabels', {})
        label = labels.get(str(instance), labels.get(instance))
        if isinstance(label, dict):
            label = label.get('primPath', label.get('prim_path'))
        target = self.sample['stomach_mesh_path']
        if label != target and not (isinstance(label, str) and label.startswith(target+'/')):
            raise RuntimeError('Selected pixel is not the stomach instance')
        if not math.isfinite(rendered) or abs(rendered-hit[3]) > max(.0005, .005*hit[3]):
            raise RuntimeError('Selected pixel depth does not agree with the geometric first hit')
        return hit

    async def capture(self, matrix, root, distance_gate=False):
        previous = getattr(builtins, REGISTRY, None)
        if previous:
            task = previous.get('task')
            if task is not None and not task.done() and task is not asyncio.current_task():
                raise RuntimeError('Another capture is running; wait for its DONE')
            remove_owned(previous)
        config = json.loads(json.dumps(self.sample['configuration']))
        config.update({'camera_world_matrix_replay': np.asarray(matrix).tolist(),
                       'replay_geometry_sha256': self.sample['geometry']['geometry_sha256'],
                       'replay_meters_per_unit': self.sample['meters_per_unit'],
                       'stomach_mesh_path': self.sample['stomach_mesh_path'],
                       'script_source_path': str(Path(__file__).resolve()),
                       'show_export_preview': False, 'save_scene_snapshot': True,
                       'output_root': str(root)})
        config.pop('accepted_distance_range_mm', None)
        if distance_gate:
            config['accepted_distance_range_mm'] = [30., 60.]
        state = {'stage': None, 'layer': None, 'window': None, 'product': None,
                 'annotators': {}, 'cleanup_requested': False, 'task': asyncio.current_task()}
        setattr(builtins, REGISTRY, state)
        try:
            return await capture_one(state, config)
        finally:
            remove_owned(state)

    async def recenter(self):
        await self.get_query()
        xy = [self.x.as_int, self.y.as_int]
        hit = await self.surface_pixel(xy)
        eye = self.sample['camera_position_world_m']
        units = self.sample['meters_per_unit']
        matrix = orbit_camera_matrix(eye, hit[0], units, [801, 801],
                                     self.sample['projection_parameters']['horizontal_fov_deg'], [400, 400])
        self.message('Recentring selected surface point; this scout frame is not training data...')
        record = await self.capture(matrix, ORBIT_HOME/'scouts')
        if not record.get('valid'):
            raise RuntimeError('Scout capture failed: '+str(record.get('invalid_reasons')))
        self.set_sample(record, Path(record['image_path']).parent)
        self.message('Recentered. Enclose the WHOLE lesion with X/Y/R, then Save its class.')

    async def look(self, yaw_deg, pitch_deg):
        await self.get_query()
        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        local_axis = np.array([math.sin(yaw)*math.cos(pitch), math.sin(pitch),
                               -math.cos(yaw)*math.cos(pitch)])
        basis = np.asarray(self.sample['camera_world_matrix_usd_row_vector'])[:3, :3]
        axis = unit(local_axis@basis)
        eye = np.asarray(self.sample['camera_position_world_m'])
        matrix = orbit_camera_matrix(eye, eye+.05*axis, self.sample['meters_per_unit'],
                                     [801, 801], 90, [400, 400])
        self.message('Scout view only: looking for a complete visible lesion...')
        record = await self.capture(matrix, ORBIT_HOME/'scouts')
        if not record.get('valid'):
            raise RuntimeError('Scout view invalid: '+str(record.get('invalid_reasons')))
        self.set_sample(record, Path(record['image_path']).parent)
        self.message('Scout saved. Locate a lesion with X/Y/R; Recenter if it is cropped.')

    async def save_region(self, name):
        await self.get_query()
        x, y, radius = self.x.as_int, self.y.as_int, self.radius.as_int
        w, h = self.sample['image_resolution']
        if x-radius < 0 or y-radius < 0 or x+radius > w-1 or y+radius > h-1:
            raise ValueError('Region is cropped. Recenter first, then mark the complete lesion.')
        self.message('Mapping selected region to the verified three-dimensional stomach...')
        center, normal, triangle, distance = await self.surface_pixel([x, y])
        landmarks = []
        for angle in np.linspace(0, 2*math.pi, 12, endpoint=False):
            xy = [float(x+radius*math.cos(angle)), float(y+radius*math.sin(angle))]
            hit = await self.surface_pixel(xy)
            landmarks.append(hit[0].tolist())
        extent = max(float(np.linalg.norm(np.asarray(p)-center)) for p in landmarks)
        if not .0005 <= extent <= .050:
            raise ValueError('Mapped region size is implausible; check X/Y/R and visible surface')
        region = {'region_id': name+'_'+uuid.uuid4().hex[:8], 'class_name': name,
                  'class_id': CLASS_MAP[name], 'abnormal': name != 'normal',
                  'center_world_m': center.tolist(), 'inward_normal_world': normal.tolist(),
                  'center_triangle_index': triangle,
                  'visible_region_enclosing_radius_m': extent,
                  'landmark_points_world_m': landmarks, 'source_pixel_circle': [x, y, radius],
                  'source_sample_path': str(self.sample_dir/'sample.json'),
                  'source_image_sha256': self.sample['image_sha256'],
                  'source_camera_matrix': self.sample['camera_world_matrix_usd_row_vector'],
                  'geometry_sha256': self.sample['geometry']['geometry_sha256'],
                  'meters_per_unit': self.sample['meters_per_unit'],
                  'annotation_basis': 'user confirmed class and circumscribed visible region',
                  'limitation': 'visible extent only; not an exact semantic mask or proof of hidden extent'}
        # Retain every calibration revision; latest per class is only an index.
        orbit_atomic_json(ORBIT_HOME/'region_history'/(region['region_id']+'.json'), region)
        self.targets[name] = region
        orbit_atomic_json(ORBIT_HOME/'targets.json', {'schema': 'gastric_orbit_targets_v1',
                          'class_mapping': CLASS_MAP, 'regions': self.targets})
        self.message('Saved '+name+' region. Next: its 12-candidate orbit pilot.')

    def check_candidate(self, region, candidate):
        if not candidate['framing_ok']:
            return 'marked_region_outside_image_circle'
        eye = np.asarray(candidate['eye_m'])
        if self.query.clearance(eye) < .005:
            return 'insufficient_surface_clearance'
        if not .90 <= abs(self.query.winding(eye)) <= 1.10:
            return 'outside_conservative_cavity'
        target = np.asarray(region['center_world_m'])
        hits = self.query.hits(eye, target-eye)
        if not len(hits) or abs(float(hits[0])-np.linalg.norm(target-eye)) > .0005:
            return 'marked_center_occluded_by_stomach'
        matrix = np.asarray(candidate['matrix'])
        axis = -matrix[2, :3]
        hits = self.query.hits(eye, axis)
        if not len(hits) or not 30 <= hits[0]*1000 <= 60:
            return 'optical_axis_distance_outside_30_60_mm'
        # Every marked rim landmark must remain visible from this viewpoint.
        for point in region['landmark_points_world_m']:
            vector = np.asarray(point)-eye
            values = self.query.hits(eye, vector)
            if not len(values) or abs(values[0]-np.linalg.norm(vector)) > .0008:
                return 'marked_region_landmark_occluded'
        return None

    async def pilot(self, name):
        await self.get_query()
        if name not in self.targets:
            raise RuntimeError('Save the '+name+' region first')
        region = self.targets[name]
        if region['geometry_sha256'] != self.query_hash:
            raise RuntimeError('Region belongs to different geometry; calibrate again')
        if not math.isclose(region['meters_per_unit'], self.sample['meters_per_unit'],
                            rel_tol=0, abs_tol=1e-12):
            raise RuntimeError('Region stage units differ; calibrate again')
        self.stop_requested = False
        batch = 'pilot_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
        destination = ORBIT_HOME/batch
        destination.mkdir(exist_ok=False)
        seed = 20261117
        rng = np.random.default_rng(seed)
        source = Path(__file__).resolve()
        shutil.copyfile(source, destination/'pilot_script.py')
        protocol = {'schema': 'gastric_orbit_pilot_v1', 'batch_id': batch, 'random_seed': seed,
                    'class_mapping': CLASS_MAP, 'region': region,
                    'distance_definition': 'optical-center optical-axis first visible stomach hit',
                    'distance_unit': 'mm', 'accepted_distance_range_mm': [30, 60],
                    'rotation_mode': 'camera positions orbit marked surface point; not optical roll',
                    'image_circle_radius_fraction': .42, 'class_label_status': 'pending_human_review',
                    'training_export': False, 'final_split': None,
                    'source_script_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        orbit_atomic_json(destination/'pilot_config.json', protocol)
        rows = []
        try:
            for index, azimuth in enumerate(range(0, 360, 30)):
                if self.stop_requested:
                    break
                # Plan all 360 degrees; infeasible directions are logged, not
                # silently relabelled or replaced with copies of successful views.
                offset_r = 90*math.sqrt(float(rng.random()))
                offset_a = float(rng.uniform(0, 2*math.pi))
                offset = [offset_r*math.cos(offset_a), offset_r*math.sin(offset_a)]
                candidate = orbit_candidate(region, azimuth, [15, 30, 45][index%3],
                                             [35, 45, 55][(index//3)%3], offset,
                                             self.sample['meters_per_unit'], [801, 801], 90)
                row = {'candidate_index': index, 'capture_group_id': batch,
                       'trajectory_id': None, 'group_type': 'static_orbit_waypoint_set_not_motion_trajectory',
                       'region_id': region['region_id'], 'candidate': candidate,
                       'proposed_class': name, 'final_class': None,
                       'label_status': 'pending_human_review', 'split': None}
                self.message('Orbit '+name+': testing azimuth %d / 360...'%azimuth)
                reason = await asyncio.to_thread(self.check_candidate, region, candidate)
                if reason:
                    row.update({'status': 'geometry_rejected', 'reason': reason})
                else:
                    record = await self.capture(candidate['matrix'], destination/'captures', True)
                    row.update({'status': 'captured_candidate' if record.get('valid') else 'capture_invalid',
                                'sample_path': str(Path(record['configuration']['output_root'])/
                                                   record['capture_id']/'sample.json'),
                                'distance_mm': record.get('distance_mm'),
                                'invalid_reasons': record.get('invalid_reasons')})
                    if record.get('valid'):
                        self.set_sample(record, Path(record['image_path']).parent)
                rows.append(row)
                with (destination/'pilot_manifest.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(json_value(row), ensure_ascii=False, allow_nan=False)+'\n')
                await omni.kit.app.get_app().next_update_async()
        finally:
            accepted = sum(row['status']=='captured_candidate' for row in rows)
            summary = {'batch_id': batch, 'tested': len(rows), 'valid_candidates': accepted,
                       'stopped': self.stop_requested, 'formal_training_samples': 0,
                       'labels_pending_review': True, 'rows': rows}
            orbit_atomic_json(destination/'pilot_summary.json', summary)
            self.message('PILOT DONE: %d / %d valid; labels pending review. %s'%(
                          accepted, len(rows), destination))


def gastric_orbit_setup():
    old = getattr(builtins, ORBIT_REGISTRY, None)
    if old is not None:
        if old.task is not None and not old.task.done():
            print('[ORBIT] Existing operation running; wait or Stop after current frame.')
            return old
        old.window.destroy()
    panel = GastricOrbitPanel()
    setattr(builtins, ORBIT_REGISTRY, panel)
    return panel




# ---------- v7 correction: use actual visible surface, adapt each azimuth -----


def orbit_depth_surface_cloud(sample, depth, ids, pixel_circle, step=6):
    """Back-project Euclidean distance_to_camera; never treat it as axial Z."""
    width, height = sample['image_resolution']
    depth, ids = np.asarray(depth), np.asarray(ids)
    if depth.shape[:2] != (height, width) or ids.shape[:2] != (height, width):
        raise ValueError('Source depth/instance shape does not match image')
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if ids.ndim == 3 and ids.shape[2] == 1:
        ids = ids[..., 0]
    if depth.ndim != 2 or ids.ndim != 2:
        raise ValueError('Expected scalar depth and scalar instance-id buffers')
    x, y, radius = map(float, pixel_circle)
    if not (radius >= 4 and x-radius >= 0 and y-radius >= 0
            and x+radius <= width-1 and y+radius <= height-1):
        raise ValueError('Stored calibration circle is invalid or cropped')
    if not isinstance(step, int) or step < 1:
        raise ValueError('Surface sampling step must be positive')
    grid_x=np.arange(math.ceil(x-radius), math.floor(x+radius)+1, step)
    grid_y=np.arange(math.ceil(y-radius), math.floor(y+radius)+1, step)
    xx,yy=np.meshgrid(grid_x,grid_y)
    inside=(xx-x)**2+(yy-y)**2 <= radius**2
    pixels=np.column_stack([xx[inside],yy[inside]]).astype(int)
    if len(pixels) < 25:
        raise ValueError('Calibration circle has too few surface samples')
    distances=depth[pixels[:,1],pixels[:,0]]
    instance=ids[pixels[:,1],pixels[:,0]]
    target=sample['stomach_mesh_path']
    labels=sample.get('instance_id_labels',{}).get('idToLabels',{})
    allowed=[]
    for key,label in labels.items():
        if isinstance(label,dict):
            label=label.get('primPath',label.get('prim_path'))
        if label == target or (isinstance(label,str) and label.startswith(target+'/')):
            allowed.append(int(key))
    good=(np.isfinite(distances)&(distances > 0)&np.isin(instance,allowed))
    if not good.all():
        raise ValueError('Calibration circle contains invalid/non-stomach source pixels; inspect its extent')
    tan_h=math.tan(math.radians(sample['projection_parameters']['horizontal_fov_deg'])/2)
    tan_v=math.tan(math.radians(sample['projection_parameters']['vertical_fov_deg'])/2)
    local=np.column_stack([(2*(pixels[:,0]+.5)/width-1)*tan_h,
                           (1-2*(pixels[:,1]+.5)/height)*tan_v,
                           -np.ones(len(pixels))])
    local/=np.linalg.norm(local,axis=1)[:,None]
    matrix=np.asarray(sample['camera_world_matrix_usd_row_vector'],float)
    origin=matrix[3,:3]*sample['meters_per_unit']
    cloud=origin+(local@matrix[:3,:3])*distances[:,None]
    return cloud,pixels


def orbit_surface_framing(candidate, cloud, units, resolution, fov_deg,
                          circle_fraction=.42, margin_px=6):
    """Finite sampled-surface framing check. Does not certify hidden extent."""
    xy,z=orbit_project(cloud,candidate['matrix'],units,resolution,fov_deg)
    center=np.array([(resolution[0]-1)/2,(resolution[1]-1)/2])
    radial=np.linalg.norm(xy-center,axis=1)
    good=(np.isfinite(xy).all(axis=1)&(z > .0002)&(z < .3)
          &(radial <= min(resolution)*circle_fraction-margin_px))
    candidate['sphere_bound_radius_px_reference_only']=candidate.get('conservative_radius_px')
    candidate['sphere_bound_used_as_gate']=False
    candidate['framing_ok']=bool(good.all())
    candidate['surface_framing']={
        'method':'projected_source_depth_surface_samples_plus_registered_landmarks',
        'sample_count':len(cloud),'outside_count':int((~good).sum()),
        'circle_radius_px':min(resolution)*circle_fraction,
        'margin_px':margin_px,
        'max_projected_radius_px':float(np.max(radial)) if np.isfinite(radial).all() else None,
        'limitation':'finite visible surface samples; human class/extent review still required'}
    return candidate


def orbit_v7_proposals(region, azimuth, index, units, cloud, rng):
    # The preferred range rotates independently of azimuth. Every direction
    # can retry all three requested distance bands before being declared unavailable.
    bands=[[35.,45.,55.],[45.,55.,35.],[55.,35.,45.]][index%3]
    ranges=bands+[58.,40.,50.]
    cones=[[15.,30.,45.],[30.,45.,15.],[45.,15.,30.]][index%3]
    r=90*math.sqrt(float(rng.random()))
    a=float(rng.uniform(0,2*math.pi))
    jitter=[r*math.cos(a),r*math.sin(a)]
    trial=0
    for distance in ranges:
        for cone in cones:
            for offset in (jitter,[0.,0.]):
                trial+=1
                candidate=orbit_candidate(region,azimuth,cone,distance,offset,units,[801,801],90)
                candidate['proposal_index']=trial
                candidate['image_offset_xy']=list(offset)
                yield orbit_surface_framing(candidate,cloud,units,[801,801],90)


class GastricOrbitPanelV7(GastricOrbitPanel):
    async def load_latest(self):
        # Restore a saved annotation on its original frame instead of making
        # the user locate and re-save the same lesion after a pilot.
        regions=sorted(self.targets.values(),key=lambda r:r.get('source_sample_path',''),reverse=True)
        for region in regions:
            path=Path(region.get('source_sample_path',''))
            if path.is_file():
                record=json.loads(path.read_text(encoding='utf-8'))
                if record.get('valid'):
                    self.set_sample(record,path.parent)
                    x,y,r=region['source_pixel_circle']
                    self.x.set_value(int(x)); self.y.set_value(int(y)); self.radius.set_value(int(r))
                    self.draw_overlay()
                    self.message('V7 restored saved '+region['class_name']+' calibration. Ready for its pilot.')
                    return
        await super().load_latest()

    def load_region_cloud(self,region):
        path=Path(region['source_sample_path'])
        if not path.is_file():
            raise RuntimeError('Original calibration sample is missing: '+str(path))
        sample=json.loads(path.read_text(encoding='utf-8'))
        if not sample.get('valid'):
            raise ValueError('Calibration source frame is invalid')
        if sample['geometry']['geometry_sha256'] != region['geometry_sha256']:
            raise ValueError('Calibration source geometry identity differs')
        if not math.isclose(sample['meters_per_unit'],region['meters_per_unit'],rel_tol=0,abs_tol=1e-12):
            raise ValueError('Calibration source units differ')
        image=path.parent/'image.png'
        expected=region['source_image_sha256']
        if (sample['image_sha256'] != expected or
                hashlib.sha256(image.read_bytes()).hexdigest() != expected):
            raise ValueError('Original calibration image hash differs')
        manifest_path=path.parent/'SHA256SUMS.json'
        if not manifest_path.is_file():
            raise ValueError('Original source checksum manifest is missing')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for name in ['depth_m.npy','instance_ids.npy']:
            if name not in manifest:
                raise ValueError('Source checksum missing for '+name)
            actual=hashlib.sha256((path.parent/name).read_bytes()).hexdigest()
            if actual != manifest[name]['sha256']:
                raise ValueError('Original source buffer hash differs: '+name)
        depth=np.load(path.parent/'depth_m.npy',allow_pickle=False)
        ids=np.load(path.parent/'instance_ids.npy',allow_pickle=False)
        cloud,pixels=orbit_depth_surface_cloud(sample,depth,ids,region['source_pixel_circle'])
        landmarks=np.asarray(region['landmark_points_world_m'],float)
        cloud=np.vstack([cloud,landmarks,np.array(region['center_world_m'])[None,:]])
        # Add a few interior samples to the existing twelve rim occlusion checks.
        center_xy=np.array(region['source_pixel_circle'][:2])
        visibility=landmarks.tolist()
        for angle in np.linspace(0,2*math.pi,8,endpoint=False):
            wanted=center_xy+.5*region['source_pixel_circle'][2]*np.array([math.cos(angle),math.sin(angle)])
            nearest=int(np.argmin(np.linalg.norm(pixels-wanted,axis=1)))
            visibility.append(cloud[nearest].tolist())
        return cloud,visibility,{'source_sample_path':str(path),'source_image_sha256':expected,
                                'depth_sha256':manifest['depth_m.npy']['sha256'],
                                'instance_ids_sha256':manifest['instance_ids.npy']['sha256'],
                                'source_pixel_stride':6,'cloud_count':len(cloud),
                                'cloud_sha256':hashlib.sha256(cloud.tobytes()).hexdigest()}

    def check_candidate(self,region,candidate):
        if not candidate['framing_ok']:
            return 'marked_region_outside_image_circle'
        diag={}
        candidate['geometry_checks']=diag
        eye=np.asarray(candidate['eye_m'])
        clearance=self.query.clearance(eye)
        diag['clearance_mm']=clearance*1000
        if clearance < .005:
            return 'insufficient_surface_clearance'
        winding=self.query.winding(eye)
        diag['winding']=winding
        if not .90 <= abs(winding) <= 1.10:
            return 'outside_conservative_cavity'
        target=np.asarray(region['center_world_m'])
        hits=self.query.hits(eye,target-eye)
        if not len(hits) or abs(float(hits[0])-np.linalg.norm(target-eye)) > .0005:
            return 'marked_center_occluded_by_stomach'
        matrix=np.asarray(candidate['matrix'])
        hits=self.query.hits(eye,-matrix[2,:3])
        diag['optical_axis_distance_mm']=float(hits[0]*1000) if len(hits) else None
        if not len(hits) or not 30 <= hits[0]*1000 <= 60:
            return 'optical_axis_distance_outside_30_60_mm'
        for point in region['landmark_points_world_m']:
            vector=np.asarray(point)-eye
            values=self.query.hits(eye,vector)
            if not len(values) or abs(values[0]-np.linalg.norm(vector)) > .0008:
                return 'marked_region_landmark_occluded'
        return None

    async def pilot(self,name):
        await self.get_query()
        if name not in self.targets:
            raise RuntimeError('Save the '+name+' region first')
        region=json.loads(json.dumps(self.targets[name]))
        if region['geometry_sha256'] != self.query_hash:
            raise RuntimeError('Region belongs to different geometry; calibrate again')
        if not math.isclose(region['meters_per_unit'],self.sample['meters_per_unit'],rel_tol=0,abs_tol=1e-12):
            raise RuntimeError('Region stage units differ')
        self.message('V7: verifying saved calibration and reconstructing source surface...')
        cloud,visibility,cloud_info=await asyncio.to_thread(self.load_region_cloud,region)
        region['landmark_points_world_m']=visibility
        self.stop_requested=False
        batch='pilot_v7_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
        destination=ORBIT_HOME/batch
        destination.mkdir(exist_ok=False)
        seed=20261117
        rng=np.random.default_rng(seed)
        source=Path(__file__).resolve()
        shutil.copyfile(source,destination/'pilot_script.py')
        np.save(destination/'source_surface_cloud_m.npy',cloud,allow_pickle=False)
        protocol={'schema':'gastric_orbit_pilot_v2','batch_id':batch,'random_seed':seed,
                  'region':region,'class_mapping':CLASS_MAP,'source_surface':cloud_info,
                  'distance_definition':'optical-center optical-axis first visible stomach hit',
                  'distance_unit':'mm','accepted_distance_range_mm':[30,60],
                  'minimum_clearance_mm':5,'image_circle_radius_fraction':.42,
                  'rotation_mode':'physical camera positions around registered region, not optical roll',
                  'selection_policy':'first verified candidate per azimuth; multiple ranges/cones/offsets',
                  'maximum_proposals_per_azimuth':36,'azimuths_deg':list(range(0,360,30)),
                  'class_label_status':'pending_human_review','training_export':False,'final_split':None,
                  'source_script_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        orbit_atomic_json(destination/'pilot_config.json',protocol)
        rows=[]
        problem=None
        try:
            for index,azimuth in enumerate(range(0,360,30)):
                if self.stop_requested:
                    break
                row={'candidate_index':index,'azimuth_deg':azimuth,'capture_group_id':batch,
                     'trajectory_id':None,'group_type':'static_orbit_waypoint_set_not_motion_trajectory',
                     'region_id':region['region_id'],'proposed_class':name,'final_class':None,
                     'label_status':'pending_human_review','split':None,'attempts':[],
                     'status':'no_verified_candidate_at_azimuth'}
                rows.append(row)
                for candidate in orbit_v7_proposals(region,azimuth,index,self.sample['meters_per_unit'],cloud,rng):
                    if self.stop_requested:
                        break
                    self.message('V7 '+name+': azimuth %d / 360, proposal %d / 36...'%(
                                 azimuth,candidate['proposal_index']))
                    reason=await asyncio.to_thread(self.check_candidate,region,candidate)
                    attempt={'candidate':candidate,'reason':reason,'status':'geometry_rejected' if reason else 'geometry_verified'}
                    row['attempts'].append(attempt)
                    if not reason:
                        record=await self.capture(candidate['matrix'],destination/'captures',True)
                        sample_path=str(Path(record['configuration']['output_root'])/record['capture_id']/'sample.json')
                        attempt.update({'status':'captured_candidate' if record.get('valid') else 'capture_invalid',
                                        'sample_path':sample_path,'invalid_reasons':record.get('invalid_reasons')})
                        row.update({'status':attempt['status'],'candidate':candidate,'sample_path':sample_path,
                                    'distance_mm':record.get('distance_mm'),'invalid_reasons':record.get('invalid_reasons')})
                        if record.get('valid'):
                            self.set_sample(record,Path(record['image_path']).parent)
                        # Rendering failures may reflect a global state change;
                        # stop rather than silently retrying and hiding that failure.
                        if record.get('status') in ('failed','cancelled'):
                            raise RuntimeError('Capture engine stopped: '+str(record.get('invalid_reasons')))
                        if record.get('valid'):
                            break
                    await omni.kit.app.get_app().next_update_async()
                with (destination/'pilot_manifest.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(json_value(row),ensure_ascii=False,allow_nan=False)+'\n')
                orbit_atomic_json(destination/'progress.json',{'completed_directions':len(rows),'rows':rows})
        except Exception as exc:
            problem=str(exc)
            raise
        finally:
            accepted=sum(r['status']=='captured_candidate' for r in rows)
            counts={}
            for row in rows:
                for attempt in row['attempts']:
                    if attempt['reason']:
                        counts[attempt['reason']]=counts.get(attempt['reason'],0)+1
            summary={'batch_id':batch,'tested':len(rows),'planned_directions':12,
                     'valid_candidates':accepted,'stopped':self.stop_requested,'error':problem,
                     'formal_training_samples':0,'labels_pending_review':True,
                     'rejection_counts_over_all_proposals':counts,'source_surface':cloud_info,'rows':rows}
            orbit_atomic_json(destination/'pilot_summary.json',summary)
            state='PILOT STOPPED' if self.stop_requested or problem else 'PILOT DONE'
            self.message('%s: %d / %d directions valid; labels pending review. %s'%(
                         state,accepted,len(rows),destination))


GastricOrbitPanel=GastricOrbitPanelV7




# ---------------- v8: usable photos, count quota, relaxed circle framing ------


def orbit_photo_framing(candidate,cloud,units,resolution,fov_deg,
                        circle_fraction=.42,margin_px=6):
    # circle_fraction is retained only for call compatibility. User asks for
    # a visible central lesion, not a whole marked area inside a circular mask.
    width,height=resolution
    target_xy=np.asarray(candidate['target_pixel_xy'],float)
    center_safe=bool(np.isfinite(target_xy).all() and
                     .15*width <= target_xy[0] <= .85*width and
                     .15*height <= target_xy[1] <= .85*height)
    xy,z=orbit_project(cloud,candidate['matrix'],units,resolution,fov_deg)
    crop_margin=12.
    in_frame=(np.isfinite(xy).all(axis=1)&(z > .0002)&(z < .3)&
              (xy[:,0] >= crop_margin)&(xy[:,0] <= width-1-crop_margin)&
              (xy[:,1] >= crop_margin)&(xy[:,1] <= height-1-crop_margin))
    candidate['sphere_bound_used_as_gate']=False
    candidate['framing_ok']=bool(center_safe and in_frame.all())
    candidate['surface_framing']={
        'method':'marked_center_15_percent_border_margin_and_ROI_inside_raw_frame',
        'whole_ROI_inside_circle_required':False,'sample_count':len(cloud),
        'outside_count':int((~in_frame).sum()),'center_safe':center_safe,
        'max_projected_radius_px':float(np.max(np.linalg.norm(
            xy-[(width-1)/2,(height-1)/2],axis=1))) if np.isfinite(xy).all() else None,
        'source_surface_crop_margin_px':crop_margin,
        'limitation':'seed annotation extent is approximate; class and visibility review pending'}
    return candidate


def orbit_photo_fingerprint(path):
    from PIL import Image
    with Image.open(path) as img:
        image=np.asarray(img.convert('L').resize((17,16)),dtype=np.int16)
    bits=(image[:,1:] > image[:,:-1]).reshape(-1)
    return int.from_bytes(np.packbits(bits).tobytes(),'big')


def orbit_pose_near_duplicate(matrix,accepted,units):
    matrix=np.asarray(matrix,float)
    for previous in accepted:
        previous=np.asarray(previous,float)
        distance=np.linalg.norm(matrix[3,:3]-previous[3,:3])*units
        angle=math.degrees(math.acos(float(np.clip((np.trace(
            matrix[:3,:3]@previous[:3,:3].T)-1)/2,-1,1))))
        if distance < .002 and angle < 5:
            return True
    return False


class GastricOrbitPanelV8(GastricOrbitPanelV7):
    async def load_latest(self):
        await super().load_latest()
        if self.sample is not None:
            self.message('V8 ready. Existing calibration loaded. Use 5 Collect: 50 images for the saved class.')

    def check_candidate(self,region,candidate):
        # Retain cavity, clearance, true optical distance and marked-center
        # occlusion checks. The perimeter of a drawn circle includes nearby
        # normal wall, whose visibility is not a lesion-class requirement.
        center_only=dict(region,landmark_points_world_m=[])
        reason=super().check_candidate(center_only,candidate)
        if reason:
            return reason
        xy,z=orbit_project([region['center_world_m']],candidate['matrix'],
                           self.sample['meters_per_unit'],[801,801],90)
        if z[0] <= .0002 or not np.isfinite(xy).all() or not (
                .15*801 <= xy[0,0] <= .85*801 and .15*801 <= xy[0,1] <= .85*801):
            return 'marked_center_too_close_to_image_edge'
        candidate['geometry_checks']['marked_center_projected_xy']=xy[0].tolist()
        return None

    def target_render_check(self,record,region):
        xy,z=orbit_project([region['center_world_m']],record['camera_world_matrix_usd_row_vector'],
                           record['meters_per_unit'],record['image_resolution'],
                           record['projection_parameters']['horizontal_fov_deg'])
        width,height=record['image_resolution']
        x,y=map(lambda value:int(round(value)),xy[0])
        result={'scope':'marked lesion-center visibility, not a semantic segmentation',
                'pixel_xy':[x,y],'passed':False,'reason':None}
        if z[0] <= 0 or not (0 <= x < width and 0 <= y < height):
            result['reason']='marked_center_outside_image'
            return result
        root=Path(record['image_path']).parent
        depth=np.load(root/'depth_m.npy',mmap_mode='r',allow_pickle=False)
        ids=np.load(root/'instance_ids.npy',mmap_mode='r',allow_pickle=False)
        rendered=float(np.asarray(depth[y,x]).reshape(-1)[0])
        identifier=int(np.asarray(ids[y,x]).reshape(-1)[0])
        labels=record.get('instance_id_labels',{}).get('idToLabels',{})
        label=labels.get(str(identifier),labels.get(identifier))
        if isinstance(label,dict):
            label=label.get('primPath',label.get('prim_path'))
        target=record['stomach_mesh_path']
        expected=float(np.linalg.norm(np.asarray(region['center_world_m'])-
                                      np.asarray(record['camera_position_world_m'])))
        # This is off-axis pixel-rounding visibility tolerance. It never
        # changes the optical-axis distance label or its 30-60 mm range.
        tolerance=max(.001,.005*expected)
        result.update({'rendered_range_mm':rendered*1000 if math.isfinite(rendered) else None,
                       'expected_marked_center_range_mm':expected*1000,
                       'visibility_tolerance_mm':tolerance*1000,'instance_path':label})
        same=(label == target or isinstance(label,str) and label.startswith(target+'/'))
        result['passed']=bool(same and math.isfinite(rendered) and rendered > 0
                              and abs(rendered-expected) <= tolerance)
        if not result['passed']:
            result['reason']='marked_target_occluded_or_render_depth_disagrees'
        return result

    async def collect(self,name,count=50):
        if name not in ('raised','ulcerated') or count != 50:
            raise ValueError('This entry collects exactly 50 images for one confirmed lesion class')
        await self.get_query()
        if name not in self.targets:
            raise RuntimeError('Calibrate '+name+' with Save first')
        region=json.loads(json.dumps(self.targets[name]))
        if region['geometry_sha256'] != self.query_hash:
            raise RuntimeError('Current geometry differs from the saved region')
        if not math.isclose(region['meters_per_unit'],self.sample['meters_per_unit'],rel_tol=0,abs_tol=1e-12):
            raise RuntimeError('Region units differ')
        self.message('Preparing '+name+' collection: target 50 distinct usable photos...')
        cloud,_,cloud_info=await asyncio.to_thread(self.load_region_cloud,region)
        self.stop_requested=False
        batch='collect_small_'+name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
        root=ORBIT_HOME/batch
        (root/'images').mkdir(parents=True,exist_ok=False)
        seed=20261117
        rng=np.random.default_rng(seed)
        script=Path(__file__).resolve()
        shutil.copyfile(script,root/'collection_script.py')
        np.save(root/'source_surface_cloud_m.npy',cloud,allow_pickle=False)
        policy={'schema':'gastric_lesion_photo_collection_v1','batch_id':batch,
                'class_name':name,'class_id':CLASS_MAP[name],'target_count':count,
                'region':region,'source_surface':cloud_info,'random_seed':seed,
                'distance_unit':'mm','accepted_distance_range_mm':[30,60],
                'distance_definition':'actual optical-axis first visible stomach hit',
                'minimum_clearance_mm':5,'marked_center_image_margin_fraction':.15, 'view_change_range_deg':[5,10], 'angular_reference':'registered seed sightline',
                'whole_ROI_inside_circle_required':False,
                'near_duplicate_pose_limits':{'translation_mm':2,'angle_deg':5},
                'near_duplicate_image_hash':'256-bit difference hash','minimum_hash_hamming':4,
                'maximum_geometry_proposals':2400,'maximum_rendered_candidates':250,
                'label_basis':'user-confirmed seed region; output views pending review',
                'training_split_exported':False,
                'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
        orbit_atomic_json(root/'collection_config.json',policy)
        rows=[]; rejections={}; poses=[]; hashes=[]
        proposals=0; rendered_count=0; error=None
        coarse=list(range(0,360,30))
        azimuths=coarse+[a for a in range(0,360,5) if a not in coarse]
        try:
            for sweep in range(3):
                if len(rows) >= count or self.stop_requested:
                    break
                for index,azimuth in enumerate(azimuths):
                    if len(rows) >= count or self.stop_requested or proposals >= 2400 or rendered_count >= 250:
                        break
                    for candidate in orbit_v7_proposals(region,azimuth,index+sweep,
                            self.sample['meters_per_unit'],cloud,rng):
                        if len(rows) >= count or self.stop_requested or proposals >= 2400 or rendered_count >= 250:
                            break
                        proposals+=1
                        self.message('SMALL VIEW %s: %d / 50 saved; set %d; pose %d...'%(
                                     name,len(rows),azimuth,proposals))
                        reason=await asyncio.to_thread(self.check_candidate,region,candidate)
                        if not reason and orbit_pose_near_duplicate(candidate['matrix'],poses,
                                                                    self.sample['meters_per_unit']):
                            reason='near_duplicate_pose'
                        attempt={'proposal_index':proposals,'proposal_phase_deg':azimuth,'sweep':sweep,
                                 'candidate':candidate,'reason':reason}
                        if not reason:
                            record=await self.capture(candidate['matrix'],root/'captures',True)
                            rendered_count+=1
                            sample_path=str(Path(record['configuration']['output_root'])/record['capture_id']/'sample.json')
                            attempt['sample_path']=sample_path
                            if not record.get('valid'):
                                reason='capture_invalid'
                                attempt['invalid_reasons']=record.get('invalid_reasons')
                                if record.get('status') in ('failed','cancelled'):
                                    raise RuntimeError('Capture engine stopped: '+str(record.get('invalid_reasons')))
                            else:
                                visibility=await asyncio.to_thread(self.target_render_check,record,region)
                                attempt['marked_target_render_check']=visibility
                                if not visibility['passed']:
                                    reason=visibility['reason']
                                else:
                                    fingerprint=await asyncio.to_thread(orbit_photo_fingerprint,record['image_path'])
                                    nearest=min(((fingerprint^h).bit_count() for h in hashes),default=256)
                                    attempt['nearest_photo_hash_hamming']=nearest
                                    if nearest < 4:
                                        reason='near_duplicate_image'
                                    else:
                                        image_name='%s_%04d.png'%(name,len(rows)+1)
                                        shutil.copyfile(record['image_path'],root/'images'/image_name)
                                        row={'sample_id':record['sample_id'],'image':'images/'+image_name,
                                             'image_sha256':record['image_sha256'],'sample_path':sample_path,
                                             'proposed_class':name,'proposed_class_id':CLASS_MAP[name],
                                             'abnormal':True,'label_status':'pending_view_review',
                                             'region_id':region['region_id'],'scene_id':record['scene_id'],
                                             'source_collection_id':batch,'capture_group_id':region['region_id']+'_small_reference',
                                             'grouping_note':'all nearby views of this reference stay in one group; audit before split',
                                             'trajectory_id':None,'split':None,'view_offset_deg':candidate['view_offset_deg'],'camera_view_change_deg':candidate['camera_view_change_deg'],
                                             'camera_position_world_m':record['camera_position_world_m'],
                                             'camera_orientation_world':record['camera_orientation_world'],
                                             'quaternion_order':'wxyz','distance_mm':record['distance_mm'],
                                             'distance_valid':True,'marked_target_render_check':visibility,
                                             'photo_dhash256':'%064x'%fingerprint,
                                             'source_script_sha256':record['script_sha256']}
                                        rows.append(row); hashes.append(fingerprint); poses.append(candidate['matrix'])
                                        with (root/'classification_manifest.jsonl').open('a',encoding='utf-8') as stream:
                                            stream.write(json.dumps(json_value(row),ensure_ascii=False,allow_nan=False)+'\n')
                                        self.set_sample(record,Path(record['image_path']).parent)
                                        self.message('COLLECT %s: %d / 50 distinct usable photos saved.'%(name,len(rows)))
                        attempt['reason']=reason
                        if reason:
                            rejections[reason]=rejections.get(reason,0)+1
                        with (root/'all_attempts.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(json_value(attempt),ensure_ascii=False,allow_nan=False)+'\n')
                        orbit_atomic_json(root/'collection_progress.json',{'target_count':count,'saved_images':len(rows),
                                          'geometry_proposals':proposals,'rendered_candidates':rendered_count,
                                          'rejection_counts':rejections,'images':rows})
                        await omni.kit.app.get_app().next_update_async()
                        if not reason:
                            break
        except Exception as exc:
            error=str(exc)
            raise
        finally:
            distances=[r['distance_mm'] for r in rows]
            summary={'batch_id':batch,'class_name':name,'target_count':count,'saved_images':len(rows),
                     'quota_met':len(rows)==count and error is None,'shortfall':max(0,count-len(rows)),
                     'stopped':self.stop_requested,'error':error,'geometry_proposals':proposals,
                     'rendered_candidates':rendered_count,'rejection_counts':rejections,
                     'distance_min_mm':min(distances) if distances else None,
                     'distance_max_mm':max(distances) if distances else None,
                     'distance_bins_30_40_50_60':[sum(lo <= d < hi or hi==60 and d==60 for d in distances)
                                                 for lo,hi in [(30,40),(40,50),(50,60)]],
                     'labels_pending_review':True,'final_training_samples':0,
                     'training_split_exported':False,'images':rows}
            orbit_atomic_json(root/'collection_summary.json',summary)
            label='COLLECTION DONE' if len(rows)==count and error is None else 'COLLECTION INCOMPLETE'
            self.message('%s: %s %d / 50 photos saved; pending view review. %s'%(
                         label,name,len(rows),root))


# V7's proposal generator resolves this function at runtime; retain all of its
# multi-range and actual-position sampling while adopting the user's photo criterion.
orbit_surface_framing=orbit_photo_framing
GastricOrbitPanel=GastricOrbitPanelV8




def small_view_proposals(region, phase, index, units, cloud, rng):
    source = json.loads(Path(region['source_sample_path']).read_text(encoding='utf-8'))
    target = np.asarray(region['center_world_m'], float)
    eye0 = np.asarray(source['camera_position_world_m'], float)
    ward = unit(eye0-target)
    right = np.asarray(source['camera_world_matrix_usd_row_vector'], float)[0,:3]
    tangent = unit(right-ward*float(right@ward))
    bitangent = unit(np.cross(ward,tangent))
    alpha = math.radians((phase+index*137.507764)%360)
    radial = math.cos(alpha)*tangent+math.sin(alpha)*bitangent
    angles = [[5.5,7.5,9.5],[7.5,9.5,5.5],[9.5,5.5,7.5]][index%3]
    ranges = [[45,50,55],[50,55,45],[55,45,50]][index%3]+[40,58,35,60,30]
    number = 0
    for angle in angles:
        for distance in ranges:
            number += 1
            theta = math.radians(angle)
            eye = target+distance*.001*(math.cos(theta)*ward+math.sin(theta)*radial)
            pixel = (np.array([400.,400.])+rng.uniform(-6,6,2)).tolist()
            matrix = orbit_camera_matrix(eye,target,units,[801,801],90,pixel)
            axis_angle = math.degrees(math.acos(float(np.clip((-matrix[2,:3])@(-ward),-1,1))))
            if not 5 <= axis_angle <= 10:
                continue
            candidate = {'eye_m':eye.tolist(),'matrix':matrix.tolist(),
                'proposal_index':number,'target_pixel_xy':pixel,
                'camera_to_marked_center_mm':distance,'view_offset_deg':angle,
                'camera_view_change_deg':axis_angle,'local_phase_deg':math.degrees(alpha)%360,
                'reference_sample_path':region['source_sample_path'],
                'rotation_mode':'local 5-10 degrees from registered reference, not full orbit'}
            yield orbit_photo_framing(candidate,cloud,units,[801,801],90)

class GastricSmallViews(GastricOrbitPanelV8):
    async def pilot(self,name):
        await self.collect(name,50)

orbit_v7_proposals = small_view_proposals
GastricOrbitPanel = GastricSmallViews
_small_panel = gastric_orbit_setup()
async def _small_start():
    loading = _small_panel.task
    if loading is not None:
        await loading
    if _small_panel.sample is None or _small_panel.stop_requested:
        print('[SMALL] Load was stopped or failed; collection not started.')
        return
    _small_panel.start(lambda: _small_panel.collect('raised',50))
asyncio.ensure_future(_small_start())
