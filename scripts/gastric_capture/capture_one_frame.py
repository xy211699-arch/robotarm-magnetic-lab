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
        "script_revision": "v5-static-isolated-preview",
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
            import omni.ui as ui
            window = ui.Window("Gastric | Exported Frame", width=700, height=740)
            state["window"] = window
            with window.frame:
                with ui.VStack():
                    text = ("VALID | distance %.6f mm" % record["distance_mm"]
                            if record["valid"] else "INVALID - inspect sample.json")
                    ui.Label(text, height=24)
                    ui.Image(str(image_path), width=680, height=680)
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


gastric_capture_run()

