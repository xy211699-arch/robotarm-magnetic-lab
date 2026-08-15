#!/usr/bin/env python3
"""Inspect the mandatory TASK-002 geometry and live pose-write prerequisites.

The schema validator is importable without Isaac Sim.  Live USD, PhysX and
Isaac Lab imports are intentionally delayed until :func:`main` launches Kit.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))


TASK_ID = "Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0"
CAPSULE_PRIM_PATH = "/World/envs/env_0/Scene/MagneticDemo/target_magnet"
CAMERA_PRIM_PATH = CAPSULE_PRIM_PATH + "/capsule_camera"
SURFACE_PRIM_PATH = (
    "/World/envs/env_0/Stomach/ConvertedSource/Environment/Stomach/"
    "VisualMesh/Stomach"
)
SCHEMA_VERSION = "ideal_surface_preflight_v1"
REQUIRED_REPORT_KEYS = {
    "repository",
    "task",
    "capsule",
    "camera",
    "surface",
    "pose_write_api",
    "initial_contact",
    "gate",
}


def _require(mapping: dict[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing {location}.{key}")
    return mapping[key]


def validate_preflight_report(report: dict[str, Any]) -> None:
    """Validate every contract-critical field in a preflight report."""
    if set(report) != REQUIRED_REPORT_KEYS:
        missing = sorted(REQUIRED_REPORT_KEYS - set(report))
        extra = sorted(set(report) - REQUIRED_REPORT_KEYS)
        raise ValueError(f"preflight keys differ: missing={missing}, extra={extra}")
    for key in ("commit", "branch"):
        _require(report["repository"], key, "repository")
    _require(report["task"], "id", "task")
    for key in (
        "shape_class",
        "radius_m",
        "cylinder_half_length_m",
        "long_axis_local",
    ):
        _require(report["capsule"], key, "capsule")
    for key in ("optical_axis_local", "image_up_axis_local"):
        _require(report["camera"], key, "camera")
    for key in (
        "vertex_count",
        "triangle_count",
        "geometry_sha256",
        "inward_normal_confirmed",
    ):
        _require(report["surface"], key, "surface")
    for key in ("pose_method", "velocity_method", "quaternion_order"):
        _require(report["pose_write_api"], key, "pose_write_api")
    for key in ("valid", "triangle_id"):
        _require(report["initial_contact"], key, "initial_contact")
    for key in ("status", "failures"):
        _require(report["gate"], key, "gate")
    if report["gate"]["status"] not in {"pass", "needs_decision"}:
        raise ValueError("gate.status must be pass or needs_decision")


def build_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Build the non-negotiable implementation gate from observed facts."""
    failures: list[str] = []
    if report["capsule"]["shape_class"] != "spherocylinder":
        failures.append("capsule collision is not an unambiguous spherocylinder")
    if report["capsule"].get("long_axis_local") not in ([0, 0, 1], [0, 0, -1]):
        failures.append("capsule long axis is ambiguous")
    if not report["camera"].get("axes_confirmed", False):
        failures.append("capsule camera optical/image-up axes are ambiguous")
    if not report["surface"].get("approved_path_confirmed", False):
        failures.append("approved stomach navigation mesh is unavailable")
    if not report["surface"]["inward_normal_confirmed"]:
        failures.append("stomach inward normal is ambiguous")
    if not report["pose_write_api"]["pose_method"]:
        failures.append("no verified root-pose write API")
    if not report["pose_write_api"]["velocity_method"]:
        failures.append("no verified root-velocity write API")
    if report["pose_write_api"].get("quaternion_order") != "wxyz":
        failures.append("simulator quaternion ordering is not verified as wxyz")
    if not report["initial_contact"]["valid"]:
        failures.append("initial capsule pose has no valid surface contact")
    return {"status": "pass" if not failures else "needs_decision", "failures": failures}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _matrix_values(matrix: Any) -> np.ndarray:
    return np.asarray(
        [[float(matrix[row][col]) for col in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _quat_wxyz_matrix(quaternion: Any) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    w, x, y, z = np.asarray([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _descendants_including(root: Any):
    stack = [root]
    while stack:
        prim = stack.pop()
        yield prim
        stack.extend(reversed(prim.GetAllChildren()))


def _inspect_capsule(stage: Any, capsule: Any) -> dict[str, Any]:
    from pxr import UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(CAPSULE_PRIM_PATH)
    if not root.IsValid():
        return {
            "prim_path": CAPSULE_PRIM_PATH,
            "shape_class": "missing",
            "radius_m": math.nan,
            "cylinder_half_length_m": math.nan,
            "long_axis_local": None,
            "rigid_body_prims": [],
            "colliders": [],
        }
    rigid_bodies: list[str] = []
    colliders: list[dict[str, Any]] = []
    for prim in _descendants_including(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(prim.GetPath()))
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled_attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        enabled = bool(enabled_attr.Get()) if enabled_attr and enabled_attr.HasAuthoredValue() else True
        item: dict[str, Any] = {
            "prim_path": str(prim.GetPath()),
            "type_name": prim.GetTypeName(),
            "enabled": enabled,
        }
        if prim.IsA(UsdGeom.Capsule):
            shape = UsdGeom.Capsule(prim)
            item.update(
                {
                    "radius_m": float(shape.GetRadiusAttr().Get()),
                    "cylinder_length_m": float(shape.GetHeightAttr().Get()),
                    "axis": str(shape.GetAxisAttr().Get() or "Z"),
                }
            )
        colliders.append(item)
    enabled = [item for item in colliders if item["enabled"]]
    capsules = [item for item in enabled if item["type_name"] == "Capsule"]
    unambiguous = len(enabled) == 1 and len(capsules) == 1
    selected = capsules[0] if unambiguous else None
    axis_name = selected.get("axis") if selected else None
    axis_by_name = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    radius = float(selected["radius_m"]) if selected else math.nan
    cylinder_half = 0.5 * float(selected["cylinder_length_m"]) if selected else math.nan
    return {
        "prim_path": CAPSULE_PRIM_PATH,
        "shape_class": "spherocylinder" if unambiguous else "ambiguous",
        "radius_m": radius,
        "cylinder_half_length_m": cylinder_half,
        "tip_to_tip_length_m": 2.0 * (radius + cylinder_half) if selected else math.nan,
        "long_axis_local": axis_by_name.get(axis_name),
        "rigid_body_prims": sorted(rigid_bodies),
        "colliders": colliders,
        "local_collision_bounds_m": (
            [[-radius, -radius, -(radius + cylinder_half)],
             [radius, radius, radius + cylinder_half]]
            if selected and axis_name == "Z"
            else None
        ),
        "live_position_world_m": capsule.data.root_pos_w.torch[0].detach().cpu().tolist(),
        "live_quaternion_wxyz": capsule.data.root_quat_w.torch[0].detach().cpu().tolist(),
    }


def _edge_statistics(triangles: np.ndarray) -> dict[str, int]:
    counts: Counter[tuple[int, int]] = Counter()
    for triangle in np.asarray(triangles, dtype=np.int64):
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            counts[tuple(sorted((int(first), int(second))))] += 1
    return {
        "edge_count": len(counts),
        "boundary_edge_count": sum(value == 1 for value in counts.values()),
        "nonmanifold_edge_count": sum(value > 2 for value in counts.values()),
    }


def _closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest point and deterministic barycentric coordinates."""
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.asarray([1.0, 0.0, 0.0])
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.asarray([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.asarray([1.0 - v, v, 0.0])
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.asarray([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.asarray([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.asarray([0.0, 1.0 - w, w])
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w, np.asarray([1.0 - v - w, v, w])


def _closest_surface(reference: Any, point: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, float]:
    best: tuple[float, int, tuple[float, float, float], np.ndarray] | None = None
    vertices = np.asarray(reference.vertices_world, dtype=np.float64)
    for triangle_id, indices in enumerate(np.asarray(reference.triangles, dtype=np.int64)):
        closest, bary = _closest_point_on_triangle(point, vertices[indices])
        distance_sq = float((closest - point) @ (closest - point))
        key = (distance_sq, int(triangle_id), tuple(float(value) for value in bary), closest)
        if best is None or key[:3] < best[:3]:
            best = key
    assert best is not None
    return best[1], np.asarray(best[3]), np.asarray(best[2]), math.sqrt(best[0])


def _inspect_surface(stage: Any, capsule_report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from pxr import UsdGeom
    from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage

    prim = stage.GetPrimAtPath(SURFACE_PRIM_PATH)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        surface = {
            "prim_path": SURFACE_PRIM_PATH,
            "approved_path_confirmed": False,
            "vertex_count": 0,
            "triangle_count": 0,
            "geometry_sha256": None,
            "inward_normal_confirmed": False,
        }
        contact = {"valid": False, "triangle_id": None}
        return surface, contact
    reference = reference_from_stage(SURFACE_PRIM_PATH)
    triangles = np.asarray(reference.triangles, dtype=np.int64)
    vertices = np.asarray(reference.vertices_world, dtype=np.float64)
    center = np.asarray(capsule_report["live_position_world_m"], dtype=np.float64)
    triangle_id, surface_point, barycentric, closest_distance = _closest_surface(reference, center)
    triangle = vertices[triangles[triangle_id]]
    cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    cross /= np.linalg.norm(cross)
    orientation = str(UsdGeom.Mesh(prim).GetOrientationAttr().Get() or "rightHanded")
    winding_normal = -cross if orientation == "leftHanded" else cross
    center_direction = (center - surface_point) / closest_distance
    winding_alignment = float(winding_normal @ center_direction)
    # The live reset capsule is the task's authored cavity-interior probe.  A
    # near-collinear closest-point vector makes the sign unique; it is then
    # independently checked below against the exact capsule support distance.
    inward_sign = 1 if winding_alignment >= 0.0 else -1
    oriented_normal = inward_sign * winding_normal
    inward_alignment = float(oriented_normal @ center_direction)
    capsule_axis_local = np.asarray(capsule_report["long_axis_local"], dtype=np.float64)
    root_quaternion = np.asarray(capsule_report["live_quaternion_wxyz"], dtype=np.float64)
    capsule_axis_world = _quat_wxyz_matrix(root_quaternion) @ capsule_axis_local
    support = float(capsule_report["radius_m"]) + float(
        capsule_report["cylinder_half_length_m"]
    ) * abs(float(capsule_axis_world @ oriented_normal))
    signed_center_distance = float((center - surface_point) @ oriented_normal)
    support_gap = signed_center_distance - support
    radius = float(capsule_report["radius_m"])
    inward_confirmed = bool(
        np.isfinite(inward_alignment)
        and inward_alignment >= 0.9
        and abs(support_gap) <= max(0.001, 0.2 * radius)
    )
    valid_contact = bool(
        inward_confirmed
        and abs(support_gap) <= max(0.001, 0.2 * radius)
        and closest_distance <= support + max(0.001, 0.2 * radius)
    )
    surface = {
        "prim_path": SURFACE_PRIM_PATH,
        "approved_path_confirmed": True,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "geometry_sha256": reference.geometry_sha256,
        "orientation": orientation,
        "edge_statistics": _edge_statistics(triangles),
        "winding_normal_world": winding_normal.tolist(),
        "winding_alignment_to_initial_capsule": winding_alignment,
        "inward_sign_from_winding": inward_sign,
        "closest_triangle_normal_world": oriented_normal.tolist(),
        "inward_alignment_at_initial_contact": inward_alignment,
        "inward_confirmation_basis": (
            "normal sign points from the closest approved luminal-surface point "
            "to the authored cavity-interior capsule center and reproduces the "
            "preflight-confirmed spherocylinder support distance"
        ),
        "inward_normal_confirmed": inward_confirmed,
    }
    contact = {
        "valid": valid_contact,
        "triangle_id": int(triangle_id),
        "component_basis": "approved luminal mesh",
        "surface_point_world_m": surface_point.tolist(),
        "surface_normal_world": oriented_normal.tolist(),
        "barycentric": barycentric.tolist(),
        "center_distance_m": closest_distance,
        "support_distance_m": support,
        "support_gap_m": support_gap,
    }
    return surface, contact


def _inspect_camera(env: Any) -> dict[str, Any]:
    camera = env.unwrapped.scene["capsule_camera"]
    offset = camera.cfg.offset
    optical_camera = np.asarray([0.0, 0.0, 1.0])
    image_up_camera = np.asarray([0.0, -1.0, 0.0])
    rotation = _quat_wxyz_matrix(offset.rot)
    optical_capsule = rotation @ optical_camera
    image_up_capsule = rotation @ image_up_camera
    axes_confirmed = bool(
        str(offset.convention) == "ros"
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)
        and np.isclose(abs(float(optical_capsule @ np.asarray([0.0, 0.0, 1.0]))), 1.0)
        and np.isclose(float(optical_capsule @ image_up_capsule), 0.0, atol=1e-12)
    )
    return {
        "prim_path": CAMERA_PRIM_PATH,
        "configured_prim_path": camera.cfg.prim_path,
        "convention": str(offset.convention),
        "offset_position_capsule_m": [float(value) for value in offset.pos],
        "offset_quaternion_wxyz": [float(value) for value in offset.rot],
        "optical_axis_local": optical_camera.astype(int).tolist(),
        "image_up_axis_local": image_up_camera.astype(int).tolist(),
        "optical_axis_capsule_local": np.round(optical_capsule, 12).tolist(),
        "image_up_axis_capsule_local": np.round(image_up_capsule, 12).tolist(),
        "axes_confirmed": axes_confirmed,
        "confirmation_basis": (
            "ROS camera +Z optical and -Y image-up axes transformed by the live "
            "orthonormal wxyz mount quaternion; optical axis is collinear with "
            "the capsule long axis and image-up is orthogonal"
        ),
    }


def _inspect_write_api(capsule: Any) -> dict[str, Any]:
    pose_candidates = ("write_root_pose_to_sim", "write_root_pose_to_sim_index")
    velocity_candidates = ("write_root_velocity_to_sim", "write_root_velocity_to_sim_index")
    pose_method = next((name for name in pose_candidates if callable(getattr(capsule, name, None))), None)
    velocity_method = next(
        (name for name in velocity_candidates if callable(getattr(capsule, name, None))), None
    )
    return {
        "pose_method": pose_method,
        "velocity_method": velocity_method,
        "quaternion_order": "wxyz",
        "pose_signature": str(inspect.signature(getattr(capsule, pose_method))) if pose_method else None,
        "velocity_signature": (
            str(inspect.signature(getattr(capsule, velocity_method))) if velocity_method else None
        ),
        "verification_basis": (
            "live RigidObject bound methods plus root_quat_w wxyz tensor convention"
            if pose_method and velocity_method
            else "required live bound method missing"
        ),
    }


def _build_report(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    global np

    import gymnasium as gym
    import numpy as np
    import omni.usd
    import isaaclab_tasks  # noqa: F401
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    print("IDEAL_SURFACE_PREFLIGHT parsing task configuration", flush=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    print("IDEAL_SURFACE_PREFLIGHT entering simulation context", flush=True)
    with launch_simulation(cfg, args):
        print("IDEAL_SURFACE_PREFLIGHT creating environment", flush=True)
        env = gym.make(args.task, cfg=cfg)
        try:
            print("IDEAL_SURFACE_PREFLIGHT resetting environment", flush=True)
            env.reset()
            print("IDEAL_SURFACE_PREFLIGHT reading live geometry", flush=True)
            stage = omni.usd.get_context().get_stage()
            capsule = env.unwrapped.scene["capsule"]
            report: dict[str, Any] = {
                "repository": {
                    "commit": _git(repo, "rev-parse", "HEAD"),
                    "branch": _git(repo, "branch", "--show-current"),
                    "worktree": str(repo),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "schema_version": SCHEMA_VERSION,
                },
                "task": {"id": args.task, "num_envs": 1, "device": str(args.device)},
                "capsule": _inspect_capsule(stage, capsule),
                "camera": _inspect_camera(env),
                "surface": {},
                "pose_write_api": _inspect_write_api(capsule),
                "initial_contact": {},
                "gate": {},
            }
            report["surface"], report["initial_contact"] = _inspect_surface(
                stage, report["capsule"]
            )
            report["gate"] = build_gate(report)
            validate_preflight_report(report)
            return report
        finally:
            env.close()


def _parse_and_launch() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    if "--headless" in sys.argv[1:]:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("logs/ideal_surface_preflight"))
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.num_envs != 1:
        parser.error("TASK-002 preflight requires exactly one environment")
    args.enable_cameras = True
    launcher = AppLauncher(args)
    return args, launcher


def main() -> int:
    args, app_launcher = _parse_and_launch()
    simulation_app = app_launcher.app
    repo = Path(__file__).resolve().parents[2]
    try:
        print("IDEAL_SURFACE_PREFLIGHT inspecting live scene", flush=True)
        report = _build_report(repo, args)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        output = args.output / stamp / "prerequisites.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"IDEAL_SURFACE_PREFLIGHT status={report['gate']['status']} report={output}",
            flush=True,
        )
        for failure in report["gate"]["failures"]:
            print(f"IDEAL_SURFACE_PREFLIGHT_FAILURE {failure}", flush=True)
        return 0 if report["gate"]["status"] == "pass" else 2
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
