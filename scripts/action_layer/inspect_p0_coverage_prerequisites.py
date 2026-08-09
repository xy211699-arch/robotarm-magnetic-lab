#!/usr/bin/env python3
"""Read-only prerequisite inspection for TASK-001 P0 coverage.

The inspector launches the existing stomach task, reads the composed USD stage,
and probes the installed Warp ray-casting implementation.  It never authors the
stage or changes an existing task configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_TASK = "Template-Robotarm-Magnetic-Stomach-Lab-v0"
ATOMIC_TASK = "Template-Robotarm-Magnetic-Atomic-Table-Lab-v0"
STOMACH_ROOT = "/World/envs/env_0/Stomach"
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_preflight"
)


def empty_report() -> dict[str, Any]:
    """Return the complete top-level report shape with conservative defaults."""
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": None,
        "repository": {"commit": None, "branch": None},
        "dependencies": {},
        "atomic_task": {
            "environment_id": ATOMIC_TASK,
            "registered": False,
            "action_ids": {},
        },
        "camera": {
            "prim_path": None,
            "update_period_s": None,
            "offset_position_m": None,
            "offset_quaternion_wxyz": None,
            "convention": None,
            "optical_axis_camera": None,
            "live_position_world_m": None,
            "live_quaternion_ros_wxyz": None,
            "optical_transform_confirmed": False,
        },
        "stomach": {
            "root_prim": STOMACH_ROOT,
            "meshes": [],
            "selection_basis": [],
            "selection_unambiguous": False,
            "selected_inner_surface_prims": [],
        },
        "gpu_ray_apis": [],
        "gate": {"status": "needs_decision", "reasons": []},
    }


def _require(mapping: dict[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing {location}.{key}")
    return mapping[key]


def validate_report(report: dict[str, Any]) -> None:
    """Validate that a report contains every contract-critical field."""
    for key in (
        "schema_version",
        "repository",
        "dependencies",
        "atomic_task",
        "camera",
        "stomach",
        "gpu_ray_apis",
        "gate",
    ):
        _require(report, key, "report")
    for key in ("commit", "branch"):
        _require(report["repository"], key, "repository")
    for key in ("environment_id", "registered", "action_ids"):
        _require(report["atomic_task"], key, "atomic_task")
    for key in (
        "prim_path",
        "update_period_s",
        "offset_position_m",
        "offset_quaternion_wxyz",
        "convention",
        "optical_transform_confirmed",
    ):
        _require(report["camera"], key, "camera")
    for key in (
        "root_prim",
        "meshes",
        "selection_unambiguous",
        "selected_inner_surface_prims",
    ):
        _require(report["stomach"], key, "stomach")
    mesh_fields = (
        "prim_path",
        "vertex_count",
        "face_count",
        "topology",
        "world_transform",
        "world_bounds_m",
        "purpose",
        "visibility",
        "material_bindings",
        "surface_role",
    )
    for index, mesh in enumerate(report["stomach"]["meshes"]):
        for key in mesh_fields:
            _require(mesh, key, f"stomach.meshes[{index}]")
    for index, api in enumerate(report["gpu_ray_apis"]):
        for key in ("name", "available", "gpu_batched", "first_hit", "face_id"):
            _require(api, key, f"gpu_ray_apis[{index}]")
    for key in ("status", "reasons"):
        _require(report["gate"], key, "gate")


def _git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _matrix_values(matrix: Any) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _all_descendants(root: Any):
    stack = list(reversed(root.GetAllChildren()))
    while stack:
        prim = stack.pop()
        yield prim
        stack.extend(reversed(prim.GetAllChildren()))


def _mesh_edges(face_counts: list[int], face_indices: list[int]) -> tuple[int, int]:
    edges: Counter[tuple[int, int]] = Counter()
    cursor = 0
    for count in face_counts:
        face = face_indices[cursor : cursor + count]
        cursor += count
        for index, first in enumerate(face):
            second = face[(index + 1) % len(face)]
            edges[tuple(sorted((int(first), int(second))))] += 1
    return len(edges), sum(value == 1 for value in edges.values())


def _geometry_hash(world_points: list[list[float]], counts: list[int], indices: list[int]) -> str:
    payload = {
        "points": [[round(value, 12) for value in point] for point in world_points],
        "face_vertex_counts": [int(value) for value in counts],
        "face_vertex_indices": [int(value) for value in indices],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _inspect_meshes(stage: Any, root_path: str) -> list[dict[str, Any]]:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"stomach root is invalid: {root_path}")

    meshes: list[dict[str, Any]] = []
    for prim in _all_descendants(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = list(mesh.GetPointsAttr().Get() or [])
        counts = [int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or [])]
        indices = [int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])]
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        world_points = []
        for point in points:
            transformed = transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
            world_points.append([float(transformed[0]), float(transformed[1]), float(transformed[2])])
        if world_points:
            bounds = [
                [min(point[axis] for point in world_points) for axis in range(3)],
                [max(point[axis] for point in world_points) for axis in range(3)],
            ]
        else:
            bounds = [[math.nan] * 3, [math.nan] * 3]
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        material_paths = [str(material.GetPath())] if material and material.GetPrim().IsValid() else []
        collision_attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        collision_enabled = bool(collision_attr.Get()) if collision_attr else False
        visibility = str(UsdGeom.Imageable(prim).GetVisibilityAttr().Get() or "inherited")
        purpose = str(UsdGeom.Imageable(prim).GetPurposeAttr().Get() or "default")
        path = str(prim.GetPath())
        if not prim.IsActive():
            role = "inactive_helper"
        elif collision_enabled and visibility == "invisible":
            role = "collision_proxy"
        elif material_paths and not collision_enabled and visibility != "invisible":
            role = "rendered_surface_candidate"
        else:
            role = "unclassified"
        edge_count, boundary_edge_count = _mesh_edges(counts, indices)
        meshes.append(
            {
                "prim_path": path,
                "prim_name": prim.GetName(),
                "active": bool(prim.IsActive()),
                "vertex_count": len(points),
                "face_count": len(counts),
                "index_count": len(indices),
                "topology": {
                    "face_vertex_counts": dict(sorted(Counter(counts).items())),
                    "edge_count": edge_count,
                    "boundary_edge_count": boundary_edge_count,
                    "orientation": str(mesh.GetOrientationAttr().Get() or "rightHanded"),
                    "subdivision_scheme": str(mesh.GetSubdivisionSchemeAttr().Get() or "catmullClark"),
                },
                "world_transform": _matrix_values(transform),
                "world_bounds_m": bounds,
                "purpose": purpose,
                "visibility": visibility,
                "collision_enabled": collision_enabled,
                "material_bindings": material_paths,
                "geometry_sha256": _geometry_hash(world_points, counts, indices),
                "surface_role": role,
            }
        )
    return sorted(meshes, key=lambda item: item["prim_path"])


def _select_inner_surface(meshes: list[dict[str, Any]]) -> tuple[list[str], list[str], bool]:
    rendered = [mesh for mesh in meshes if mesh["surface_role"] == "rendered_surface_candidate"]
    collision = [mesh for mesh in meshes if mesh["surface_role"] == "collision_proxy"]
    unclassified_active = [
        mesh for mesh in meshes if mesh["active"] and mesh["surface_role"] == "unclassified"
    ]
    basis: list[str] = []
    if len(rendered) != 1:
        basis.append(f"expected exactly one rendered material-bound surface, found {len(rendered)}")
        return [], basis, False
    selected = rendered[0]
    if unclassified_active:
        basis.append(
            "active unclassified meshes remain: "
            + ", ".join(mesh["prim_path"] for mesh in unclassified_active)
        )
        return [], basis, False
    nonduplicate_collision = [
        mesh for mesh in collision if mesh["geometry_sha256"] != selected["geometry_sha256"]
    ]
    if nonduplicate_collision:
        basis.append("a collision proxy has geometry different from the rendered surface")
        return [], basis, False
    if selected["topology"]["boundary_edge_count"] <= 0:
        basis.append("rendered surface is closed; inner versus outer shell cannot be separated")
        return [], basis, False
    basis.extend(
        [
            "the selected mesh is the only active visible material-bound non-colliding stomach mesh",
            "all active collision-only stomach meshes are invisible exact geometry duplicates",
            "inactive planning/helper meshes are excluded",
            "the selected topology is one thin open surface with anatomical boundary edges; no outer wall, thickness side wall, or cap mesh exists",
        ]
    )
    return [selected["prim_path"]], basis, True


def _probe_gpu_ray_api(device: str) -> dict[str, Any]:
    import numpy as np
    import torch

    from isaaclab.utils.warp.ops import convert_to_warp_mesh, raycast_mesh

    result: dict[str, Any] = {
        "name": "isaaclab.utils.warp.ops.raycast_mesh",
        "available": False,
        "gpu_batched": False,
        "first_hit": False,
        "face_id": False,
        "device": device,
        "probe": {},
    }
    try:
        points = np.asarray(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        indices = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
        mesh = convert_to_warp_mesh(points, indices, device=device)
        starts = torch.tensor([[[0.5, -0.5, 1.0], [-0.5, 0.5, 2.0]]], device=device)
        directions = torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]], device=device)
        hits, distances, _, face_ids = raycast_mesh(
            starts,
            directions,
            mesh,
            max_dist=3.0,
            return_distance=True,
            return_face_id=True,
        )
        expected = torch.tensor([[1.0, 2.0]], device=device)
        passed = bool(torch.allclose(distances, expected, atol=1.0e-5))
        passed = passed and bool(torch.all(face_ids >= 0)) and bool(torch.all(torch.isfinite(hits)))
        result.update(
            {
                "available": passed,
                "gpu_batched": passed and str(device).startswith("cuda"),
                "first_hit": passed,
                "face_id": passed,
                "probe": {
                    "ray_count": 2,
                    "distances_m": distances.detach().cpu().reshape(-1).tolist(),
                    "face_ids": face_ids.detach().cpu().reshape(-1).tolist(),
                },
            }
        )
    except Exception as error:  # report the installed-stack limitation verbatim
        result["probe"] = {"error_type": type(error).__name__, "error": str(error)}
    return result


def _dependency_versions() -> dict[str, str]:
    names = ("torch", "warp-lang", "gymnasium", "numpy", "isaaclab")
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed-as-distribution"
    return versions


def _build_report(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    import omni.usd
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.action_layer import (
        AtomicAction,
    )

    report = empty_report()
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    report["repository"] = {
        "commit": _git_value(repo, "rev-parse", "HEAD"),
        "branch": _git_value(repo, "branch", "--show-current"),
        "worktree": str(repo),
    }
    report["dependencies"] = _dependency_versions()
    try:
        gym.spec(ATOMIC_TASK)
        registered = True
    except gym.error.Error:
        registered = False
    report["atomic_task"] = {
        "environment_id": ATOMIC_TASK,
        "registered": registered,
        "action_ids": {action.name: int(action) for action in AtomicAction},
    }

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    camera_cfg = cfg.scene.capsule_camera
    report["camera"].update(
        {
            "prim_path": camera_cfg.prim_path,
            "update_period_s": float(camera_cfg.update_period),
            "width_px": int(camera_cfg.width),
            "height_px": int(camera_cfg.height),
            "offset_position_m": [float(value) for value in camera_cfg.offset.pos],
            "offset_quaternion_wxyz": [float(value) for value in camera_cfg.offset.rot],
            "convention": str(camera_cfg.offset.convention),
            "optical_axis_camera": [0.0, 0.0, 1.0],
        }
    )

    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            env.reset()
            camera = env.unwrapped.scene["capsule_camera"]
            camera_pos = camera.data.pos_w.torch[0].detach().cpu().tolist()
            camera_quat = camera.data.quat_w_ros.torch[0].detach().cpu().tolist()
            report["camera"].update(
                {
                    "live_prim_path": str(camera.cfg.prim_path),
                    "live_position_world_m": [float(value) for value in camera_pos],
                    "live_quaternion_ros_wxyz": [float(value) for value in camera_quat],
                    "optical_transform_confirmed": (
                        camera_cfg.offset.convention == "ros"
                        and tuple(float(value) for value in camera_cfg.offset.pos) == (0.0, 0.0, -0.0127)
                        and tuple(float(value) for value in camera_cfg.offset.rot) == (0.0, 1.0, 0.0, 0.0)
                    ),
                    "confirmation_basis": (
                        "live Camera sensor matches configured capsule prim; ROS optical +Z is mounted "
                        "at capsule local -Z end by the frozen (0,1,0,0) wxyz offset"
                    ),
                }
            )
            stage = omni.usd.get_context().get_stage()
            meshes = _inspect_meshes(stage, STOMACH_ROOT)
            selected, basis, unambiguous = _select_inner_surface(meshes)
            report["stomach"].update(
                {
                    "meshes": meshes,
                    "selection_basis": basis,
                    "selection_unambiguous": unambiguous,
                    "selected_inner_surface_prims": selected,
                }
            )
            report["gpu_ray_apis"] = [_probe_gpu_ray_api(args.device)]
        finally:
            env.close()

    reasons: list[str] = []
    if not report["stomach"]["selection_unambiguous"]:
        reasons.append("complete inner luminal surface selection is ambiguous")
    if not report["camera"]["optical_transform_confirmed"]:
        reasons.append("capsule camera optical transform is not confirmed")
    suitable_api = any(
        api["available"] and api["gpu_batched"] and api["first_hit"] and api["face_id"]
        for api in report["gpu_ray_apis"]
    )
    if not suitable_api:
        reasons.append("no suitable GPU-batched first-hit ray API with face IDs")
    if not report["atomic_task"]["registered"]:
        reasons.append("existing atomic table task is not registered")
    report["gate"] = {"status": "pass" if not reasons else "needs_decision", "reasons": reasons}
    validate_report(report)
    return report


def _parse_and_launch() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    # Kit 110 resolves headless mode from visualizer intent/HEADLESS but no
    # longer exposes ``--headless`` through AppLauncher.  Preserve the task
    # contract's CLI by translating that spelling before AppLauncher parses.
    if "--headless" in sys.argv[1:]:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.num_envs != 1:
        parser.error("TASK-001 prerequisite inspection requires --num_envs 1")
    args.enable_cameras = True
    launcher = AppLauncher(args)
    return args, launcher.app


def main() -> int:
    args, simulation_app = _parse_and_launch()
    repo = Path(__file__).resolve().parents[2]
    try:
        report = _build_report(repo, args)
        if args.output is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = DEFAULT_OUTPUT_ROOT / stamp / "prerequisites.json"
        else:
            output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"P0_PREFLIGHT status={report['gate']['status']} report={output}", flush=True)
        for reason in report["gate"]["reasons"]:
            print(f"P0_PREFLIGHT_REASON {reason}", flush=True)
        return 0 if report["gate"]["status"] == "pass" else 2
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
