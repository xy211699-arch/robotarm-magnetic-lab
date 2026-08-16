"""TASK-003 live physics inspection and strict real-dynamics decision gate.

The schema and validator intentionally have no Isaac Sim imports so focused
tests can exercise the contract before launching Kit.  Live inspection is
added below the pure gate and imports version-dependent APIs only at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
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

CAPSULE_PRIM_PATH = "/World/envs/env_0/Scene/MagneticDemo/target_magnet"
STOMACH_COLLISION_PRIM_PATH = (
    "/World/envs/env_0/Stomach/ConvertedSource/Environment/Stomach/"
    "Physics_Collision_Mesh/Stomach"
)
SCHEMA_VERSION = "dynamic_force_preflight_v1"


TASK_ID = "Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0"
REQUIRED_REPORT_KEYS = (
    "repository",
    "task",
    "physics",
    "capsule",
    "stomach",
    "contact_sensor",
    "runtime_contract",
    "gate",
)
FORBIDDEN_RUNTIME_CALLS = (
    "write_root_pose_to_sim",
    "write_root_velocity_to_sim",
    "set_transforms",
    "set_velocities",
)


def _close(value: Any, expected: float, tolerance: float = 1.0e-9) -> bool:
    try:
        return math.isfinite(float(value)) and abs(float(value) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def build_gate(report: dict) -> dict:
    """Evaluate the immutable TASK-003 preflight contract."""
    failures: list[str] = []
    missing = [key for key in REQUIRED_REPORT_KEYS[:-1] if key not in report]
    if missing:
        return {
            "status": "needs_decision",
            "failures": [f"missing report section: {key}" for key in missing],
        }

    capsule = report["capsule"]
    physics = report["physics"]
    stomach = report["stomach"]
    task = report["task"]
    runtime = report["runtime_contract"]

    if task.get("id") != TASK_ID:
        failures.append("unexpected task id")
    if int(task.get("num_envs", 0)) != 1:
        failures.append("task must use exactly one environment")
    if list(task.get("action_terms", [])) != ["dynamic_force"]:
        failures.append("dynamic_force must be the only action term")
    forbidden_terms = list(runtime.get("magnetic_or_ideal_terms", []))
    if forbidden_terms:
        failures.append(f"forbidden actuator terms are active: {forbidden_terms}")

    if not _close(physics.get("dt_s"), 1.0 / 240.0):
        failures.append("physics dt is not 1/240 s")
    if not _close(physics.get("environment_rate_hz"), 60.0):
        failures.append("environment rate is not 60 Hz")
    if int(physics.get("render_interval", 0)) != 4:
        failures.append("render interval is not four physics steps")
    if not bool(physics.get("scene_ccd_enabled")):
        failures.append("CCD is not active at scene and body levels")

    if bool(capsule.get("kinematic_enabled")):
        failures.append("capsule is kinematic")
    if not bool(capsule.get("gravity_enabled")):
        failures.append("capsule gravity is disabled")
    if not bool(capsule.get("ccd_enabled")):
        failures.append("CCD is not active at scene and body levels")
    if not bool(capsule.get("collision_enabled")):
        failures.append("capsule collision is disabled")
    if capsule.get("shape") != "Capsule" or str(capsule.get("axis", "")).upper() != "Z":
        failures.append("capsule collider is not a Z-axis spherocylinder")
    if not _close(capsule.get("radius_m"), 0.0065):
        failures.append("capsule radius is not 6.5 mm")
    if not _close(capsule.get("cylinder_height_m"), 0.012):
        failures.append("capsule cylinder height is not 12 mm")
    if not _close(capsule.get("total_length_m"), 0.025):
        failures.append("capsule total length is not 25 mm")
    try:
        mass = float(capsule.get("mass_kg"))
    except (TypeError, ValueError):
        mass = math.nan
    if not math.isfinite(mass):
        failures.append("capsule mass is not finite")
    try:
        if float(capsule.get("mass_kg", 0.0)) <= 0.0:
            failures.append("capsule mass is not positive")
    except (TypeError, ValueError):
        failures.append("capsule mass is not positive")
    inertia = capsule.get("inertia_kg_m2", [])
    try:
        if len(inertia) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in inertia):
            failures.append("capsule inertia is not finite and positive")
    except (TypeError, ValueError):
        failures.append("capsule inertia is not finite and positive")

    if not bool(stomach.get("collision_enabled")):
        failures.append("stomach collision is disabled")
    if not bool(stomach.get("static")):
        failures.append("stomach collider is not static")
    if not bool(report["contact_sensor"].get("present")):
        failures.append("contact sensor is unavailable")
    if list(runtime.get("forbidden_calls", [])):
        failures.append("forbidden runtime state writer")
    if not bool(runtime.get("force_at_center_of_mass")):
        failures.append("force application at center of mass is unverified")
    if not bool(runtime.get("commanded_torque_zero")):
        failures.append("commanded torque is not identically zero")

    return {"status": "pass" if not failures else "needs_decision", "failures": failures}


def validate_preflight_report(report: dict) -> None:
    """Raise when a report cannot prove the frozen real-dynamics contract."""
    missing = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"missing report sections: {missing}")
    evaluated = build_gate(report)
    recorded = report.get("gate")
    if recorded != evaluated:
        raise ValueError(f"stale or inconsistent gate: recorded={recorded}, evaluated={evaluated}")
    if evaluated["status"] != "pass":
        raise ValueError("; ".join(evaluated["failures"]))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _descendants_including(root: Any):
    stack = [root]
    while stack:
        prim = stack.pop()
        yield prim
        stack.extend(reversed(prim.GetAllChildren()))


def _authored_bool(attr: Any, default: bool) -> bool:
    if not attr or not attr.IsValid():
        return default
    value = attr.Get()
    return default if value is None else bool(value)


def _find_attr_value(prim: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid():
            value = attr.Get()
            if value is not None:
                return value
    return default


def _inspect_capsule(stage: Any, capsule: Any) -> dict[str, Any]:
    import numpy as np
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(CAPSULE_PRIM_PATH)
    rigid_api = UsdPhysics.RigidBodyAPI(root)
    physx_api = PhysxSchema.PhysxRigidBodyAPI(root)
    colliders: list[dict[str, Any]] = []
    for prim in _descendants_including(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_api = UsdPhysics.CollisionAPI(prim)
        enabled = _authored_bool(collision_api.GetCollisionEnabledAttr(), True)
        item: dict[str, Any] = {
            "prim_path": str(prim.GetPath()),
            "type_name": prim.GetTypeName(),
            "enabled": enabled,
        }
        if prim.IsA(UsdGeom.Capsule):
            shape = UsdGeom.Capsule(prim)
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            # The delivered task authors unit scale; record the live USD
            # dimensions directly and preserve transform evidence separately.
            item.update(
                radius_m=float(shape.GetRadiusAttr().Get()),
                cylinder_height_m=float(shape.GetHeightAttr().Get()),
                axis=str(shape.GetAxisAttr().Get() or "Z"),
                world_transform_has_finite_rotation=bool(
                    np.isfinite(
                        np.asarray(
                            [[float(transform[i][j]) for j in range(4)] for i in range(4)]
                        )
                    ).all()
                ),
            )
        colliders.append(item)
    enabled_capsules = [
        item for item in colliders if item["enabled"] and item["type_name"] == "Capsule"
    ]
    selected = enabled_capsules[0] if len(enabled_capsules) == 1 else {}
    mass = float(capsule.data.body_mass.torch[0, 0].item())
    inertia = capsule.data.body_inertia.torch[0, 0].detach().cpu().numpy().reshape(3, 3)
    radius = float(selected.get("radius_m", math.nan))
    cylinder_height = float(selected.get("cylinder_height_m", math.nan))
    return {
        "prim_path": CAPSULE_PRIM_PATH,
        "rigid_body_api": bool(rigid_api),
        "kinematic_enabled": _authored_bool(rigid_api.GetKinematicEnabledAttr(), False),
        "gravity_enabled": not _authored_bool(physx_api.GetDisableGravityAttr(), False),
        "ccd_enabled": _authored_bool(physx_api.GetEnableCCDAttr(), False),
        "collision_enabled": len(enabled_capsules) == 1,
        "shape": "Capsule" if len(enabled_capsules) == 1 else "ambiguous",
        "axis": selected.get("axis"),
        "radius_m": radius,
        "cylinder_height_m": cylinder_height,
        "total_length_m": cylinder_height + 2.0 * radius,
        "mass_kg": mass,
        "inertia_kg_m2": np.diag(inertia).astype(float).tolist(),
        "center_of_mass_local_pose": capsule.data.body_com_pose_b.torch[0, 0]
        .detach()
        .cpu()
        .tolist(),
        "max_linear_velocity_m_s": float(
            _find_attr_value(root, ("physxRigidBody:maxLinearVelocity",), math.nan)
        ),
        "colliders": colliders,
    }


def _triangulate_mesh(mesh: Any) -> tuple[Any, Any]:
    import numpy as np

    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + int(count)]
        cursor += int(count)
        for offset in range(1, int(count) - 1):
            triangles.append((int(face[0]), int(face[offset]), int(face[offset + 1])))
    return points, np.asarray(triangles, dtype=np.int64)


def _edge_statistics(triangles: Any) -> dict[str, int]:
    counts: Counter[tuple[int, int]] = Counter()
    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            counts[tuple(sorted((int(first), int(second))))] += 1
    return {
        "edge_count": len(counts),
        "boundary_edge_count": sum(value == 1 for value in counts.values()),
        "nonmanifold_edge_count": sum(value > 2 for value in counts.values()),
    }


def _inspect_stomach(stage: Any) -> dict[str, Any]:
    import numpy as np
    from pxr import UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(STOMACH_COLLISION_PRIM_PATH)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        return {
            "prim_path": STOMACH_COLLISION_PRIM_PATH,
            "collision_enabled": False,
            "static": False,
            "vertex_count": 0,
            "triangle_count": 0,
            "geometry_sha256": None,
            "edge_statistics": {},
        }
    collision_api = UsdPhysics.CollisionAPI(prim)
    points, triangles = _triangulate_mesh(UsdGeom.Mesh(prim))
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(points, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(triangles, dtype="<i8").tobytes())
    has_rigid_ancestor = False
    cursor = prim
    while cursor and cursor.IsValid() and not cursor.IsPseudoRoot():
        if cursor.HasAPI(UsdPhysics.RigidBodyAPI):
            has_rigid_ancestor = True
            break
        cursor = cursor.GetParent()
    return {
        "prim_path": STOMACH_COLLISION_PRIM_PATH,
        "collision_enabled": bool(collision_api)
        and _authored_bool(collision_api.GetCollisionEnabledAttr(), True),
        "static": not has_rigid_ancestor,
        "vertex_count": int(len(points)),
        "triangle_count": int(len(triangles)),
        "geometry_sha256": digest.hexdigest(),
        "edge_statistics": _edge_statistics(triangles),
    }


def _inspect_physics(stage: Any, env: Any) -> dict[str, Any]:
    from pxr import PhysxSchema, UsdPhysics

    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        scene_path = None
        scene_ccd = False
    else:
        scene_path = str(scenes[0].GetPath())
        scene_api = PhysxSchema.PhysxSceneAPI(scenes[0])
        scene_ccd = _authored_bool(scene_api.GetEnableCCDAttr(), False)
    return {
        "scene_prim_path": scene_path,
        "device": str(env.unwrapped.device),
        "dt_s": float(env.unwrapped.cfg.sim.dt),
        "environment_rate_hz": 1.0
        / float(env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation),
        "render_interval": int(env.unwrapped.cfg.sim.render_interval),
        "scene_ccd_enabled": scene_ccd,
        "cpu_physx_required_by_installed_api": True,
    }


def _scan_runtime_contract(action_term: Any) -> dict[str, Any]:
    runtime_paths = (
        ROOT
        / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/"
        "robotarm_magnetic_lab/mdp/dynamic_force_action.py",
        ROOT / "scripts/dynamic_force/teleop_dynamic_force_stomach.py",
        ROOT / "scripts/dynamic_force/validate_dynamic_force_stomach.py",
    )
    forbidden: list[dict[str, str]] = []
    scanned: list[str] = []
    for path in runtime_paths:
        if not path.exists():
            continue
        scanned.append(str(path.relative_to(ROOT)))
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_RUNTIME_CALLS:
            if token in text:
                forbidden.append({"path": str(path.relative_to(ROOT)), "call": token})
    method = action_term.capsule.permanent_wrench_composer.set_forces_and_torques_index
    doc = inspect.getdoc(method) or ""
    source = inspect.getsource(type(action_term))
    terms = [
        name
        for name in action_term._env.action_manager.active_terms
        if any(token in name.lower() for token in ("magnet", "ideal", "torque", "robot"))
    ]
    return {
        "scanned_files": scanned,
        "forbidden_calls": forbidden,
        "magnetic_or_ideal_terms": terms,
        "force_at_center_of_mass": "positions=None" in source
        and ("center of mass" in doc.lower() or "com" in doc.lower()),
        "commanded_torque_zero": "_applied_torque_world.zero_()" in source,
        "wrench_method_signature": str(inspect.signature(method)),
        "wrench_method_documentation": doc,
    }


def _build_report(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import omni.usd
    import isaaclab_tasks  # noqa: F401
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device="cpu", num_envs=1, use_fabric=not args.disable_fabric)
    cfg.sim.device = "cpu"
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            env.reset()
            stage = omni.usd.get_context().get_stage()
            capsule = env.unwrapped.scene["capsule"]
            action_term = env.unwrapped.action_manager.get_term("dynamic_force")
            report: dict[str, Any] = {
                "repository": {
                    "commit": _git(repo, "rev-parse", "HEAD"),
                    "branch": _git(repo, "branch", "--show-current"),
                    "worktree": str(repo),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "schema_version": SCHEMA_VERSION,
                },
                "task": {
                    "id": args.task,
                    "num_envs": int(env.unwrapped.num_envs),
                    "action_terms": list(env.unwrapped.action_manager.active_terms),
                },
                "physics": _inspect_physics(stage, env),
                "capsule": _inspect_capsule(stage, capsule),
                "stomach": _inspect_stomach(stage),
                "contact_sensor": {
                    "present": "capsule_contact" in env.unwrapped.scene.keys(),
                    "prim_path": str(env.unwrapped.scene["capsule_contact"].cfg.prim_path),
                },
                "runtime_contract": _scan_runtime_contract(action_term),
                "gate": {},
            }
            report["gate"] = build_gate(report)
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
    parser.add_argument(
        "--output", type=Path, default=Path("logs/dynamic_force_preflight")
    )
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.num_envs != 1:
        parser.error("TASK-003 preflight requires exactly one environment")
    args.device = "cpu"
    args.enable_cameras = True
    return args, AppLauncher(args)


def main() -> int:
    args, app_launcher = _parse_and_launch()
    try:
        report = _build_report(ROOT, args)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        output = args.output / stamp / "prerequisites.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"DYNAMIC_FORCE_PREFLIGHT_{report['gate']['status'].upper()} report={output}",
            flush=True,
        )
        for failure in report["gate"]["failures"]:
            print(f"DYNAMIC_FORCE_PREFLIGHT_FAILURE {failure}", flush=True)
        return 0 if report["gate"]["status"] == "pass" else 2
    except ValueError as error:
        print(f"DYNAMIC_FORCE_PREFLIGHT_NEEDS_DECISION {error}", flush=True)
        return 2
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
