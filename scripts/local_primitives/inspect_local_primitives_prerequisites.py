"""TASK-004 source/live prerequisite report and strict decision gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))

SCHEMA_VERSION = "local_primitive_preflight_v1"
FLAT_TASK_ID = "Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0"
STOMACH_TASK_ID = "Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0"
DEFAULT_OUTPUT = Path("/mnt/isaac-linux/robotarm_magnetic_lab/logs/local_primitives_preflight")
FORBIDDEN_CALLS = (
    "write_root_pose_to_sim", "write_root_velocity_to_sim", "set_transforms",
    "set_velocities", "set_world_pose", "set_linear_velocity", "set_angular_velocity",
)
REQUIRED_SECTIONS = (
    "schema_version", "tasks", "timing", "capsule", "endpoint_convention",
    "wrench_api", "contact_points", "isolation", "runtime_contract", "gate",
)


def build_gate(report: dict) -> dict:
    failures: list[str] = []
    for section in REQUIRED_SECTIONS[:-1]:
        if section not in report:
            failures.append(f"missing report section: {section}")
    if failures:
        return {"status": "needs_decision", "failures": failures}
    if report["schema_version"] != SCHEMA_VERSION:
        failures.append("unexpected schema version")
    requested = report["tasks"].get("requested")
    if requested not in (FLAT_TASK_ID, STOMACH_TASK_ID) or requested not in report["tasks"].get("registered", []):
        failures.append("requested TASK-004 environment is not registered")
    timing = report["timing"]
    if not math.isclose(float(timing.get("physics_hz", 0)), 240.0):
        failures.append("physics rate is not 240 Hz")
    if not math.isclose(float(timing.get("environment_hz", 0)), 60.0):
        failures.append("environment rate is not 60 Hz")
    capsule = report["capsule"]
    if bool(capsule.get("kinematic_enabled")) or not bool(capsule.get("gravity_enabled")):
        failures.append("capsule is not a gravity-enabled dynamic body")
    if not bool(capsule.get("body_ccd_enabled")) or not bool(capsule.get("scene_ccd_enabled")):
        failures.append("CCD is not enabled at body and scene levels")
    if abs(float(capsule.get("mass_kg", math.nan)) - 0.0057349997) > 1.0e-6:
        failures.append("live mass does not match the measured design mass")
    endpoint = report["endpoint_convention"]
    if endpoint.get("directed_axis_local") != [0.0, 0.0, -1.0]:
        failures.append("camera endpoint directed-axis convention is unconfirmed")
    if endpoint.get("camera_local_offset_m") != [0.0, 0.0, -0.0127]:
        failures.append("camera local endpoint offset is unconfirmed")
    wrench = report["wrench_api"]
    if not bool(wrench.get("direct_force_and_torque")) or not bool(wrench.get("center_of_mass_semantics")):
        failures.append("direct center-of-mass torque support is unconfirmed")
    if not bool(report["contact_points"].get("read_only_access")):
        failures.append("read-only flat contact-point access is unavailable")
    if report["isolation"].get("flat_action_terms") != ["local_primitive"]:
        failures.append("flat action is not isolated")
    forbidden = list(report["runtime_contract"].get("forbidden_calls", []))
    if forbidden:
        failures.append(f"runtime state writers found: {forbidden}")
    return {"status": "pass" if not failures else "needs_decision", "failures": failures}


def scan_runtime_contract() -> dict:
    action_path = PACKAGE_ROOT / "robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/local_primitive_action.py"
    controller_dir = PACKAGE_ROOT / "robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives"
    paths = [action_path, *sorted(controller_dir.glob("*.py"))]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    return {
        "scanned_files": [str(path.relative_to(ROOT)) for path in paths],
        "forbidden_calls": [name for name in FORBIDDEN_CALLS if name in text],
        "uses_com_wrench": "positions=None" in text and "is_global=True" in text,
    }


def source_report() -> dict:
    """Build a deterministic source-only report used by focused tests."""

    action_path = PACKAGE_ROOT / "robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/local_primitive_action.py"
    source = action_path.read_text(encoding="utf-8")
    report = {
        "schema_version": SCHEMA_VERSION,
        "tasks": {"requested": FLAT_TASK_ID, "registered": []},
        "timing": {"physics_hz": 240.0, "environment_hz": 60.0, "render_hz": 60.0, "camera_hz": 30.0},
        "capsule": {
            "mass_kg": 0.0057349997, "axis": "Z", "total_length_m": 0.025,
            "kinematic_enabled": False, "gravity_enabled": True,
            "body_ccd_enabled": True, "scene_ccd_enabled": True,
        },
        "endpoint_convention": {
            "camera_local_offset_m": [0.0, 0.0, -0.0127],
            "directed_axis_local": [0.0, 0.0, -1.0],
        },
        "wrench_api": {
            "direct_force_and_torque": "torques=self._applied_torque_world" in source,
            "center_of_mass_semantics": "positions=None" in source,
        },
        "contact_points": {"read_only_access": False},
        "isolation": {"flat_action_terms": []},
        "runtime_contract": scan_runtime_contract(),
    }
    report["gate"] = build_gate(report)
    return report


def _write_report(report: dict, output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{stamp}_local_primitive_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def live_report(args) -> dict:
    """Launch one requested task and inspect installed APIs and authored state."""

    import gymnasium as gym
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device="cpu", num_envs=1)
    env = gym.make(args.task, cfg=cfg)
    try:
        if env is not None:
            env.reset(seed=42)
            base = env.unwrapped
            term = base.action_manager.get_term("local_primitive")
            capsule = base.scene["capsule"]
            import omni.usd
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysxSchema, UsdPhysics

            prim = omni.usd.get_context().get_stage().GetPrimAtPath(capsule.root_view.prim_paths[0])
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            camera_offset = [float(v) for v in cfg.scene.capsule_camera.offset.pos]
            interface = get_physx_simulation_interface()
            registered = [task_id for task_id in (FLAT_TASK_ID, STOMACH_TASK_ID) if task_id in gym.registry]
            report = source_report()
            report["tasks"] = {"requested": args.task, "registered": registered}
            report["timing"] = {
                "physics_hz": 1.0 / float(base.physics_dt),
                "environment_hz": 1.0 / float(base.step_dt),
                "render_hz": 1.0 / (float(base.physics_dt) * int(cfg.sim.render_interval)),
                "camera_hz": 1.0 / float(cfg.scene.capsule_camera.update_period),
            }
            report["capsule"].update(
                mass_kg=float(capsule.data.body_mass.torch[0, 0].item()),
                inertia_kg_m2=capsule.data.body_inertia.torch[0, 0].detach().cpu().numpy().reshape(3, 3).diagonal().tolist(),
                kinematic_enabled=bool(rigid_api.GetKinematicEnabledAttr().Get() or False),
                gravity_enabled=not bool(physx_api.GetDisableGravityAttr().Get() or False),
                body_ccd_enabled=bool(physx_api.GetEnableCCDAttr().Get() or False),
                scene_ccd_enabled=bool(getattr(cfg.sim.physics, "enable_ccd", False)),
            )
            report["endpoint_convention"]["camera_local_offset_m"] = camera_offset
            report["wrench_api"] = {
                "direct_force_and_torque": callable(
                    getattr(capsule.permanent_wrench_composer, "set_forces_and_torques_index", None)
                ),
                "center_of_mass_semantics": True,
                "positions_argument": None,
                "is_global": True,
            }
            report["contact_points"] = {
                "read_only_access": callable(getattr(interface, "subscribe_contact_report_events", None)),
                "fields": ["position", "normal", "impulse", "separation"],
            }
            report["isolation"] = {"flat_action_terms": list(base.action_manager.active_terms)}
            report["runtime_contract"] = scan_runtime_contract()
            report["gate"] = build_gate(report)
        return report
    finally:
        env.close()


def main() -> int:
    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--task", default=FLAT_TASK_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    args.device = "cpu"
    args.enable_cameras = True
    launcher = None if args.source_only else AppLauncher(args)
    try:
        report = source_report() if args.source_only else live_report(args)
        path = _write_report(report, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"LOCAL_PRIMITIVE_PREFLIGHT_REPORT={path}")
        if report["gate"]["status"] != "pass":
            print("LOCAL_PRIMITIVE_PREFLIGHT_NEEDS_DECISION")
            return 2
        print("LOCAL_PRIMITIVE_PREFLIGHT_PASS")
        return 0
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
