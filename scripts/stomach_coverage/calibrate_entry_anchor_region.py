#!/usr/bin/env python3
"""Calibrate TASK-009B by dynamic settling and connected surface geodesics."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

from isaaclab.app import AppLauncher


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument(
    "--config_directory", type=Path, default=ROOT / "configs" / "task009b"
)
parser.add_argument(
    "--log_root", type=Path, default=ROOT / "logs" / "task009b_entry_calibration"
)
parser.add_argument(
    "--resume_anchor",
    action="store_true",
    help="Load the confirmed anchor and open region selection without repeating the drop.",
)
parser.add_argument(
    "--initial_radius_mm",
    type=int,
    choices=tuple(range(10, 81, 5)),
    default=None,
    help="Initial geodesic preview radius; useful with --resume_anchor.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this calibrator only accepts {TASK_ID}")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import carb.input
import gymnasium as gym
import numpy as np
import omni.appwindow
import omni.ui
import omni.usd
import torch
from pxr import Gf, UsdGeom, Vt

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.coverage.entry_surface_region import (
    ANCHOR_SCHEMA,
    ENTRY_RADII_M,
    anchor_record,
    geodesic_face_distances,
    load_and_validate,
    nearest_surface_point,
    region_record,
    save_and_reload,
    shared_edge_adjacency,
    surface_region_from_distances,
)
from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage
from robotarm_magnetic_lab.teleop.atomic_keyboard import normalize_key
from robotarm_magnetic_lab.teleop.parameterized_force_keyboard import (
    ParameterizedForceKeyboard,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    CONTROL_HZ,
    PHYSICS_HZ,
    PHYSICS_STEPS_PER_CONTROL,
    ParameterizedForceMode,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_stomach_env_cfg import (
    STOMACH_SCENE_USD_PATH,
)


PHYSICS_DT_S = 1.0 / PHYSICS_HZ
STABLE_DURATION_S = 0.25
STABLE_STEPS = 60
MAX_SETTLE_S = 2.0
MAX_SETTLE_STEPS = 480
MAX_LINEAR_SPEED_M_S = 0.002
MAX_ANGULAR_SPEED_RAD_S = np.deg2rad(5.0)
CAPSULE_ASSET_IDENTIFIER = (
    f"{STOMACH_SCENE_USD_PATH}#/World/MagneticDemo/target_magnet"
)
CONTROLS = (
    "CONTROL: hold A/D=MOVE-/+, Q/E=VIEW-/+, W=UP, Space=HOLD; "
    "Z/X/C=alpha 0/0.5/1; R=reset; Enter=settle | "
    "SETTLED: Y=accept, Backspace=continue | REGION: [ / ]=radius, Enter=save | Esc=exit"
)


class SessionLog:
    def __init__(self, root: Path) -> None:
        self.directory = Path(root) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "events.jsonl"

    def emit(self, event: str, **values) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **values,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"TASK009B_{event} {json.dumps(values, sort_keys=True)}", flush=True)


class Keyboard:
    def __init__(self) -> None:
        self.down: set[str] = set()
        self.presses: deque[str] = deque()
        self.force = ParameterizedForceKeyboard(alpha=0.5)
        self._input = carb.input.acquire_input_interface()
        self._device = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._device, self._on_event
        )

    def _on_event(self, event, *_args):
        # Kit 110 emits both carb.input.Input objects and plain strings here,
        # depending on the physical key/backend. Normalize both forms.
        raw_input = event.input
        key = normalize_key(getattr(raw_input, "name", raw_input))
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key not in self.down:
                self.presses.append(key)
            self.down.add(key)
            self.force.key_event(key, True)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.down.discard(key)
            self.force.key_event(key, False)
        return True

    def release_actions(self) -> None:
        self.force.release_all()

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._device, self._subscription)
            self._subscription = None
        self.down.clear()
        self.force.release_all()


class Panel:
    def __init__(self) -> None:
        self.window = omni.ui.Window("TASK-009B Entry Anchor and Region", width=820, height=230)
        with self.window.frame:
            with omni.ui.VStack(spacing=5):
                self.phase = omni.ui.Label("", height=28)
                self.state = omni.ui.Label("", height=50, word_wrap=True)
                self.region = omni.ui.Label("", height=48, word_wrap=True)
                omni.ui.Label(CONTROLS, height=58, word_wrap=True)
                self.status = omni.ui.Label("Waiting for anchor and region confirmation.", height=35)

    def update_control(self, position, mode, alpha, telemetry) -> None:
        self.phase.text = "Phase: DYNAMIC_CONTROL"
        self.state.text = (
            f"capsule_world_m={np.round(position, 6).tolist()} "
            f"mode={mode.name} alpha={alpha:.1f} "
            f"force_ratio={0.0 if telemetry is None else telemetry.force_ratio:.3f} "
            "physics_paused=False"
        )

    def update_settle(self, elapsed, linear_speed, angular_speed, stable_time) -> None:
        self.phase.text = "Phase: DYNAMIC_SETTLE"
        self.state.text = (
            f"elapsed={elapsed:.3f}s linear={1000.0 * linear_speed:.3f} mm/s "
            f"angular={np.rad2deg(angular_speed):.3f} deg/s stable_for={stable_time:.3f}s"
        )

    def update_region(self, closest, region) -> None:
        self.phase.text = "Phase: GEODESIC_REGION"
        self.region.text = (
            f"seed_face={closest.triangle_index} radius={1000.0 * region.radius_m:.0f} mm "
            f"faces={len(region.triangle_indices)} vertices={len(region.vertex_indices)} "
            f"area={region.area_m2:.9f} m^2 components={region.connected_components}"
        )


class RegionDebugGeometry:
    ROOT = "/World/TASK009BEntryRegion"

    @staticmethod
    def _interpolation(gprim, name: str, token) -> None:
        primvar = UsdGeom.PrimvarsAPI(gprim.GetPrim()).GetPrimvar(name)
        if not primvar or not primvar.IsDefined():
            raise RuntimeError(f"missing USD primvar {gprim.GetPath()}.{name}")
        primvar.SetInterpolation(token)

    def __init__(self) -> None:
        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, self.ROOT)
        root.GetPrim().SetMetadata("comment", "TASK-009B visual-only connected entry region")
        self.mesh = UsdGeom.Mesh.Define(stage, self.ROOT + "/SelectedSurface")
        self.mesh.CreatePointsAttr([])
        self.mesh.CreateFaceVertexCountsAttr([])
        self.mesh.CreateFaceVertexIndicesAttr([])
        self.mesh.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.15, 0.0)])
        self.mesh.CreateDisplayOpacityAttr([0.84])
        self._interpolation(self.mesh, "displayColor", UsdGeom.Tokens.constant)
        self._interpolation(self.mesh, "displayOpacity", UsdGeom.Tokens.constant)
        self.mesh.CreateDoubleSidedAttr(True)
        self.seed = UsdGeom.Points.Define(stage, self.ROOT + "/ClosestSurfacePoint")
        self.seed.CreatePointsAttr([])
        self.seed.CreateWidthsAttr([0.006])
        self.seed.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        self.seed.CreateDisplayColorAttr([Gf.Vec3f(0.0, 1.0, 0.15)])
        self._interpolation(self.seed, "displayColor", UsdGeom.Tokens.constant)

    def update(self, reference, closest, region) -> None:
        triangle_points = reference.vertices_world[
            reference.triangles[region.triangle_indices]
        ]
        if len(triangle_points):
            normals = np.cross(
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 0],
            )
            lengths = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15)
            offsets = 0.00035 * normals / lengths
            points = np.concatenate(
                [triangle_points + offsets[:, None, :], triangle_points - offsets[:, None, :]],
                axis=0,
            ).reshape(-1, 3)
        else:
            points = np.empty((0, 3), dtype=np.float64)
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        self.mesh.GetFaceVertexCountsAttr().Set([3] * (2 * len(triangle_points)))
        self.mesh.GetFaceVertexIndicesAttr().Set(list(range(len(points))))
        self.seed.GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(closest.point_world_m.reshape(1, 3).astype(np.float32))
        )


def _tensor(value):
    return getattr(value, "torch", value)


def _pose(capsule) -> np.ndarray:
    return _tensor(capsule.data.root_pose_w)[0].detach().cpu().numpy().astype(np.float64)


def _velocity(capsule) -> np.ndarray:
    return _tensor(capsule.data.root_com_vel_w)[0].detach().cpu().numpy().astype(np.float64)


def _write_state(capsule, pose_xyzw: np.ndarray, device: str) -> None:
    pose = torch.as_tensor(pose_xyzw, device=device, dtype=torch.float32).reshape(1, 7)
    capsule.write_root_pose_to_sim_index(root_pose=pose)
    capsule.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=device, dtype=torch.float32)
    )


def _is_exit(key: str) -> bool:
    return key in ("ESC", "ESCAPE")


def _is_left_bracket(key: str) -> bool:
    return key in ("[", "LEFT_BRACKET", "BRACKET_LEFT")


def _is_right_bracket(key: str) -> bool:
    return key in ("]", "RIGHT_BRACKET", "BRACKET_RIGHT")


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    env = keyboard = None
    session = SessionLog(args_cli.log_root)
    anchor_path = args_cli.config_directory / "entry_anchor_v1.json"
    region_path = args_cli.config_directory / "entry_region_v1.json"
    session.emit("ENTRY_CALIBRATION_STARTED", log_directory=str(session.directory.resolve()))

    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            base = env.unwrapped
            capsule = base.scene["capsule"]
            term = base.action_manager.get_term("parameterized_force")
            sim = base.sim
            reference = reference_from_stage()
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(capsule.root_view.prim_paths[0])
            from pxr import UsdPhysics

            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if not rigid_api:
                raise RuntimeError("capsule is not a USD Dynamic rigid body")
            kinematic = rigid_api.GetKinematicEnabledAttr()
            if kinematic and bool(kinematic.Get()):
                raise RuntimeError("capsule must remain Dynamic; kinematic capsule is forbidden")

            default_pose = _pose(capsule)
            release_pose = None
            settled_pose = None
            anchor = None
            closest = distances = region = None
            requested_radius_m = (
                None
                if args_cli.initial_radius_mm is None
                else float(args_cli.initial_radius_mm) / 1000.0
            )
            radius_index = 2
            if requested_radius_m is not None:
                radius_index = min(
                    range(len(ENTRY_RADII_M)),
                    key=lambda index: abs(ENTRY_RADII_M[index] - requested_radius_m),
                )
            geometry = RegionDebugGeometry()
            keyboard = Keyboard()
            panel = Panel()
            term.reset()
            capsule.permanent_wrench_composer.reset()
            phase = "DYNAMIC_CONTROL"
            settle_steps = stable_steps = 0
            release_start_sim_time = None
            paused_sim_time = None
            last_status_wall = 0.0
            active_signature = None
            render_interval = max(1, int(cfg.sim.render_interval))
            if args_cli.resume_anchor:
                sim.physics_manager.pause()
                paused_sim_time = float(sim.physics_manager.get_simulation_time())
                anchor = load_and_validate(anchor_path, ANCHOR_SCHEMA)
                if anchor["stomach_geometry_sha256"] != reference.geometry_sha256:
                    raise RuntimeError(
                        "saved anchor is invalid for the current stomach geometry: "
                        f"{anchor['stomach_geometry_sha256']} != {reference.geometry_sha256}"
                    )
                default_pose = np.asarray(
                    anchor["default_pose_world_xyzw"], dtype=np.float64
                )
                release_pose = np.asarray(
                    anchor["release_pose_world_xyzw"], dtype=np.float64
                )
                settled_pose = np.asarray(
                    anchor["settled_pose_world_xyzw"], dtype=np.float64
                )
                _write_state(capsule, settled_pose, base.device)
                sim.forward()
                base.scene.update(0.0)
                closest = nearest_surface_point(reference, settled_pose[:3])
                distances = geodesic_face_distances(
                    shared_edge_adjacency(reference), closest.triangle_index
                )
                region = surface_region_from_distances(
                    reference, distances, ENTRY_RADII_M[radius_index]
                )
                geometry.update(reference, closest, region)
                panel.update_region(closest, region)
                panel.status.text = "Confirmed anchor restored. Select radius and press Enter."
                phase = "GEODESIC_REGION"
                session.emit(
                    "ENTRY_ANCHOR_RESUMED",
                    path=str(anchor_path.resolve()),
                    config_sha256=anchor["config_sha256"],
                    settled_pose_world_xyzw=settled_pose.tolist(),
                    seed_triangle_index=closest.triangle_index,
                    radius_m=region.radius_m,
                    physics_paused=True,
                    simulation_time_s=paused_sim_time,
                    reload_validated=True,
                )
            else:
                sim.physics_manager.play()
                session.emit(
                    "ENTRY_DYNAMIC_CONTROL_READY",
                    default_pose_world_xyzw=default_pose.tolist(),
                    physics_hz=PHYSICS_HZ,
                    control_hz=CONTROL_HZ,
                    physics_steps_per_control=PHYSICS_STEPS_PER_CONTROL,
                    physics_paused=False,
                    simulation_time_s=float(sim.physics_manager.get_simulation_time()),
                    controls=CONTROLS,
                )

            def continue_dynamic_control(reason: str, *, reset_pose: bool = False) -> None:
                nonlocal phase, settle_steps, stable_steps, paused_sim_time, active_signature
                keyboard.release_actions()
                term.reset()
                capsule.permanent_wrench_composer.reset()
                if reset_pose:
                    _write_state(capsule, default_pose, base.device)
                    sim.forward()
                    base.scene.update(0.0)
                settle_steps = stable_steps = 0
                paused_sim_time = None
                active_signature = None
                phase = "DYNAMIC_CONTROL"
                sim.physics_manager.play()
                session.emit(
                    "ENTRY_DYNAMIC_CONTROL_RESUMED",
                    reason=reason,
                    reset_to_default=reset_pose,
                    pose_world_xyzw=_pose(capsule).tolist(),
                    simulation_time_s=float(sim.physics_manager.get_simulation_time()),
                )

            def run_control_cycle(
                mode: ParameterizedForceMode,
                alpha: float,
                incoming_stable_steps: int | None = None,
            ) -> tuple[np.ndarray, np.ndarray, int | None]:
                action = torch.tensor(
                    [[float(mode), float(alpha)]], device=base.device, dtype=torch.float32
                )
                base.action_manager.process_action(action)
                pose = _pose(capsule)
                velocity = _velocity(capsule)
                continuous_stable_steps = incoming_stable_steps
                for substep in range(PHYSICS_STEPS_PER_CONTROL):
                    base.action_manager.apply_action()
                    base.scene.write_data_to_sim()
                    sim.step(render=False)
                    if substep % render_interval == 0:
                        sim.render()
                    base.scene.update(PHYSICS_DT_S)
                    pose = _pose(capsule)
                    velocity = _velocity(capsule)
                    if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
                        break
                    if continuous_stable_steps is not None:
                        linear_speed = float(np.linalg.norm(velocity[:3]))
                        angular_speed = float(np.linalg.norm(velocity[3:]))
                        continuous_stable_steps = (
                            continuous_stable_steps + 1
                            if linear_speed <= MAX_LINEAR_SPEED_M_S
                            and angular_speed <= MAX_ANGULAR_SPEED_RAD_S
                            else 0
                        )
                return pose, velocity, continuous_stable_steps

            while simulation_app.is_running():
                if phase in ("DYNAMIC_CONTROL", "DYNAMIC_SETTLE"):
                    exit_requested = False
                    while keyboard.presses:
                        key = keyboard.presses.popleft()
                        if _is_exit(key):
                            exit_requested = True
                            break
                        if phase == "DYNAMIC_CONTROL" and key == "ENTER":
                            release_pose = _pose(capsule).copy()
                            release_velocity = _velocity(capsule).copy()
                            keyboard.release_actions()
                            settle_steps = stable_steps = 0
                            release_start_sim_time = float(
                                sim.physics_manager.get_simulation_time()
                            )
                            phase = "DYNAMIC_SETTLE"
                            active_signature = None
                            session.emit(
                                "ENTRY_SETTLE_STARTED",
                                release_pose_world_xyzw=release_pose.tolist(),
                                release_velocity_world=release_velocity.tolist(),
                                initial_velocity_zero=False,
                                all_active_forces_cleared_at_boundary=True,
                                simulation_time_s=release_start_sim_time,
                            )
                        elif phase == "DYNAMIC_CONTROL" and key in ("R", "BACKSPACE"):
                            continue_dynamic_control("operator_reset", reset_pose=True)
                        elif phase == "DYNAMIC_SETTLE" and key == "BACKSPACE":
                            continue_dynamic_control("operator_cancelled_settle")
                    if exit_requested:
                        session.emit("ENTRY_CALIBRATION_EXITED", phase=phase)
                        return 0

                    if phase == "DYNAMIC_CONTROL":
                        mode, alpha = keyboard.force.command
                    else:
                        mode, alpha = ParameterizedForceMode.HOLD, keyboard.force.alpha
                    signature = (int(mode), float(alpha))
                    if signature != active_signature:
                        session.emit(
                            "ENTRY_CONTROL_CHANGED",
                            mode=mode.name,
                            mode_id=int(mode),
                            alpha=float(alpha),
                            physics_steps_per_control=PHYSICS_STEPS_PER_CONTROL,
                        )
                        active_signature = signature

                    cycle_started = time.perf_counter()
                    pose, velocity, observed_stable_steps = run_control_cycle(
                        mode,
                        alpha,
                        stable_steps if phase == "DYNAMIC_SETTLE" else None,
                    )
                    remaining = 1.0 / CONTROL_HZ - (time.perf_counter() - cycle_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
                    if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
                        continue_dynamic_control("non_finite_state", reset_pose=True)
                        continue
                    linear_speed = float(np.linalg.norm(velocity[:3]))
                    angular_speed = float(np.linalg.norm(velocity[3:]))
                    if phase == "DYNAMIC_CONTROL":
                        panel.update_control(pose[:3], mode, alpha, term.last_telemetry)
                        now = time.monotonic()
                        if now - last_status_wall >= 1.0:
                            last_status_wall = now
                            session.emit(
                                "ENTRY_DYNAMIC_STATE",
                                pose_world_xyzw=pose.tolist(),
                                linear_speed_m_s=linear_speed,
                                angular_speed_rad_s=angular_speed,
                                mode=mode.name,
                                alpha=float(alpha),
                                physics_paused=False,
                            )
                        continue

                    settle_steps += PHYSICS_STEPS_PER_CONTROL
                    # Stability was evaluated at every 240 Hz physical substep,
                    # while HOLD occupied the complete 0.1 s control period.
                    stable_steps = int(observed_stable_steps)
                    elapsed = settle_steps * PHYSICS_DT_S
                    panel.update_settle(elapsed, linear_speed, angular_speed, stable_steps * PHYSICS_DT_S)
                    if stable_steps >= STABLE_STEPS:
                        sim.physics_manager.pause()
                        settled_pose = pose.copy()
                        phase = "AWAIT_ANCHOR_CONFIRMATION"
                        paused_sim_time = float(sim.physics_manager.get_simulation_time())
                        session.emit(
                            "ENTRY_STABLE",
                            elapsed_s=elapsed,
                            stable_duration_s=stable_steps * PHYSICS_DT_S,
                            linear_speed_m_s=linear_speed,
                            angular_speed_rad_s=angular_speed,
                            settled_pose_world_xyzw=settled_pose.tolist(),
                            simulation_time_s=paused_sim_time,
                            instruction="Y=accept, Backspace=continue control",
                        )
                        panel.status.text = "Stable. Press Y to accept or Backspace to continue control."
                    elif settle_steps >= MAX_SETTLE_STEPS:
                        continue_dynamic_control("not_stable_within_2_seconds")
                    continue

                # Confirmation, region selection and completion remain paused.
                sim.render()
                current_sim_time = float(sim.physics_manager.get_simulation_time())
                if abs(current_sim_time - paused_sim_time) > 1.0e-12:
                    raise RuntimeError(
                        "simulation time advanced during paused calibration: "
                        f"{paused_sim_time} -> {current_sim_time}"
                    )

                while keyboard.presses:
                    key = keyboard.presses.popleft()
                    if _is_exit(key):
                        session.emit("ENTRY_CALIBRATION_EXITED", phase=phase)
                        return 0
                    if phase == "AWAIT_ANCHOR_CONFIRMATION" and key == "Y":
                        final_velocity = _velocity(capsule)
                        stable_detection = {
                            "result": "stable",
                            "required_continuous_duration_s": STABLE_DURATION_S,
                            "observed_continuous_duration_s": stable_steps * PHYSICS_DT_S,
                            "maximum_wait_s": MAX_SETTLE_S,
                            "release_elapsed_s": settle_steps * PHYSICS_DT_S,
                            "linear_speed_limit_m_s": MAX_LINEAR_SPEED_M_S,
                            "angular_speed_limit_rad_s": MAX_ANGULAR_SPEED_RAD_S,
                            "final_linear_speed_m_s": float(np.linalg.norm(final_velocity[:3])),
                            "final_angular_speed_rad_s": float(np.linalg.norm(final_velocity[3:])),
                            "release_start_simulation_time_s": release_start_sim_time,
                            "settled_simulation_time_s": paused_sim_time,
                            "positioning_controller": "parameterized_force_10hz",
                            "physics_hz": PHYSICS_HZ,
                            "control_hz": CONTROL_HZ,
                            "physics_steps_per_control": PHYSICS_STEPS_PER_CONTROL,
                        }
                        anchor = anchor_record(
                            default_pose_xyzw=default_pose,
                            release_pose_xyzw=release_pose,
                            settled_pose_xyzw=settled_pose,
                            stable_detection=stable_detection,
                            stomach_geometry_sha256=reference.geometry_sha256,
                            capsule_asset_identifier=CAPSULE_ASSET_IDENTIFIER,
                        )
                        save_and_reload(anchor_path, anchor)
                        closest = nearest_surface_point(reference, settled_pose[:3])
                        adjacency = shared_edge_adjacency(reference)
                        distances = geodesic_face_distances(
                            adjacency, closest.triangle_index
                        )
                        region = surface_region_from_distances(
                            reference, distances, ENTRY_RADII_M[radius_index]
                        )
                        geometry.update(reference, closest, region)
                        panel.update_region(closest, region)
                        panel.status.text = "Anchor saved. Select radius, then press Enter."
                        phase = "GEODESIC_REGION"
                        session.emit(
                            "ENTRY_ANCHOR_SAVED",
                            path=str(anchor_path.resolve()),
                            config_sha256=anchor["config_sha256"],
                            stomach_geometry_sha256=reference.geometry_sha256,
                            seed_triangle_index=closest.triangle_index,
                            closest_surface_point_world_m=closest.point_world_m.tolist(),
                            capsule_to_surface_distance_m=closest.distance_m,
                            reload_validated=True,
                        )
                    elif phase == "AWAIT_ANCHOR_CONFIRMATION" and key == "BACKSPACE":
                        continue_dynamic_control("operator_rejected_stable_pose")
                    elif phase == "GEODESIC_REGION" and _is_left_bracket(key):
                        radius_index = max(0, radius_index - 1)
                        region = surface_region_from_distances(
                            reference, distances, ENTRY_RADII_M[radius_index]
                        )
                        geometry.update(reference, closest, region)
                        panel.update_region(closest, region)
                        session.emit(
                            "ENTRY_REGION_PREVIEW",
                            radius_m=region.radius_m,
                            triangle_count=len(region.triangle_indices),
                            vertex_count=len(region.vertex_indices),
                            area_m2=region.area_m2,
                            connected_components=region.connected_components,
                        )
                    elif phase == "GEODESIC_REGION" and _is_right_bracket(key):
                        radius_index = min(len(ENTRY_RADII_M) - 1, radius_index + 1)
                        region = surface_region_from_distances(
                            reference, distances, ENTRY_RADII_M[radius_index]
                        )
                        geometry.update(reference, closest, region)
                        panel.update_region(closest, region)
                        session.emit(
                            "ENTRY_REGION_PREVIEW",
                            radius_m=region.radius_m,
                            triangle_count=len(region.triangle_indices),
                            vertex_count=len(region.vertex_indices),
                            area_m2=region.area_m2,
                            connected_components=region.connected_components,
                        )
                    elif phase == "GEODESIC_REGION" and key == "ENTER":
                        if region.connected_components != 1:
                            panel.status.text = "ERROR: region is not one connected component."
                            session.emit(
                                "ENTRY_REGION_SAVE_REJECTED",
                                reason="multiple_connected_components",
                                connected_components=region.connected_components,
                            )
                            continue
                        record = region_record(
                            anchor_config_sha256=anchor["config_sha256"],
                            settled_pose_xyzw=settled_pose,
                            closest=closest,
                            region=region,
                            stomach_geometry_sha256=reference.geometry_sha256,
                        )
                        save_and_reload(region_path, record)
                        phase = "COMPLETE"
                        panel.status.text = "Anchor and region saved and reloaded successfully."
                        session.emit(
                            "ENTRY_REGION_SAVED",
                            path=str(region_path.resolve()),
                            config_sha256=record["config_sha256"],
                            anchor_config_sha256=anchor["config_sha256"],
                            radius_m=region.radius_m,
                            seed_triangle_index=closest.triangle_index,
                            triangle_count=len(region.triangle_indices),
                            vertex_count=len(region.vertex_indices),
                            area_m2=region.area_m2,
                            connected_components=region.connected_components,
                            stomach_geometry_sha256=reference.geometry_sha256,
                            reload_validated=True,
                            status="complete",
                        )

        finally:
            if keyboard is not None:
                keyboard.close()
            if env is not None:
                env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
