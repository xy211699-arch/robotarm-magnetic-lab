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
    ENTRY_RADII_M,
    anchor_record,
    geodesic_face_distances,
    nearest_surface_point,
    region_record,
    save_and_reload,
    shared_edge_adjacency,
    surface_region_from_distances,
)
from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage
from robotarm_magnetic_lab.teleop.atomic_keyboard import normalize_key
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_stomach_env_cfg import (
    STOMACH_SCENE_USD_PATH,
)


MOVE_SPEED_M_S = 0.010
FINE_MOVE_SPEED_M_S = 0.002
PHYSICS_DT_S = 1.0 / 240.0
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
    "POSITION: W/S=world +/-Y, A/D=world -/+X, LeftShift=fine, Enter=release | "
    "SETTLED: Y=accept, Backspace=reject | REGION: [ / ]=radius, Enter=save | Esc=exit"
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
        self._input = carb.input.acquire_input_interface()
        self._device = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._device, self._on_event
        )

    def _on_event(self, event, *_args):
        key = normalize_key(event.input.name)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key not in self.down:
                self.presses.append(key)
            self.down.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.down.discard(key)
        return True

    @property
    def fine(self) -> bool:
        return any("SHIFT" in key for key in self.down)

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._device, self._subscription)
            self._subscription = None
        self.down.clear()


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

    def update_position(self, phase: str, position, offset, speed, paused: bool) -> None:
        self.phase.text = f"Phase: {phase}"
        self.state.text = (
            f"candidate_world_m={np.round(position, 6).tolist()} "
            f"horizontal_offset_m={np.round(offset, 6).tolist()} "
            f"move_speed={speed:.3f} m/s physics_paused={paused}"
        )

    def update_settle(self, elapsed, linear_speed, angular_speed, stable_time) -> None:
        self.phase.text = "Phase: NATURAL_DROP"
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
            candidate_pose = default_pose.copy()
            release_pose = None
            settled_pose = None
            anchor = None
            closest = distances = region = None
            radius_index = 2
            geometry = RegionDebugGeometry()
            keyboard = Keyboard()
            panel = Panel()
            term.reset()
            capsule.permanent_wrench_composer.reset()

            # Pause only PhysX. The Kit visualizer remains active so direct
            # Dynamic-body pose writes are visible without advancing sim time.
            sim.physics_manager.pause()
            paused_sim_time = float(sim.physics_manager.get_simulation_time())
            phase = "POSITION"
            fall_steps = stable_steps = 0
            release_start_sim_time = None
            last_wall = time.monotonic()
            last_status_wall = 0.0
            session.emit(
                "ENTRY_POSITION_READY",
                default_pose_world_xyzw=default_pose.tolist(),
                physics_paused=True,
                simulation_time_s=paused_sim_time,
                controls=CONTROLS,
            )

            def restore_release(reason: str) -> None:
                nonlocal phase, candidate_pose, fall_steps, stable_steps, paused_sim_time
                sim.physics_manager.pause()
                candidate_pose = release_pose.copy()
                _write_state(capsule, candidate_pose, base.device)
                sim.forward()
                base.scene.update(0.0)
                paused_sim_time = float(sim.physics_manager.get_simulation_time())
                fall_steps = stable_steps = 0
                phase = "POSITION"
                session.emit(
                    "ENTRY_RELEASE_REJECTED",
                    reason=reason,
                    restored_pose_world_xyzw=candidate_pose.tolist(),
                    simulation_time_s=paused_sim_time,
                )

            while simulation_app.is_running():
                now = time.monotonic()
                wall_dt = min(max(now - last_wall, 0.0), 0.05)
                last_wall = now

                if phase == "NATURAL_DROP":
                    base.scene.write_data_to_sim()
                    sim.step(render=False)
                    fall_steps += 1
                    if fall_steps % int(cfg.sim.render_interval) == 0:
                        sim.render()
                    base.scene.update(PHYSICS_DT_S)
                    pose = _pose(capsule)
                    velocity = _velocity(capsule)
                    if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
                        restore_release("non_finite_state")
                        continue
                    linear_speed = float(np.linalg.norm(velocity[:3]))
                    angular_speed = float(np.linalg.norm(velocity[3:]))
                    stable_steps = (
                        stable_steps + 1
                        if linear_speed <= MAX_LINEAR_SPEED_M_S
                        and angular_speed <= MAX_ANGULAR_SPEED_RAD_S
                        else 0
                    )
                    elapsed = fall_steps * PHYSICS_DT_S
                    panel.update_settle(
                        elapsed, linear_speed, angular_speed, stable_steps * PHYSICS_DT_S
                    )
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
                            instruction="Y=accept, Backspace=reject",
                        )
                        panel.status.text = "Stable. Press Y to accept or Backspace to reject."
                    elif fall_steps >= MAX_SETTLE_STEPS:
                        restore_release("not_stable_within_2_seconds")
                    continue

                # All remaining phases are PhysX-paused. Rendering pumps input
                # but cannot advance simulation time.
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
                    if phase == "POSITION" and key == "ENTER":
                        release_pose = candidate_pose.copy()
                        term.reset()
                        capsule.permanent_wrench_composer.reset()
                        _write_state(capsule, release_pose, base.device)
                        fall_steps = stable_steps = 0
                        release_start_sim_time = paused_sim_time
                        phase = "NATURAL_DROP"
                        sim.physics_manager.play()
                        session.emit(
                            "ENTRY_RELEASED",
                            release_pose_world_xyzw=release_pose.tolist(),
                            all_active_forces_cleared=True,
                            initial_velocity_zero=True,
                            simulation_time_s=release_start_sim_time,
                        )
                    elif phase == "AWAIT_ANCHOR_CONFIRMATION" and key == "Y":
                        final_velocity = _velocity(capsule)
                        stable_detection = {
                            "result": "stable",
                            "required_continuous_duration_s": STABLE_DURATION_S,
                            "observed_continuous_duration_s": stable_steps * PHYSICS_DT_S,
                            "maximum_wait_s": MAX_SETTLE_S,
                            "release_elapsed_s": fall_steps * PHYSICS_DT_S,
                            "linear_speed_limit_m_s": MAX_LINEAR_SPEED_M_S,
                            "angular_speed_limit_rad_s": MAX_ANGULAR_SPEED_RAD_S,
                            "final_linear_speed_m_s": float(np.linalg.norm(final_velocity[:3])),
                            "final_angular_speed_rad_s": float(np.linalg.norm(final_velocity[3:])),
                            "release_start_simulation_time_s": release_start_sim_time,
                            "settled_simulation_time_s": paused_sim_time,
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
                        restore_release("operator_rejected")
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

                if phase == "POSITION":
                    direction = np.asarray(
                        [
                            float("D" in keyboard.down) - float("A" in keyboard.down),
                            float("W" in keyboard.down) - float("S" in keyboard.down),
                        ],
                        dtype=np.float64,
                    )
                    norm = float(np.linalg.norm(direction))
                    speed = FINE_MOVE_SPEED_M_S if keyboard.fine else MOVE_SPEED_M_S
                    if norm > 0.0:
                        candidate_pose[:2] += speed * wall_dt * direction / norm
                        _write_state(capsule, candidate_pose, base.device)
                        sim.forward()
                        base.scene.update(0.0)
                    offset = candidate_pose[:2] - default_pose[:2]
                    panel.update_position(
                        phase, candidate_pose[:3], offset, speed, paused=True
                    )
                    if now - last_status_wall >= 1.0:
                        last_status_wall = now
                        session.emit(
                            "ENTRY_CANDIDATE",
                            position_world_m=candidate_pose[:3].tolist(),
                            horizontal_offset_from_default_m=offset.tolist(),
                            move_speed_m_s=speed,
                            physics_paused=True,
                            simulation_time_s=current_sim_time,
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

