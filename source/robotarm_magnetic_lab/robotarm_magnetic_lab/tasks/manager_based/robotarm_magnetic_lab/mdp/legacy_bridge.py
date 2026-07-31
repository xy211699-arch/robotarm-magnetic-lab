"""Isaac Lab bridge for the validated Isaac Sim magnetic and collision models.

The original extension remains the single source of truth for magnet geometry,
material parameters, streamline integration, and the cuMotion/XRDF collision
model.  This module deliberately loads only its pure computation files; it does
not start the old Kit extension or register its per-frame callbacks.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from datetime import datetime

import numpy as np
import torch
import yaml
from isaaclab.managers import ManagerTermBase


LEGACY_EXTENSION_ROOT = Path("/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim")
LAB_RUNTIME_LOG = Path("/mnt/isaac-linux/robotarm_magnetic_lab/logs/runtime.txt")


def _load_source_module(name: str, relative_path: str) -> ModuleType:
    """Load one legacy source file without importing its eager extension package."""
    path = LEGACY_EXTENSION_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert an Isaac Lab xyzw quaternion to a column-vector rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= limit or magnitude <= 1.0e-12:
        return vector
    return vector * (limit / magnitude)


def _empty_state(num_envs: int, device: str) -> dict[str, torch.Tensor]:
    return {
        "wrench": torch.zeros((num_envs, 12), device=device),
        "raw_wrench": torch.zeros((num_envs, 12), device=device),
        "asm_clearance": torch.full((num_envs, 1), math.inf, device=device),
        "collision": torch.zeros((num_envs, 1), dtype=torch.bool, device=device),
        "field_anchor": torch.full((num_envs, 3), math.nan, device=device),
    }


class XrdfSphereCollisionModel:
    """Runtime ASM-to-arm clearance using the original XRDF sphere geometry.

    Isaac Lab 3.0 in this installation does not expose the deprecated
    ``isaacsim.robot_motion.cumotion`` Python extension.  The required geometry
    is nevertheless completely described by the XRDF.  Using Isaac Lab's live
    body transforms avoids duplicating forward kinematics and gives the same
    signed sphere clearance used by the old planner.
    """

    def __init__(self, robot, xrdf_path: Path):
        with xrdf_path.open("r", encoding="utf-8") as stream:
            xrdf = yaml.safe_load(stream)
        geometry_name = xrdf["self_collision"]["geometry"]
        sphere_data = xrdf["geometry"][geometry_name]["spheres"]
        self.spheres = {
            frame: (
                np.asarray([item["center"] for item in items], dtype=np.float64),
                np.asarray([item["radius"] for item in items], dtype=np.float64),
            )
            for frame, items in sphere_data.items()
            if frame in robot.data.body_names
        }
        if "l6" not in self.spheres:
            raise ValueError(f"XRDF has no l6/ASM collision spheres: {xrdf_path}")
        self.body_indices = {
            frame: robot.data.body_names.index(frame) for frame in self.spheres
        }

    def clearance_by_frame(self, robot, env_id: int) -> dict[str, float]:
        body_positions = robot.data.body_pos_w.torch[env_id]
        body_quaternions = robot.data.body_quat_w.torch[env_id]

        def world_spheres(frame: str) -> tuple[np.ndarray, np.ndarray]:
            body_index = self.body_indices[frame]
            position = body_positions[body_index].detach().cpu().numpy()
            rotation = _quat_xyzw_to_matrix(
                body_quaternions[body_index].detach().cpu().numpy()
            )
            centers, radii = self.spheres[frame]
            return position + centers @ rotation.T, radii

        asm_centers, asm_radii = world_spheres("l6")
        result = {}
        for frame in self.spheres:
            if frame == "l6":
                continue
            frame_centers, frame_radii = world_spheres(frame)
            delta = asm_centers[:, None, :] - frame_centers[None, :, :]
            distances = np.linalg.norm(delta, axis=2)
            signed_clearance = (
                distances - asm_radii[:, None] - frame_radii[None, :]
            )
            result[frame] = float(np.min(signed_clearance))
        return result


class LegacyMagneticCollisionBridge(ManagerTermBase):
    """Apply analytical magnet forces and evaluate the original collision model."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        config_module = _load_source_module(
            "_robotarm_legacy_config", "robotarm/magnetic_sim/config.py"
        )
        field_module = _load_source_module(
            "_robotarm_legacy_field_models",
            "robotarm/magnetic_sim/magnetics/field_models.py",
        )
        streamline_module = _load_source_module(
            "_robotarm_legacy_streamlines",
            "robotarm/magnetic_sim/magnetics/streamlines.py",
        )
        visual_module = _load_source_module(
            "_robotarm_legacy_field_visualization",
            "robotarm/magnetic_sim/visualization/magnetic_field.py",
        )

        self.config = config_module.load_config(LEGACY_EXTENSION_ROOT)
        self.model = field_module.FiniteMagnetSystem(self.config)
        self.collision_model = XrdfSphereCollisionModel(
            env.scene["robot"],
            LEGACY_EXTENSION_ROOT / self.config["planning"]["robot_xrdf"],
        )
        self._trace_streamlines = streamline_module.trace_streamlines
        self._create_or_update_streamlines = visual_module.create_or_update_streamlines

        self.robot = env.scene["robot"]
        self.capsule = env.scene["capsule"]
        self.arm_indices = [
            self.robot.data.joint_names.index(name)
            for name in self.config["robot"]["arm_joint_names"]
        ]
        self.magnet_body_index = self.robot.data.body_names.index("magl")
        self.base_body_index = self.robot.data.body_names.index("base_link")
        self.elapsed = torch.zeros(env.num_envs, device=env.device)
        self.frame_count = 0
        self._field_last_position = np.full(3, np.nan)
        self.state = _empty_state(env.num_envs, env.device)
        self._filtered_wrench = torch.zeros((env.num_envs, 12), device=env.device)
        env._legacy_bridge_state = self.state

        self._local_streamlines = self._build_local_streamlines()
        LAB_RUNTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
        with LAB_RUNTIME_LOG.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{datetime.now().isoformat(timespec='seconds')} "
                "==================== LAB_SESSION_START ====================\n"
            )
        self._log(
            "LAB_BRIDGE_READY "
            f"magnet_body_index={self.magnet_body_index} "
            f"arm_indices={self.arm_indices} "
            f"streamlines={len(self._local_streamlines)}"
        )

    def _log(self, message: str) -> None:
        line = (
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"[robotarm.magnetic_lab] {message}"
        )
        print(line, flush=True)
        try:
            with LAB_RUNTIME_LOG.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError:
            pass

    def _build_local_streamlines(self) -> list[np.ndarray]:
        cube_cfg = self.config["magnets"]["main_cube"]

        def local_field(points):
            return self.model.field_tesla(points, np.zeros(3), np.eye(3))

        return self._trace_streamlines(
            local_field,
            cube_cfg["dimensions_m"],
            self.config["visualization"],
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.elapsed[env_ids] = 0.0
        self.state["wrench"][env_ids] = 0.0
        self.state["raw_wrench"][env_ids] = 0.0
        self._filtered_wrench[env_ids] = 0.0
        self.state["asm_clearance"][env_ids] = math.inf
        self.state["collision"][env_ids] = False
        self.state["field_anchor"][env_ids] = math.nan
        self.robot.permanent_wrench_composer.reset(env_ids=env_ids)
        self.capsule.permanent_wrench_composer.reset(env_ids=env_ids)

    def _base_from_world(
        self,
        position_world: np.ndarray,
        rotation_world: np.ndarray,
        base_position_world: np.ndarray,
        base_rotation_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        base_rotation_inverse = base_rotation_world.T
        return (
            base_rotation_inverse @ (position_world - base_position_world),
            base_rotation_inverse @ rotation_world,
        )

    def _update_field_visualization(
        self,
        env_id: int,
        magnet_position: np.ndarray,
        magnet_rotation: np.ndarray,
    ) -> None:
        # This single-environment bring-up task updates the debug curves at the
        # policy rate (20 Hz). The previous 2 Hz throttle produced a visible
        # field/ball separation of up to centimetres during arm motion.
        update_period = 1
        if self.frame_count % update_period != 0:
            return
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            path = f"/World/envs/env_{env_id}/Scene/MagneticDemo/field_vectors"
            world_lines = [
                magnet_position + line @ magnet_rotation.T
                for line in self._local_streamlines
            ]
            self._create_or_update_streamlines(
                stage,
                path,
                world_lines,
                float(self.config["visualization"]["line_width_m"]),
            )
            # Magnetic lines are diagnostic overlays, not physical light.
            # Mark them as guide geometry so RTX camera observations do not
            # contain information unavailable to the real capsule camera.
            field_prim = stage.GetPrimAtPath(path)
            if field_prim.IsValid():
                UsdGeom.Imageable(field_prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            self._field_last_position = magnet_position.copy()
        except Exception as error:
            if self.frame_count == 0:
                self._log(f"FIELD_VISUALIZATION_DISABLED reason={error!r}")

    def __call__(self, env, env_ids) -> None:
        """Keep the manager event alive; physics work runs in the action term.

        Isaac Lab invokes interval events only once per policy step. Magnetic
        forces must instead be refreshed by ``MagneticPhysicsAction`` from the
        action manager, whose ``apply_actions`` hook runs once per 240 Hz
        simulation substep.
        """
        return

    def physics_step(self, env, env_ids=None) -> None:
        """Refresh magnetic wrench at the PhysX simulation frequency."""
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_id_list = env_ids.detach().cpu().tolist()
        self.elapsed[env_ids] += float(env.physics_dt)
        self.frame_count += 1
        diagnostic_stride = max(
            int(round(float(env.step_dt) / float(env.physics_dt))), 1
        )
        diagnostics_due = self.frame_count % diagnostic_stride == 0

        magnet_positions = self.robot.data.body_pos_w.torch[:, self.magnet_body_index]
        magnet_quaternions = self.robot.data.body_quat_w.torch[:, self.magnet_body_index]
        base_positions = self.robot.data.body_pos_w.torch[:, self.base_body_index]
        base_quaternions = self.robot.data.body_quat_w.torch[:, self.base_body_index]
        capsule_positions = self.capsule.data.root_pos_w.torch
        capsule_quaternions = self.capsule.data.root_quat_w.torch
        capsule_linear_velocities = self.capsule.data.root_lin_vel_w.torch
        capsule_angular_velocities = self.capsule.data.root_ang_vel_w.torch
        joint_positions = self.robot.data.joint_pos.torch

        force_on_robot = torch.zeros((len(env_id_list), 1, 3), device=env.device)
        torque_on_robot = torch.zeros_like(force_on_robot)
        force_on_capsule = torch.zeros_like(force_on_robot)
        torque_on_capsule = torch.zeros_like(force_on_robot)
        force_position_robot = torch.zeros_like(force_on_robot)
        force_position_capsule = torch.zeros_like(force_on_robot)

        sim_cfg = self.config["simulation"]
        cylinder_cfg = self.config["magnets"]["target_cylinder"]
        capsule_cfg = self.config["external_magnet"]["capsule"]
        ramp_seconds = max(
            float(self.config["external_magnet"]["coupling_ramp_s"]),
            1.0e-6,
        )
        release_delay = float(self.config["external_magnet"]["release_delay_s"])

        for local_index, env_id in enumerate(env_id_list):
            main_position = magnet_positions[env_id].detach().cpu().numpy().astype(np.float64)
            main_rotation = _quat_xyzw_to_matrix(
                magnet_quaternions[env_id].detach().cpu().numpy()
            )
            capsule_position = (
                capsule_positions[env_id].detach().cpu().numpy().astype(np.float64)
            )
            capsule_rotation = _quat_xyzw_to_matrix(
                capsule_quaternions[env_id].detach().cpu().numpy()
            )
            capsule_linear_velocity = (
                capsule_linear_velocities[env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            capsule_angular_velocity = (
                capsule_angular_velocities[env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            offset_local = np.array(
                [0.0, 0.0, float(cylinder_cfg["center_offset_axis_m"])],
                dtype=np.float64,
            )
            target_magnet_position = capsule_position + capsule_rotation @ offset_local

            robot_force, robot_torque = self.model.force_torque_on_cube_si(
                target_magnet_position,
                capsule_rotation,
                main_position,
                main_rotation,
            )
            capsule_force, capsule_torque = self.model.force_torque_si(
                main_position,
                main_rotation,
                target_magnet_position,
                capsule_rotation,
            )
            raw_wrench = np.concatenate(
                (robot_force, robot_torque, capsule_force, capsule_torque)
            )
            self.state["raw_wrench"][env_id] = torch.as_tensor(
                raw_wrench, device=env.device, dtype=torch.float32
            )
            robot_force = _clip_norm(robot_force, float(sim_cfg["max_force_n"]))
            robot_torque = _clip_norm(robot_torque, float(sim_cfg["max_torque_nm"]))
            capsule_force = _clip_norm(capsule_force, float(sim_cfg["max_force_n"]))
            capsule_torque = _clip_norm(capsule_torque, float(sim_cfg["max_torque_nm"]))
            ramp = min(
                max((float(self.elapsed[env_id]) - release_delay) / ramp_seconds, 0.0),
                1.0,
            )
            # Lumped viscous resistance of fluid, mucosa and rolling contact.
            # This is a passive force opposing measured motion, not an
            # actuator or commanded capsule velocity.
            capsule_force_total = capsule_force * ramp - float(
                capsule_cfg.get("linear_drag_n_per_m_s", 0.0)
            ) * capsule_linear_velocity
            capsule_torque_total = capsule_torque * ramp - float(
                capsule_cfg.get("angular_drag_nm_per_rad_s", 0.0)
            ) * capsule_angular_velocity
            capsule_force_total = _clip_norm(
                capsule_force_total, float(sim_cfg["max_force_n"])
            )
            capsule_torque_total = _clip_norm(
                capsule_torque_total, float(sim_cfg["max_torque_nm"])
            )

            target_wrench = torch.as_tensor(
                np.concatenate(
                    (
                        robot_force * ramp,
                        robot_torque * ramp,
                        capsule_force_total,
                        capsule_torque_total,
                    )
                ),
                device=env.device,
                dtype=torch.float32,
            )
            filter_tau = max(
                float(sim_cfg.get("wrench_filter_time_constant_s", 0.0)), 0.0
            )
            filter_alpha = (
                1.0
                if filter_tau <= 0.0
                else 1.0 - math.exp(-float(env.physics_dt) / filter_tau)
            )
            self._filtered_wrench[env_id] += filter_alpha * (
                target_wrench - self._filtered_wrench[env_id]
            )
            applied_wrench = self._filtered_wrench[env_id]
            force_on_robot[local_index, 0] = applied_wrench[0:3]
            torque_on_robot[local_index, 0] = applied_wrench[3:6]
            force_on_capsule[local_index, 0] = applied_wrench[6:9]
            torque_on_capsule[local_index, 0] = applied_wrench[9:12]
            force_position_robot[local_index, 0] = magnet_positions[env_id]
            force_position_capsule[local_index, 0] = torch.as_tensor(
                target_magnet_position, device=env.device, dtype=torch.float32
            )
            self.state["wrench"][env_id] = applied_wrench

            if diagnostics_due:
                clearances = self.collision_model.clearance_by_frame(self.robot, env_id)
                minimum_clearance = min(clearances.values()) if clearances else math.inf
                self.state["asm_clearance"][env_id, 0] = minimum_clearance
                self.state["collision"][env_id, 0] = minimum_clearance < 0.0
                self._update_field_visualization(env_id, main_position, main_rotation)
                if np.isfinite(self._field_last_position).all():
                    self.state["field_anchor"][env_id] = torch.as_tensor(
                        self._field_last_position,
                        device=env.device,
                        dtype=torch.float32,
                    )

        if bool(sim_cfg["apply_forces"]):
            self.robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=force_on_robot,
                torques=torque_on_robot,
                positions=force_position_robot,
                body_ids=[self.magnet_body_index],
                env_ids=env_ids,
                is_global=True,
            )
            self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
                forces=force_on_capsule,
                torques=torque_on_capsule,
                positions=force_position_capsule,
                body_ids=[0],
                env_ids=env_ids,
                is_global=True,
            )

        log_stride = max(int(round(1.0 / float(env.physics_dt))), 1)
        if self.frame_count % log_stride == 0:
            for env_id in env_id_list:
                wrench = self.state["wrench"][env_id].detach().cpu().numpy()
                raw_wrench = self.state["raw_wrench"][env_id].detach().cpu().numpy()
                self._log(
                    "MAGNETIC_COLLISION_STATUS "
                    f"env={env_id} "
                    f"force_robot_N={np.round(wrench[:3], 6).tolist()} "
                    f"torque_robot_Nm={np.round(wrench[3:6], 6).tolist()} "
                    f"force_capsule_N={np.round(wrench[6:9], 6).tolist()} "
                    f"torque_capsule_Nm={np.round(wrench[9:12], 6).tolist()} "
                    f"raw_force_capsule_N={np.round(raw_wrench[6:9], 6).tolist()} "
                    f"raw_torque_capsule_Nm={np.round(raw_wrench[9:12], 6).tolist()} "
                    f"asm_clearance_m={float(self.state['asm_clearance'][env_id, 0]):.6f} "
                    f"collision={bool(self.state['collision'][env_id, 0])} "
                    f"field_anchor_m={np.round(self._field_last_position, 6).tolist()} "
                    f"magnet_position_m={np.round(magnet_positions[env_id].detach().cpu().numpy(), 6).tolist()}"
                )


def magnetic_wrench(env) -> torch.Tensor:
    """Robot and capsule magnetic force/torque in world coordinates."""
    state = getattr(env, "_legacy_bridge_state", None)
    if state is None:
        return torch.zeros((env.num_envs, 12), device=env.device)
    return state["wrench"]


def asm_clearance(env) -> torch.Tensor:
    """Minimum XRDF sphere clearance between mounted ASM and robot links."""
    state = getattr(env, "_legacy_bridge_state", None)
    if state is None:
        return torch.full((env.num_envs, 1), 1.0, device=env.device)
    return torch.nan_to_num(state["asm_clearance"], posinf=1.0)


def collision_penalty(env) -> torch.Tensor:
    """Binary penalty for self/world collision according to the legacy model."""
    state = getattr(env, "_legacy_bridge_state", None)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    return state["collision"].squeeze(-1).float()


def collision_detected(env) -> torch.Tensor:
    """Terminate a training episode when the safety model reports collision."""
    state = getattr(env, "_legacy_bridge_state", None)
    if state is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return state["collision"].squeeze(-1)
