"""Isaac Lab ActionTerm for the privileged ideal-surface controller."""

from __future__ import annotations

import math

import numpy as np
import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from ..controllers.ideal_surface import (
    CapsulePose,
    ControllerSnapshot,
    IdealSurfaceConfig,
    IdealSurfaceController,
    LocalFrame,
    Spherocylinder,
    SurfaceFlags,
    SurfaceNavigationMesh,
    assess_pose,
    orientation_from_axis_and_image_up,
    quaternion_wxyz_to_matrix,
)


class IdealSurfaceActionTerm(ActionTerm):
    """Execute one discrete ideal capsule action over every 240 Hz substep."""

    cfg: "IdealSurfaceActionTermCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("IdealSurfaceActionTerm requires exactly one environment")
        self.capsule = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros((1, 1), device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._controller: IdealSurfaceController | None = None
        self._request_id = 0
        self._pending_reset = True
        self._enable_kinematic_capsule()

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def controller(self) -> IdealSurfaceController | None:
        return self._controller

    @property
    def last_result(self):
        return None if self._controller is None else self._controller.last_result

    def _enable_kinematic_capsule(self) -> None:
        """Author kinematic behavior only on this task's namespaced capsule."""
        import omni.usd
        from pxr import UsdPhysics

        prim_path = self.cfg.capsule_prim_path.format(env_index=0)
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"ideal capsule prim is unavailable: {prim_path}")
        api = UsdPhysics.RigidBodyAPI(prim)
        if not api:
            api = UsdPhysics.RigidBodyAPI.Apply(prim)
        attribute = api.GetKinematicEnabledAttr()
        if not attribute:
            attribute = api.CreateKinematicEnabledAttr()
        attribute.Set(True)

    def _ensure_runtime(self) -> None:
        if self._controller is not None:
            return
        from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage

        reference = reference_from_stage(self.cfg.surface_prim_path)
        if reference.geometry_sha256 != self.cfg.expected_surface_geometry_sha256:
            raise RuntimeError(
                "approved surface geometry hash changed: "
                f"expected={self.cfg.expected_surface_geometry_sha256} "
                f"observed={reference.geometry_sha256}"
            )
        mesh = SurfaceNavigationMesh.from_reference(reference, self.cfg.inward_normal_sign)
        config = IdealSurfaceConfig()
        capsule = Spherocylinder(config.capsule_radius_m, config.capsule_cylinder_half_length_m)
        self._controller = IdealSurfaceController(mesh, cfg=config, capsule=capsule)
        self._controller.reset(self._snapshot_from_live_capsule(idealize=True))
        self._pending_reset = False

    def _snapshot_from_live_capsule(self, *, idealize: bool) -> ControllerSnapshot:
        assert self._controller is not None
        position = self.capsule.data.root_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
        quaternion = self.capsule.data.root_quat_w.torch[0].detach().cpu().numpy().astype(np.float64)
        rotation = quaternion_wxyz_to_matrix(quaternion)
        live_axis = rotation @ np.asarray([0.0, 0.0, 1.0])
        live_image_up = rotation @ np.asarray([0.0, 1.0, 0.0])
        if self._controller.snapshot is None or self._pending_reset:
            hit = self._controller.mesh.closest_hit(position)
        else:
            previous = self._controller.snapshot
            hit = self._controller.mesh.advance(
                previous.surface_triangle_id,
                previous.surface_point_world,
                position - previous.position_world,
                self._controller.cfg.recovery_query_radius_scale * self._controller.capsule.radius_m,
            )
        normal = hit.normal_world
        tangent = live_axis - float(live_axis @ normal) * normal
        tangent_norm = float(np.linalg.norm(tangent))
        projected_up = live_image_up - float(live_image_up @ normal) * normal
        if float(np.linalg.norm(projected_up)) <= 1.0e-12:
            projected_up = tangent
        if float(np.linalg.norm(projected_up)) <= 1.0e-12:
            projected_up = np.asarray([0.0, 1.0, 0.0])
            projected_up -= float(projected_up @ normal) * normal
        frame = LocalFrame(hit.point_world, normal, projected_up)
        theta = min(
            math.acos(float(np.clip(live_axis @ normal, -1.0, 1.0))),
            math.pi / 2.0,
        )
        if tangent_norm <= 1.0e-12:
            phi = 0.0
            ideal_axis = normal.copy()
        else:
            tangent /= tangent_norm
            phi = math.atan2(float(tangent @ frame.e2), float(tangent @ frame.e1))
            ideal_axis = math.cos(theta) * normal + math.sin(theta) * frame.direction(phi)
        if idealize:
            position = hit.point_world + self._controller.capsule.support_distance(
                ideal_axis, normal
            ) * normal
            quaternion = orientation_from_axis_and_image_up(ideal_axis, live_image_up)
            rotation = quaternion_wxyz_to_matrix(quaternion)
            live_image_up = rotation @ np.asarray([0.0, 1.0, 0.0])
        pose_assessment = assess_pose(
            self._controller.mesh,
            self._controller.capsule,
            CapsulePose(position, ideal_axis, live_image_up),
            hit.triangle_id,
            self._controller.cfg,
        )
        return ControllerSnapshot(
            sim_time_s=float(self._env.episode_length_buf[0].item()) * float(self._env.step_dt),
            position_world=position,
            quaternion_for_sim=quaternion,
            axis_world=ideal_axis,
            image_up_world=live_image_up,
            surface_point_world=hit.point_world,
            surface_normal_world=normal,
            surface_triangle_id=hit.triangle_id,
            theta_rad=theta,
            phi_rad=phi,
            flags=SurfaceFlags(
                upright=theta <= self._controller.cfg.upright_enter_rad,
                side_contact=pose_assessment.side_contact,
                contact_limited=False,
                boundary_limited=False,
            ),
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        action_ids = torch.round(actions).to(torch.int64)
        self._processed_actions[:] = action_ids.to(actions.dtype)
        self._ensure_runtime()
        assert self._controller is not None
        if self._pending_reset:
            self._controller.reset(self._snapshot_from_live_capsule(idealize=True))
            self._pending_reset = False
        if self._controller.ready:
            self._request_id += 1
            self._controller.submit(
                int(action_ids[0, 0].item()), self._controller.snapshot, self._request_id
            )

    def apply_actions(self) -> None:
        self._ensure_runtime()
        assert self._controller is not None
        output = self._controller.step(float(self._env.sim.cfg.dt))
        pose = torch.as_tensor(
            np.concatenate((output.position_world, output.quaternion_for_sim)),
            device=self._env.device,
            dtype=torch.float32,
        ).reshape(1, 7)
        velocity = torch.as_tensor(
            np.concatenate((output.linear_velocity_world, output.angular_velocity_world)),
            device=self._env.device,
            dtype=torch.float32,
        ).reshape(1, 6)
        self.capsule.write_root_pose_to_sim(pose)
        self.capsule.write_root_velocity_to_sim(velocity)

    def reset(self, env_ids=None) -> None:
        del env_ids
        self._pending_reset = True

    def action_mask(self) -> np.ndarray:
        self._ensure_runtime()
        assert self._controller is not None
        return self._controller.action_mask().copy()

    def acknowledge_result(self) -> None:
        if self._controller is not None:
            self._controller.acknowledge_result()


@configclass
class IdealSurfaceActionTermCfg(ActionTermCfg):
    class_type: type[ActionTerm] = IdealSurfaceActionTerm
    asset_name: str = "capsule"
    capsule_prim_path: str = "/World/envs/env_{env_index}/Scene/MagneticDemo/target_magnet"
    surface_prim_path: str = (
        "/World/envs/env_0/Stomach/ConvertedSource/Environment/Stomach/VisualMesh/Stomach"
    )
    expected_surface_geometry_sha256: str = (
        "85ddd3e79438509364245c87be9a9564d1bf9ca29afb2c922fc013b2f7561d09"
    )
    inward_normal_sign: int = -1


def ideal_surface_hard_failure(env, term_name: str = "ideal_surface") -> torch.Tensor:
    term = env.action_manager.get_term(term_name)
    failed = bool(
        term.controller is not None
        and term.controller.state.value == "TERMINAL_FAULT"
    )
    return torch.full((env.num_envs,), failed, dtype=torch.bool, device=env.device)
