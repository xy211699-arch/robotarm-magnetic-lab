"""Isaac Lab adapter for the scalar atomic magnetic action interface."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from ..controllers.action_layer import (
    ActionLayerConfig,
    AtomicActionExecutor,
    AtomicCommandPlanner,
    DeviceSnapshot,
    HardSafetyMonitor,
    UrdfXrdfSafetyModel,
    initial_command_state,
)
from ..controllers.table_motion import BallFieldPlanner, arm_gradient_plan


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class AtomicMagnetAction(ActionTerm):
    """Expose one integer action and execute its 20 Hz trajectory internally."""

    cfg: "AtomicMagnetActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("Stage-one AtomicMagnetAction supports exactly one environment")
        self.robot = env.scene[cfg.asset_name]
        self.joint_names = tuple(cfg.arm_joint_names) + tuple(cfg.ball_joint_names)
        self.joint_indices = [self.robot.data.joint_names.index(name) for name in self.joint_names]
        self.arm_indices = self.joint_indices[: len(cfg.arm_joint_names)]
        self.ball_indices = self.joint_indices[len(cfg.arm_joint_names) :]
        self.magnet_body_index = self.robot.data.body_names.index(cfg.magnet_body_name)
        self._raw_actions = torch.zeros((env.num_envs, 1), device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target = self.robot.data.joint_pos.torch[:, self.joint_indices].clone()
        self._executor = None
        self._field_planner = None
        self._request_id = 0
        self._last_action_id = None
        self._accepting_request = True
        self._previous_velocity = np.zeros(len(self.joint_names), dtype=np.float64)

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
    def executor(self):
        return self._executor

    @property
    def last_result(self):
        return None if self._executor is None else self._executor.last_result

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        action_ids = torch.round(actions).to(dtype=torch.int64)
        self._processed_actions[:] = action_ids.to(dtype=actions.dtype)
        self._ensure_runtime()
        snapshot = self._snapshot()
        assert self._executor is not None
        if (
            self._accepting_request
            and not self._executor.busy
            and self._executor.state.value in ("IDLE", "DONE")
        ):
            action_id = int(action_ids[0, 0].item())
            # A new decision is accepted only at an action boundary. Repeating
            # the same ID after DONE is still a valid new SMDP request.
            self._request_id += 1
            self._executor.submit(action_id, snapshot, self._request_id)
            self._last_action_id = action_id
            self._accepting_request = False
        output = self._executor.step(snapshot)
        self._target[0] = torch.as_tensor(
            output.joint_target_rad, device=self._env.device, dtype=torch.float32
        )

    def apply_actions(self) -> None:
        self.robot.set_joint_position_target_index(
            target=self._target,
            joint_ids=self.joint_indices,
        )

    def reset(self, env_ids=None) -> None:
        if self._executor is None:
            return
        snapshot = self._snapshot()
        direction = self._field_planner.field_world(snapshot.joint_position_rad[-3:])
        command = initial_command_state(snapshot, direction)
        self._executor.reset(snapshot, command)
        self._target[0] = torch.as_tensor(
            snapshot.joint_position_rad, device=self._env.device, dtype=torch.float32
        )
        self._previous_velocity[:] = 0.0
        self._accepting_request = True

    def action_mask(self) -> np.ndarray:
        self._ensure_runtime()
        return self._executor.action_mask(self._snapshot()).values.copy()

    def acknowledge_result(self) -> None:
        """Release a DONE boundary so an SMDP wrapper may submit the next ID."""
        if self._executor is not None and self._executor.state.value == "DONE":
            self._accepting_request = True

    def _ensure_runtime(self) -> None:
        if self._executor is not None:
            return
        bridge = self._env.event_manager.get_term_cfg("magnetic_collision_bridge").func
        q = self.robot.data.joint_pos.torch[0, self.joint_indices].detach().cpu().numpy()
        position = self.robot.data.body_pos_w.torch[0, self.magnet_body_index].detach().cpu().numpy()
        rotation = _quat_xyzw_to_matrix(
            self.robot.data.body_quat_w.torch[0, self.magnet_body_index].detach().cpu().numpy()
        )
        self._field_planner = BallFieldPlanner(
            bridge.model,
            position,
            rotation,
            q[-3:],
            np.asarray(self.cfg.registered_field_point_world_m, dtype=np.float64),
            q[-3:],
            action_scale_rad=math.pi,
        )
        layer_cfg = ActionLayerConfig(
            registered_field_point_world_m=tuple(self.cfg.registered_field_point_world_m)
        )

        def solve_ball(desired, current):
            self._field_planner.current = np.asarray(current, dtype=np.float64).copy()
            return self._field_planner.solve(desired, global_search=False)

        def field_for_ball(ball):
            field = self._field_planner.field_world(ball)
            return field / max(float(np.linalg.norm(field)), 1.0e-12)

        def solve_arm(_snapshot, displacement):
            jacobian_index = (
                self.magnet_body_index - 1 if self.robot.is_fixed_base else self.magnet_body_index
            )
            jacobian = (
                self.robot.data.body_link_jacobian_w.torch[
                    0, jacobian_index, :, self.arm_indices
                ]
                .detach()
                .cpu()
                .numpy()
            )
            return arm_gradient_plan(
                jacobian,
                displacement,
                action_scale_rad=0.25,
                max_joint_delta_rad=0.08,
                damping=2.0e-3,
                locked_joint_indices=(5,),
                orientation_weight=0.2,
            ).joint_delta_rad

        planner = AtomicCommandPlanner(
            layer_cfg,
            solve_ball_field=solve_ball,
            solve_arm_displacement=solve_arm,
            field_for_ball=field_for_ball,
        )
        kinematics = UrdfXrdfSafetyModel(
            self.cfg.urdf_path,
            self.cfg.xrdf_path,
            ignored_frames=tuple(self.cfg.ignored_collision_frames),
        )
        world_collision_checker = None
        if self.cfg.environment_collision_mesh_prim_path:
            from .world_collision import StomachMeshCollisionChecker

            base_body_index = self.robot.data.body_names.index(
                self.cfg.robot_base_body_name
            )
            base_position = (
                self.robot.data.body_pos_w.torch[0, base_body_index]
                .detach()
                .cpu()
                .numpy()
            )
            base_rotation = _quat_xyzw_to_matrix(
                self.robot.data.body_quat_w.torch[0, base_body_index]
                .detach()
                .cpu()
                .numpy()
            )
            world_collision_checker = StomachMeshCollisionChecker(
                kinematics,
                mesh_prim_path=self.cfg.environment_collision_mesh_prim_path,
                robot_base_position_world_m=base_position,
                robot_base_rotation_world=base_rotation,
                device=str(self._env.device),
                required_clearance_m=self.cfg.environment_collision_clearance_m,
                trajectory_samples=layer_cfg.trajectory_collision_samples,
            )
            initial_clearance = world_collision_checker.check_configuration(
                q[: len(self.cfg.arm_joint_names)]
            )
            print(
                "ATOMIC_WORLD_COLLISION_READY "
                f"mesh={self.cfg.environment_collision_mesh_prim_path} "
                f"initial_clearance_m={initial_clearance.clearance_m:.6f} "
                f"required_clearance_m={self.cfg.environment_collision_clearance_m:.6f} "
                f"frame={initial_clearance.frame}"
            )
        snapshot = self._snapshot()
        command = initial_command_state(snapshot, field_for_ball(q[-3:]))
        self._executor = AtomicActionExecutor(
            layer_cfg,
            planner,
            HardSafetyMonitor(
                layer_cfg,
                kinematics,
                validate_ground=False,
                world_collision_checker=world_collision_checker,
            ),
            command,
        )

    def _snapshot(self) -> DeviceSnapshot:
        position = self.robot.data.joint_pos.torch[0, self.joint_indices].detach().cpu().numpy()
        velocity = self.robot.data.joint_vel.torch[0, self.joint_indices].detach().cpu().numpy()
        acceleration = (velocity - self._previous_velocity) / float(self._env.step_dt)
        self._previous_velocity = velocity.copy()
        limits = self.robot.data.soft_joint_pos_limits.torch[0, self.joint_indices].detach().cpu().numpy()
        velocity_limits = self.robot.data.joint_vel_limits.torch[0, self.joint_indices].detach().cpu().numpy()
        magnet_position = self.robot.data.body_pos_w.torch[0, self.magnet_body_index].detach().cpu().numpy()
        magnet_rotation = _quat_xyzw_to_matrix(
            self.robot.data.body_quat_w.torch[0, self.magnet_body_index].detach().cpu().numpy()
        )
        bridge_state = getattr(self._env, "_legacy_bridge_state", None)
        clearance = math.inf
        if bridge_state is not None:
            clearance = float(bridge_state["asm_clearance"][0, 0].item())
        terminated = False
        manager = getattr(self._env, "termination_manager", None)
        if manager is not None and hasattr(manager, "terminated"):
            terminated = bool(manager.terminated[0].item())
        return DeviceSnapshot(
            sim_time_s=float(self._env.episode_length_buf[0].item()) * float(self._env.step_dt),
            joint_position_rad=position,
            joint_velocity_rad_s=velocity,
            joint_acceleration_rad_s2=acceleration,
            joint_position_limits_rad=limits,
            joint_velocity_limits_rad_s=velocity_limits,
            joint_acceleration_limits_rad_s2=np.asarray(
                ActionLayerConfig().fallback_acceleration_limits_rad_s2
            ),
            magnet_position_world_m=magnet_position,
            magnet_rotation_world=magnet_rotation,
            asm_clearance_m=clearance,
            controller_connected=True,
            environment_terminated=terminated,
        )


@configclass
class AtomicMagnetActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = AtomicMagnetAction
    arm_joint_names: tuple[str, ...] = ("j1", "j2", "j3", "j4", "j5", "j6")
    ball_joint_names: tuple[str, ...] = ("ballxj", "ballyj", "ballzj")
    magnet_body_name: str = "magl"
    urdf_path: str = "/home/multirobo/Desktop/robotarm/urdf/robotarm.urdf"
    xrdf_path: str = (
        "/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/data/planning/robot.xrdf"
    )
    ignored_collision_frames: tuple[str, ...] = ()
    robot_base_body_name: str = "base_link"
    environment_collision_mesh_prim_path: str | None = None
    # A 5 mm predictive buffer covers collision-sphere approximation error and
    # one low-speed control interval before the runtime hold takes effect.
    environment_collision_clearance_m: float = 0.005
    registered_field_point_world_m: tuple[float, float, float] = (
        1.0608155,
        0.1145374,
        0.0065,
    )


def external_magnet_state(env, asset_name: str = "robot", body_name: str = "magl"):
    """External-magnet world pose from device/FK state only."""
    robot = env.scene[asset_name]
    body_index = robot.data.body_names.index(body_name)
    return torch.cat(
        (
            robot.data.body_pos_w.torch[:, body_index],
            robot.data.body_quat_w.torch[:, body_index],
        ),
        dim=-1,
    )


def atomic_hard_failure(env, term_name: str = "atomic") -> torch.Tensor:
    """Terminate the episode after the executor enters failure containment."""
    term = env.action_manager.get_term(term_name)
    failed = (
        term.executor is not None
        and term.executor.state.value in ("HARD_FAILURE", "SAFE_RECOVER")
    )
    return torch.full(
        (env.num_envs,), failed, dtype=torch.bool, device=env.device
    )
