"""New RL schedules reuse the archived geometry, servo and stopping guards."""
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.utils.configclass import configclass

from .new_stomach_actuator_action import NewStomachActuatorVectorAction, NewStomachActuatorVectorActionCfg
from .actuator_vector_action import array
from ..controllers.new_stomach_rl_trajectory import JointState, build_schedule
from ..controllers.actuator_vector import SCALES, decode
from ..controllers.source_kinematics import source_pose


class NewStomachRLAction(NewStomachActuatorVectorAction):
    def __init__(self, cfg, env):
        if cfg.mode not in ('single', 'chunk') or cfg.required_clearance_m < .005:
            raise ValueError('RL single/chunk mode and >=5mm margin required')
        super().__init__(cfg, env)
        self._raw = torch.zeros((1, self.action_dim), device=env.device)
        self._processed = self._raw.clone()
        self._last_history = torch.zeros((1, 4, 9), device=env.device)
        self._planned = None
        self.geometry_tick_callback = None

    @property
    def action_dim(self):
        return 9 if self.cfg.mode == 'single' else 36

    @property
    def executed_history(self):
        # Only the completed, physically issued previous second is observable.
        return self._last_history.clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, '_last_history'):
            self._last_history.zero_()
        self._planned = None

    def process_actions(self, actions):
        if tuple(actions.shape) != (1, self.action_dim):
            raise ValueError('RL action tensor shape mismatch')
        started = time.monotonic()
        raw = array(actions)[0]
        raw = raw.reshape(4, 9) if self.cfg.mode == 'chunk' else raw
        self._raw[:] = actions
        q = self._rebase()
        ref = array(self._target[0])
        self._geometry()
        p, r = self.pose(self.ee)
        jac = array(self.robot.data.body_link_jacobian_w.torch[0, self.ee - int(self.robot.is_fixed_base)])
        jac = jac[:, np.asarray(self.ids[:6]) + int(self.robot.num_base_dofs)]

        def fk(angles):
            pp, rr = source_pose(p, r, jac, (angles-q[:6])[None])
            return pp[0], rr[0]

        self.fk_audit = fk
        limits = array(self.robot.data.soft_joint_pos_limits.torch[0, self.ids])
        error = q - ref

        def certified(path):
            return min(self.clearance(path), self.clearance(path + error))

        state = JointState(ref, fk, self.pose(self.base)[1], limits, certified)
        planned = build_schedule(self.cfg.mode, raw, state)
        self._planned = planned
        self.execution = planned.q.copy()
        self.velocities = planned.velocity.copy()
        self.goal_position, self.goal_rotation, self.goal_ball = planned.final_position, planned.final_rotation, planned.final_ball
        normalized = decode(raw if self.cfg.mode == 'chunk' else np.repeat(raw[None], 4, axis=0))[0]
        normalized = normalized if self.cfg.mode == 'chunk' else normalized[:1]
        self._processed[:] = self._processed.new_tensor(normalized.reshape(1, -1) * planned.projection_scale)
        self.tick = 0
        self.records = []
        self.physics_records = []
        self.telemetry = dict(mode=self.cfg.mode, raw_action=raw.tolist(), physical_scales=SCALES.tolist(),
            projection_scale=planned.projection_scale, reasons=list(planned.reasons),
            status='committed' if planned.projection_scale else 'rejected_hold',
            committed_seconds=1, preview_seconds=0, physics_steps=240,
            subsegment_ticks=list(planned.subsegment_ticks),
            predicted_min_clearance_m=planned.clearance_m,
            predicted_joint_knots_rad=planned.knots.tolist(),
            analytic_peak_speed_ratio=planned.analytic_peak_speed_ratio,
            analytic_peak_acceleration_ratio=planned.analytic_peak_acceleration_ratio,
            goal_position_m=self.goal_position.tolist(), goal_quat_xyzw=Rotation.from_matrix(self.goal_rotation).as_quat().tolist(),
            goal_ball_rad=self.goal_ball.tolist(), initial_servo_error_rad=error.tolist(),
            planner_wall_s=time.monotonic()-started, envelope=self.envelope)

    def apply_actions(self):
        if self.tick and self.tick % 24 == 0 and self.geometry_tick_callback is not None:
            self.geometry_tick_callback(self.tick)
        # Exactly the existing measured/reference stopping-path checks and
        # gravity-compensated joint targets; magnetic term stays 240Hz.
        super().apply_actions()
        if self.tick == 240 and self._planned is not None:
            self._last_history[0] = self._last_history.new_tensor(self._planned.subsegment_actions_si / SCALES)


@configclass
class NewStomachRLActionCfg(NewStomachActuatorVectorActionCfg):
    class_type: type = NewStomachRLAction
    mode: str = 'single'
