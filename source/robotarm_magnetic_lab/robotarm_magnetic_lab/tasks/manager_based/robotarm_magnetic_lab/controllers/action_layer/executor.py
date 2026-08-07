"""Non-blocking, non-preemptive atomic action executor."""

from __future__ import annotations

import numpy as np

from .action_mask import ActionMask, compute_action_mask
from .config import ActionLayerConfig
from .hard_safety import HardSafetyMonitor, SafetyCheck
from .planner import AtomicCommandPlanner, PlannerError
from .types import (
    ActionRequest,
    ActionResult,
    ActionStatus,
    AtomicAction,
    DeviceSnapshot,
    ExecutionState,
    ExecutorStep,
    HardFailureCode,
    MagnetCommandState,
)


class AtomicActionExecutor:
    """Execute one accepted request to a terminal state before accepting another."""

    def __init__(
        self,
        cfg: ActionLayerConfig,
        planner: AtomicCommandPlanner,
        safety: HardSafetyMonitor,
        command_state: MagnetCommandState,
    ) -> None:
        self.cfg = cfg
        self.planner = planner
        self.safety = safety
        self.command_state = command_state.copy()
        self.state = ExecutionState.IDLE
        self.request: ActionRequest | None = None
        self.plan = None
        self.waypoint_index = 0
        self.settle_steps = 0
        self.control_steps = 0
        self.last_result: ActionResult | None = None
        self.last_rejected_code: HardFailureCode | None = None
        self.state_timestamps_s: dict[str, float] = {}
        self.minimum_asm_clearance_m = np.inf
        self.last_safe_target = np.concatenate(
            (self.command_state.arm_joint_target_rad, self.command_state.ball_joint_target_rad)
        )

    @property
    def busy(self) -> bool:
        return self.state in {
            ExecutionState.PRECHECK,
            ExecutionState.PLAN,
            ExecutionState.EXECUTE,
            ExecutionState.HARD_FAILURE,
            ExecutionState.SAFE_RECOVER,
        }

    def action_mask(self, snapshot: DeviceSnapshot) -> ActionMask:
        return compute_action_mask(self.command_state, snapshot, self.cfg, busy=self.busy)

    def submit(self, action_id: int, snapshot: DeviceSnapshot, request_id: int) -> bool:
        self.last_rejected_code = None
        if self.busy:
            self.last_rejected_code = HardFailureCode.BUSY
            return False
        try:
            action = AtomicAction(int(action_id))
        except (TypeError, ValueError):
            action = AtomicAction.HOLD
            self.request = ActionRequest(request_id, action, snapshot.sim_time_s)
            self._fail(snapshot, HardFailureCode.INVALID_ACTION, f"invalid action ID {action_id!r}")
            return False
        mask = compute_action_mask(self.command_state, snapshot, self.cfg)
        self.request = ActionRequest(request_id, action, snapshot.sim_time_s)
        self.last_result = None
        self.state_timestamps_s = {}
        self._transition(ExecutionState.PRECHECK, snapshot.sim_time_s)
        if not mask.allows(action):
            self._fail(
                snapshot,
                HardFailureCode.ACTION_MASKED,
                mask.reasons[int(action)],
            )
            return False
        return True

    def step(self, snapshot: DeviceSnapshot) -> ExecutorStep:
        accepted = False
        if self.state is ExecutionState.PRECHECK:
            check = self.safety.check_snapshot(snapshot)
            if not check.ok:
                self._fail_check(snapshot, check)
            else:
                self._transition(ExecutionState.PLAN, snapshot.sim_time_s)
        if self.state is ExecutionState.PLAN:
            try:
                assert self.request is not None
                self.plan = self.planner.plan(self.request, snapshot, self.command_state)
                check = self.safety.check_plan(self.plan, snapshot)
                if not check.ok:
                    self._fail_check(snapshot, check)
                else:
                    self.minimum_asm_clearance_m = min(
                        snapshot.asm_clearance_m, check.minimum_asm_clearance_m
                    )
                    self.waypoint_index = 0
                    self.settle_steps = 0
                    self.control_steps = 0
                    self._transition(ExecutionState.EXECUTE, snapshot.sim_time_s)
            except PlannerError as error:
                self._fail(snapshot, error.code, str(error))
            except Exception as error:
                self._fail(snapshot, HardFailureCode.INTERNAL_ERROR, repr(error))
        if self.state is ExecutionState.EXECUTE:
            check = self.safety.check_snapshot(snapshot)
            if not check.ok:
                self._fail_check(snapshot, check)
            else:
                assert self.plan is not None
                self.minimum_asm_clearance_m = min(
                    self.minimum_asm_clearance_m, snapshot.asm_clearance_m
                )
                last_index = self.plan.joint_targets_rad.shape[0] - 1
                self.waypoint_index = min(self.waypoint_index, last_index)
                target = self.plan.joint_targets_rad[self.waypoint_index].copy()
                self.last_safe_target = target
                self.control_steps += 1
                if self.waypoint_index < last_index:
                    self.waypoint_index += 1
                else:
                    error = float(np.max(np.abs(snapshot.joint_position_rad - target)))
                    speed = float(np.max(np.abs(snapshot.joint_velocity_rad_s)))
                    if error <= self.cfg.settle_position_tolerance_rad and speed <= self.cfg.settle_velocity_tolerance_rad_s:
                        self.settle_steps += 1
                    else:
                        self.settle_steps = 0
                    if self.settle_steps >= self.cfg.settle_required_steps:
                        self.command_state = self.plan.final_command_state.copy()
                        self._done(snapshot)
                    elif snapshot.sim_time_s - self.request.requested_at_s > self.cfg.maximum_duration_s + 0.5:
                        self._fail(snapshot, HardFailureCode.EXECUTION_TIMEOUT, "target did not settle")
        if self.state is ExecutionState.HARD_FAILURE:
            # Initial containment policy: hold the last validated target and
            # terminate the episode. No unvalidated recovery motion is claimed.
            self._transition(ExecutionState.SAFE_RECOVER, snapshot.sim_time_s)
        return ExecutorStep(
            state=self.state,
            joint_target_rad=self.last_safe_target.copy(),
            accepted_request=accepted,
            result=self.last_result,
        )

    def reset(self, snapshot: DeviceSnapshot, command_state: MagnetCommandState) -> None:
        self.command_state = command_state.copy()
        self.last_safe_target = snapshot.joint_position_rad.copy()
        self.state = ExecutionState.IDLE
        self.request = None
        self.plan = None
        self.last_result = None
        self.last_rejected_code = None
        self.state_timestamps_s = {}

    def _transition(self, state: ExecutionState, time_s: float) -> None:
        self.state = state
        self.state_timestamps_s[state.value] = float(time_s)

    def _done(self, snapshot: DeviceSnapshot) -> None:
        self._transition(ExecutionState.DONE, snapshot.sim_time_s)
        self.last_result = self._result(snapshot, ActionStatus.DONE)

    def _fail_check(self, snapshot: DeviceSnapshot, check: SafetyCheck) -> None:
        self._fail(snapshot, check.code or HardFailureCode.INTERNAL_ERROR, check.detail)

    def _fail(self, snapshot: DeviceSnapshot, code: HardFailureCode, detail: str) -> None:
        self._transition(ExecutionState.HARD_FAILURE, snapshot.sim_time_s)
        self.last_result = self._result(snapshot, ActionStatus.HARD_FAILURE, code, detail)

    def _result(
        self,
        snapshot: DeviceSnapshot,
        status: ActionStatus,
        code: HardFailureCode | None = None,
        detail: str | None = None,
    ) -> ActionResult:
        request = self.request or ActionRequest(-1, AtomicAction.HOLD, snapshot.sim_time_s)
        started = self.state_timestamps_s.get(ExecutionState.EXECUTE.value, request.requested_at_s)
        return ActionResult(
            request_id=request.request_id,
            action=request.action,
            status=status,
            requested_at_s=request.requested_at_s,
            started_at_s=started,
            ended_at_s=snapshot.sim_time_s,
            control_steps=self.control_steps,
            final_joint_position_rad=snapshot.joint_position_rad.copy(),
            final_ball_position_rad=snapshot.joint_position_rad[-self.cfg.ball_joint_count :].copy(),
            final_magnet_position_world_m=snapshot.magnet_position_world_m.copy(),
            final_magnet_rotation_world=snapshot.magnet_rotation_world.copy(),
            minimum_asm_clearance_m=self.minimum_asm_clearance_m,
            state_timestamps_s=dict(self.state_timestamps_s),
            hard_failure_code=code,
            hard_failure_detail=detail,
        )
