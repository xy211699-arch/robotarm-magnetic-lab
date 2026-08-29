"""Vectorized, non-farmable TASK-010 stagnation recovery state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch
from torch.nn import functional as F


class RecoveryPhase(IntEnum):
    NORMAL = 0
    ESCAPING = 1
    WAITING_COVERAGE = 2
    LOCKED = 3


@dataclass(frozen=True)
class RecoveryStep:
    phase_one_hot_4: torch.Tensor
    stagnation_progress: torch.Tensor
    max_escape_progress: torch.Tensor
    timer_fraction: torch.Tensor
    escape_progress_delta: torch.Tensor
    no_progress: torch.Tensor
    coverage_resumed: torch.Tensor
    delta_coverage: torch.Tensor
    coverage_reward: torch.Tensor
    escape_reward: torch.Tensor
    no_progress_reward: torch.Tensor
    coverage_resumed_reward: torch.Tensor
    total_reward: torch.Tensor


def _normalized_quaternion(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError("rotation must be a [N,4] xyzw quaternion")
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(1.0e-12)


def _angle_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    alignment = torch.abs((_normalized_quaternion(first) * _normalized_quaternion(second)).sum(dim=1))
    return torch.rad2deg(2.0 * torch.acos(alignment.clamp(0.0, 1.0)))


class Task010RecoveryTracker:
    def __init__(self, num_envs: int, device: str | torch.device = "cpu", dt_s: float = 0.1) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.default_dt_s = float(dt_s)
        self.phase = torch.full((self.num_envs,), int(RecoveryPhase.NORMAL), device=self.device, dtype=torch.int64)
        self.anchor_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.anchor_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.anchor_rotation[:, 3] = 1.0
        self.max_escape_progress = torch.zeros(self.num_envs, device=self.device)
        self.recovery_timer = torch.zeros(self.num_envs, device=self.device)
        self.wait_coverage = torch.zeros(self.num_envs, device=self.device)
        self.previous_coverage = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._position_history: list[torch.Tensor] = []
        self._rotation_history: list[torch.Tensor] = []
        self._coverage_history: list[torch.Tensor] = []

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        rows = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids.to(self.device, torch.int64)
        self.phase[rows] = int(RecoveryPhase.NORMAL)
        self.max_escape_progress[rows] = 0.0
        self.recovery_timer[rows] = 0.0
        self.wait_coverage[rows] = 0.0
        self.previous_coverage[rows] = 0.0
        self._initialized[rows] = False
        # Formal TASK-010 reset is synchronous; a partial test reset still
        # clears history conservatively so no row inherits stale evidence.
        self._position_history.clear()
        self._rotation_history.clear()
        self._coverage_history.clear()

    def update(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        coverage: torch.Tensor,
        dt_s: float | None = None,
    ) -> RecoveryStep:
        dt = self.default_dt_s if dt_s is None else float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        position = position.to(device=self.device, dtype=torch.float32)
        rotation = _normalized_quaternion(rotation.to(device=self.device, dtype=torch.float32))
        coverage = coverage.to(device=self.device, dtype=torch.float32).reshape(-1)
        if position.shape != (self.num_envs, 3) or rotation.shape != (self.num_envs, 4) or coverage.shape != (self.num_envs,):
            raise ValueError("TASK-010 recovery batch shape mismatch")
        if not (torch.isfinite(position).all() and torch.isfinite(rotation).all() and torch.isfinite(coverage).all()).item():
            raise RuntimeError("TASK-010 recovery input is non-finite")

        initialized_before = self._initialized.clone()
        delta_coverage = torch.where(initialized_before, coverage - self.previous_coverage, coverage).clamp_min(0.0)
        self._initialized.fill_(True)
        self._position_history.append(position.detach().clone())
        self._rotation_history.append(rotation.detach().clone())
        self._coverage_history.append(coverage.detach().clone())
        window_count = max(1, int(round(5.0 / dt)))
        for history in (self._position_history, self._rotation_history, self._coverage_history):
            del history[:-window_count]

        escape_delta = torch.zeros(self.num_envs, device=self.device)
        no_progress = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        coverage_resumed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if len(self._position_history) >= window_count:
            base_position = self._position_history[0]
            position_motion = torch.stack(
                [torch.linalg.vector_norm(item - base_position, dim=1) for item in self._position_history]
            ).amax(dim=0)
            base_rotation = self._rotation_history[0]
            rotation_motion = torch.stack([_angle_degrees(item, base_rotation) for item in self._rotation_history]).amax(dim=0)
            coverage_gain = coverage - self._coverage_history[0]
            stagnant = (
                (self.phase == int(RecoveryPhase.NORMAL))
                & (position_motion <= 0.001)
                & (rotation_motion <= 5.0)
                & (coverage_gain <= 0.001)
            )
            if stagnant.any().item():
                self.phase[stagnant] = int(RecoveryPhase.ESCAPING)
                self.anchor_position[stagnant] = position[stagnant]
                self.anchor_rotation[stagnant] = rotation[stagnant]
                self.max_escape_progress[stagnant] = 0.0
                self.recovery_timer[stagnant] = 0.0

        escaping = self.phase == int(RecoveryPhase.ESCAPING)
        if escaping.any().item():
            self.recovery_timer[escaping] += dt
            displacement = torch.linalg.vector_norm(position - self.anchor_position, dim=1) / 0.003
            angle = _angle_degrees(rotation, self.anchor_rotation) / 15.0
            progress = torch.maximum(displacement, angle).clamp(0.0, 1.0)
            escape_delta[escaping] = (progress[escaping] - self.max_escape_progress[escaping]).clamp_min(0.0)
            self.max_escape_progress[escaping] = torch.maximum(self.max_escape_progress[escaping], progress[escaping])
            no_progress[escaping] = escape_delta[escaping] <= 0.0
            reached = escaping & (self.max_escape_progress >= 1.0)
            if reached.any().item():
                self.phase[reached] = int(RecoveryPhase.WAITING_COVERAGE)
                self.wait_coverage[reached] = coverage[reached]
            expired = escaping & (self.recovery_timer >= 10.0)
            self.phase[expired] = int(RecoveryPhase.LOCKED)
            escape_delta[expired] = 0.0

        waiting = self.phase == int(RecoveryPhase.WAITING_COVERAGE)
        if waiting.any().item():
            self.recovery_timer[waiting] += dt
            gained = waiting & ((coverage - self.wait_coverage) >= 0.001)
            coverage_resumed[gained] = True
            self.phase[gained] = int(RecoveryPhase.NORMAL)
            self.max_escape_progress[gained] = 0.0
            self.recovery_timer[gained] = 0.0
            still_waiting = self.phase == int(RecoveryPhase.WAITING_COVERAGE)
            no_progress[still_waiting] = True
            wait_expired = still_waiting & (self.recovery_timer >= 2.0)
            self.phase[wait_expired] = int(RecoveryPhase.LOCKED)

        locked = self.phase == int(RecoveryPhase.LOCKED)
        unlocked = locked & (delta_coverage > 0.0)
        if unlocked.any().item():
            self.phase[unlocked] = int(RecoveryPhase.NORMAL)
            self.max_escape_progress[unlocked] = 0.0
            self.recovery_timer[unlocked] = 0.0
        no_progress[self.phase == int(RecoveryPhase.LOCKED)] = True

        stagnation_progress = torch.full(
            (self.num_envs,), min(1.0, len(self._position_history) * dt / 5.0), device=self.device
        )
        phase_one_hot = F.one_hot(self.phase, num_classes=4).to(dtype=torch.float32)
        coverage_reward = 100.0 * delta_coverage
        escape_reward = 0.1 * escape_delta
        no_progress_reward = -0.002 * no_progress.to(dtype=torch.float32)
        coverage_resumed_reward = 0.2 * coverage_resumed.to(dtype=torch.float32)
        total = coverage_reward + escape_reward + no_progress_reward + coverage_resumed_reward
        self.previous_coverage.copy_(coverage)
        return RecoveryStep(
            phase_one_hot, stagnation_progress, self.max_escape_progress.clone(),
            (self.recovery_timer / 10.0).clamp(0.0, 1.0), escape_delta, no_progress,
            coverage_resumed, delta_coverage, coverage_reward, escape_reward,
            no_progress_reward, coverage_resumed_reward, total,
        )
