from __future__ import annotations

import math

import torch

from robotarm_magnetic_lab.runtime.task010_recovery import RecoveryPhase, Task010RecoveryTracker


def _quat_z(degrees: float) -> torch.Tensor:
    half = math.radians(degrees) / 2.0
    return torch.tensor([[0.0, 0.0, math.sin(half), math.cos(half)]])


def _update(tracker, *, position=(0.0, 0.0, 0.0), rotation=None, coverage=0.0):
    return tracker.update(
        torch.tensor([position], dtype=torch.float32),
        _quat_z(0.0) if rotation is None else rotation,
        torch.tensor([coverage], dtype=torch.float32),
        dt_s=0.1,
    )


def test_stagnation_needs_all_three_conditions_for_five_seconds():
    tracker = Task010RecoveryTracker(num_envs=1)
    for _ in range(49):
        step = _update(tracker)
    assert tracker.phase.item() == RecoveryPhase.NORMAL
    step = _update(tracker)
    assert tracker.phase.item() == RecoveryPhase.ESCAPING
    assert step.phase_one_hot_4[0, RecoveryPhase.ESCAPING] == 1


def test_motion_or_rotation_or_coverage_prevents_trigger():
    for case in ("motion", "rotation", "coverage"):
        tracker = Task010RecoveryTracker(num_envs=1)
        for index in range(55):
            position = (0.002 * index / 54.0, 0.0, 0.0) if case == "motion" else (0.0, 0.0, 0.0)
            rotation = _quat_z(8.0 * index / 54.0) if case == "rotation" else _quat_z(0.0)
            coverage = 0.002 * index / 54.0 if case == "coverage" else 0.0
            _update(tracker, position=position, rotation=rotation, coverage=coverage)
        assert tracker.phase.item() == RecoveryPhase.NORMAL


def test_escape_progress_reward_is_paid_only_for_new_maximum():
    tracker = Task010RecoveryTracker(num_envs=1)
    for _ in range(50):
        _update(tracker)
    first = _update(tracker, position=(0.0015, 0.0, 0.0))
    repeated = _update(tracker, position=(0.0015, 0.0, 0.0))
    assert first.escape_progress_delta.item() == pytest.approx(0.5, abs=1e-5)
    assert repeated.escape_progress_delta.item() == 0.0
    assert repeated.no_progress.item()


def test_physical_escape_then_coverage_resume_returns_normal():
    tracker = Task010RecoveryTracker(num_envs=1)
    for _ in range(50):
        _update(tracker)
    reached = _update(tracker, position=(0.0031, 0.0, 0.0))
    assert tracker.phase.item() == RecoveryPhase.WAITING_COVERAGE
    resumed = _update(tracker, position=(0.0031, 0.0, 0.0), coverage=0.0011)
    assert resumed.coverage_resumed.item()
    assert tracker.phase.item() == RecoveryPhase.NORMAL


def test_escape_reward_cannot_be_farmed_after_ten_seconds():
    tracker = Task010RecoveryTracker(num_envs=1)
    for _ in range(50):
        _update(tracker)
    for index in range(101):
        step = _update(tracker, position=(0.0029 * ((index % 10) + 1) / 10.0, 0.0, 0.0))
    assert tracker.phase.item() == RecoveryPhase.LOCKED
    assert step.escape_progress_delta.item() == 0.0


def test_reward_components_match_contract():
    tracker = Task010RecoveryTracker(num_envs=1)
    first = _update(tracker, coverage=0.01)
    assert first.coverage_reward.item() == pytest.approx(1.0)


import pytest  # placed last to keep helpers uncluttered
