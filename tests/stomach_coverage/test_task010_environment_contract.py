from __future__ import annotations

import torch
from types import SimpleNamespace

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.task009d0_vector_env import (
    Task009D0VectorEnv,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.task010_vector_env import (
    Task010VectorEnv,
)


def test_d0_keeps_positive_reachable_c0_requirement():
    env = object.__new__(Task009D0VectorEnv)
    env._is_closed = True
    valid = env._initial_reachable_coverage_is_valid(
        torch.tensor([0.0, 0.1]), torch.tensor([0.1, 0.1])
    )
    assert valid.tolist() == [False, True]


def test_task010_accepts_zero_reachable_when_raw_is_positive_and_finite():
    env = object.__new__(Task010VectorEnv)
    env._is_closed = True
    valid = env._initial_reachable_coverage_is_valid(
        torch.tensor([0.0, 0.1, 0.0]),
        torch.tensor([0.1, 0.1, 0.0]),
    )
    assert valid.tolist() == [True, True, False]


def test_task010_horizon_is_terminal_not_timeout():
    env = object.__new__(Task010VectorEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=3)
    env.sim = SimpleNamespace(device="cpu")
    terminated, truncated, time_outs = env._horizon_termination_flags()
    assert terminated.tolist() == [True, True, True]
    assert truncated.tolist() == [False, False, False]
    assert time_outs.tolist() == [False, False, False]


def test_task010_registered_with_independent_task_id():
    import gymnasium as gym
    import robotarm_magnetic_lab.tasks  # noqa: F401

    spec = gym.spec("Template-Robotarm-Magnetic-Task010-CNN-GRU-Coverage-Lab-v0")
    assert spec.entry_point.endswith("task010_vector_env:Task010VectorEnv")
