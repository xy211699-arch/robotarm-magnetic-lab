from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
from robotarm_magnetic_lab.runtime.task010_visual_intervention import (
    Task010VisualIntervention,
    replace_actor_visual_features,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp import (
    task010_terms as terms,
)


class FakeEncoder:
    def __init__(self, features):
        self.features = features
        self.forward_image_count = 0

    def __call__(self, rgb, frame_ids):
        self.forward_image_count += int(frame_ids.shape[0])
        return self.features


def _fake_env(condition, features, previous):
    encoder = FakeEncoder(features)
    env = SimpleNamespace(
        cfg=SimpleNamespace(task010_visual_condition=condition),
        num_envs=features.shape[0],
        device="cpu",
        _task010_visual_encoder=encoder,
        action_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(
                previous_action_features=previous
            )
        ),
    )
    return env, encoder


def _patch_runtime(monkeypatch, *, num_envs: int = 2):
    rgb = torch.zeros((num_envs, 720, 1280, 3), dtype=torch.uint8)
    monkeypatch.setattr(terms, "task009d0_rgb", lambda env: rgb)
    monkeypatch.setattr(
        terms,
        "task009d0_runtime",
        lambda env: SimpleNamespace(
            rgb_sync=SimpleNamespace(latest=torch.arange(num_envs, dtype=torch.int64) + 1)
        ),
    )


def test_blind_zeroes_only_visual_slice_after_encoder_forward(monkeypatch):
    features = torch.randn((2, 512))
    previous = torch.randn((2, 7))
    env, encoder = _fake_env("blind", features, previous)
    monkeypatch.setattr(terms, "task010_visual_encoder", lambda env: encoder)
    _patch_runtime(monkeypatch)

    observation = terms.task010_actor_observation(env)

    assert encoder.forward_image_count == 2
    assert torch.equal(observation[:, :512], torch.zeros_like(observation[:, :512]))
    assert torch.equal(observation[:, 512:], previous)


def test_replace_actor_visual_features_preserves_target_previous_action():
    target = torch.randn(3, 519)
    donor = torch.randn(3, 512)
    changed = replace_actor_visual_features(target, donor)
    assert torch.equal(changed[:, :512], donor)
    assert torch.equal(changed[:, 512:], target[:, 512:])


def test_first_frame_is_per_environment_and_resettable():
    state = Task010VisualIntervention("first_frame", num_envs=2, feature_dim=512)
    first = torch.stack((torch.ones(512), torch.full((512,), 2.0)))
    assert torch.equal(state.apply(first), first)
    assert torch.equal(state.apply(torch.full_like(first, 9.0)), first)
    state.reset(torch.tensor([1]))
    next_features = torch.stack((torch.full((512,), 8.0), torch.full((512,), 3.0)))
    output = state.apply(next_features)
    assert torch.equal(output[0], first[0])
    assert torch.equal(output[1], next_features[1])


def _trainable_manifest(model):
    return tuple(
        (name, tuple(parameter.shape), parameter.numel(), parameter.requires_grad)
        for name, parameter in model.named_parameters()
    )


def test_normal_and_blind_have_identical_trainable_parameter_manifest():
    assert _trainable_manifest(Task010Actor()) == _trainable_manifest(Task010Actor())


def test_normal_and_blind_both_execute_resnet(monkeypatch):
    counts = []
    for condition in ("normal", "blind"):
        features = torch.randn((12, 512))
        previous = torch.zeros((12, 7))
        env, encoder = _fake_env(condition, features, previous)
        monkeypatch.setattr(terms, "task010_visual_encoder", lambda env, encoder=encoder: encoder)
        _patch_runtime(monkeypatch, num_envs=12)
        terms.task010_actor_observation(env)
        counts.append(encoder.forward_image_count)
    assert counts == [12, 12]


def test_donor_condition_is_rejected_in_training_observation(monkeypatch):
    features = torch.randn((2, 512))
    previous = torch.zeros((2, 7))
    env, encoder = _fake_env("donor", features, previous)
    monkeypatch.setattr(terms, "task010_visual_encoder", lambda env: encoder)
    _patch_runtime(monkeypatch)
    with pytest.raises(ValueError, match="validation-only"):
        terms.task010_actor_observation(env)
