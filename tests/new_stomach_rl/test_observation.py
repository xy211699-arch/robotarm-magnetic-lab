from types import SimpleNamespace
import torch
from robotarm_magnetic_lab.runtime.new_stomach_rl_runtime import NewStomachRLRuntime
from robotarm_magnetic_lab.runtime.task010_visual_encoder import FrozenResNet18Encoder
from robotarm_magnetic_lab.runtime.new_stomach_rl_library import FrozenPoseLibrary
from pathlib import Path
import pytest


class Forbidden:
    def __getattr__(self,name):
        raise AssertionError('Actor read privileged truth: '+name)


def test_actor_only_reads_visual_and_actual_issued_history():
    runtime = object.__new__(NewStomachRLRuntime)
    runtime.visual_features = torch.zeros((1,512))
    actual = torch.arange(36.).reshape(1,4,9)
    runtime.term = SimpleNamespace(executed_history=actual, raw_actions=Forbidden(), telemetry=Forbidden())
    runtime.capsule = runtime.coverage = runtime.reward = runtime.env = Forbidden()
    observation = runtime.actor_observation()
    assert observation.shape == (1,548)
    assert observation[:,:512].count_nonzero() == 0
    assert torch.equal(observation[:,512:],actual.reshape(1,36))


def test_blind_visual_path_never_reads_rgb_or_encoder():
    runtime = object.__new__(NewStomachRLRuntime)
    runtime.env = SimpleNamespace(cfg=SimpleNamespace(group='A'))
    runtime.rgb = runtime.encoder = Forbidden()
    runtime.visual_features = torch.ones((1,512))
    runtime._encode_visual()
    assert runtime.visual_features.count_nonzero() == 0


def test_frozen_visual_encoder_one_forward_per_new_frame():
    class Backbone(torch.nn.Module):
        def forward(self,image):
            return image.mean((1,2,3))[:,None].expand(-1,512)
    encoder = FrozenResNet18Encoder(backbone=Backbone())
    rgb = torch.zeros((1,720,1280,3),dtype=torch.uint8)
    first = encoder(rgb,torch.tensor([1]))
    repeated = encoder(torch.ones_like(rgb),torch.tensor([1]))
    assert torch.equal(first,repeated)
    assert encoder.forward_image_count == 1
    changed = encoder(torch.full_like(rgb,255),torch.tensor([2]))
    assert not torch.equal(first,changed)
    assert encoder.forward_image_count == 2
    assert not changed.requires_grad
    encoder.reset()
    encoder(rgb,torch.tensor([0]))
    assert encoder.forward_image_count == 3


def test_private_sampler_seeds_independent_and_no_test_split():
    manifest = Path(__file__).resolve().parents[2].parent/'new-stomach-coverage-regenerate/configs/new_stomach_v1/entry/pose_library_manifest_v1.json'
    left = FrozenPoseLibrary(manifest,'train',8)
    right = FrozenPoseLibrary(manifest,'train',8)
    assert [left.sample()[0] for _ in range(5)] == [right.sample()[0] for _ in range(5)]
    left.sample()
    reference = FrozenPoseLibrary(manifest,'train',8)
    expected = [reference.sample()[0] for _ in range(6)][-1]
    assert right.sample()[0] == expected
    with pytest.raises(ValueError,match='test split'):
        FrozenPoseLibrary(manifest,'test',8)
    with pytest.raises(ValueError,match='requested split'):
        FrozenPoseLibrary(manifest,'validation',8,'train-0003')
