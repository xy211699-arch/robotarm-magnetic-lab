from __future__ import annotations

import torch
from torch import nn

from robotarm_magnetic_lab.runtime.task010_visual_encoder import (
    FrozenResNet18Encoder,
    center_crop_circular_rgb,
    preprocess_task010_rgb,
)


class TinyFrozenBackbone(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pooled = value.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return pooled.expand(-1, 512)


def test_center_crop_accepts_nhwc_and_nchw_without_stretching():
    rgb = torch.zeros((2, 720, 1280, 3), dtype=torch.uint8)
    rgb[:, 0, 280, :] = 255
    rgb[:, -1, 999, :] = 128
    cropped = center_crop_circular_rgb(rgb)
    assert cropped.shape == (2, 3, 720, 720)
    assert torch.equal(cropped[:, :, 0, 0], torch.full((2, 3), 255, dtype=torch.uint8))
    assert torch.equal(center_crop_circular_rgb(rgb.permute(0, 3, 1, 2)), cropped)


def test_preprocess_is_finite_float32_and_224_square():
    rgb = torch.randint(0, 256, (2, 720, 1280, 3), dtype=torch.uint8)
    output = preprocess_task010_rgb(rgb)
    assert output.shape == (2, 3, 224, 224)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_encoder_is_frozen_and_cached_per_environment_frame():
    encoder = FrozenResNet18Encoder(backbone=TinyFrozenBackbone(), weights_name="TEST")
    rgb = torch.randint(0, 256, (2, 720, 1280, 3), dtype=torch.uint8)
    first = encoder(rgb, torch.tensor([4, 4]))
    second = encoder(rgb, torch.tensor([4, 4]))
    assert first.shape == (2, 512)
    assert torch.equal(first, second)
    assert encoder.forward_image_count == 2
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert not first.requires_grad
    changed = encoder(rgb, torch.tensor([5, 4]))
    assert encoder.forward_image_count == 3
    assert torch.equal(changed[1], second[1])


def test_encoder_reset_invalidates_only_selected_rows():
    encoder = FrozenResNet18Encoder(backbone=TinyFrozenBackbone(), weights_name="TEST")
    rgb = torch.zeros((2, 720, 1280, 3), dtype=torch.uint8)
    encoder(rgb, torch.tensor([1, 1]))
    encoder.reset(torch.tensor([0]))
    encoder(rgb, torch.tensor([1, 1]))
    assert encoder.forward_image_count == 3
