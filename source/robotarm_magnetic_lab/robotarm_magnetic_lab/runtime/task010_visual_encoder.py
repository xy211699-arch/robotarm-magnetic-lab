"""Deterministic TASK-010 RGB preprocessing and frozen ResNet18 encoding."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET18_IMAGENET1K_V1_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def center_crop_circular_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Center-crop 1280x720 RGB to 720x720 without geometric stretching."""
    if rgb.ndim != 4:
        raise ValueError("TASK-010 RGB must be a rank-4 batch")
    if rgb.shape[-1] == 3:
        nchw = rgb.permute(0, 3, 1, 2)
    elif rgb.shape[1] == 3:
        nchw = rgb
    else:
        raise ValueError("TASK-010 RGB must be NHWC or NCHW with three channels")
    height, width = nchw.shape[-2:]
    if (height, width) != (720, 1280):
        raise ValueError(f"TASK-010 RGB expected 720x1280, got {height}x{width}")
    left = (width - height) // 2
    return nchw[:, :, :, left : left + height].contiguous()


def preprocess_task010_rgb(rgb: torch.Tensor) -> torch.Tensor:
    cropped = center_crop_circular_rgb(rgb)
    value = cropped.to(dtype=torch.float32)
    if not cropped.dtype.is_floating_point:
        value = value / 255.0
    elif value.numel() and (value.min().item() < 0.0 or value.max().item() > 1.0):
        raise ValueError("floating TASK-010 RGB must already be in [0,1]")
    value = F.interpolate(value, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, device=value.device, dtype=value.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=value.device, dtype=value.dtype).view(1, 3, 1, 1)
    value = (value - mean) / std
    if not torch.isfinite(value).all().item():
        raise RuntimeError("TASK-010 preprocessed RGB is non-finite")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenResNet18Encoder(nn.Module):
    """Frozen encoder with a per-environment, monotonic frame cache."""

    def __init__(self, *, backbone: nn.Module | None = None, weights_name: str = "IMAGENET1K_V1") -> None:
        super().__init__()
        self.weights_name = str(weights_name)
        self.weight_identity: dict[str, str | None]
        if backbone is None:
            from torchvision.models import ResNet18_Weights, resnet18
            import torch.hub

            if self.weights_name != "IMAGENET1K_V1":
                raise ValueError("TASK-010 requires ResNet18 IMAGENET1K_V1")
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            model.fc = nn.Identity()
            backbone = model
            filename = Path(urlparse(weights.url).path).name
            cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
            if not cache_path.is_file():
                raise RuntimeError("ResNet18 weight file was not retained in the torch hub cache")
            self.weight_identity = {
                "enum": self.weights_name,
                "url": weights.url,
                "path": str(cache_path),
                "sha256": _file_sha256(cache_path),
            }
            if self.weight_identity["sha256"] != RESNET18_IMAGENET1K_V1_SHA256:
                raise RuntimeError("TASK-010 ResNet18 IMAGENET1K_V1 weight hash mismatch")
        else:
            self.weight_identity = {"enum": self.weights_name, "url": None, "path": None, "sha256": None}
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self._cached_frame_ids: torch.Tensor | None = None
        self._cached_features: torch.Tensor | None = None
        self.forward_image_count = 0
        super().train(False)

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        self.backbone.eval()
        return self

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._cached_frame_ids is None:
            return
        if env_ids is None:
            self._cached_frame_ids.fill_(-1)
            if self._cached_features is not None:
                self._cached_features.zero_()
            return
        rows = env_ids.to(device=self._cached_frame_ids.device, dtype=torch.int64).reshape(-1)
        self._cached_frame_ids[rows] = -1
        if self._cached_features is not None:
            self._cached_features[rows] = 0.0

    def forward(self, rgb: torch.Tensor, frame_ids: torch.Tensor) -> torch.Tensor:
        frame_ids = frame_ids.to(device=rgb.device, dtype=torch.int64).reshape(-1)
        if rgb.shape[0] != frame_ids.shape[0]:
            raise ValueError("TASK-010 RGB and frame ID batch sizes differ")
        batch = frame_ids.shape[0]
        if self._cached_frame_ids is None or self._cached_frame_ids.shape[0] != batch:
            self._cached_frame_ids = torch.full((batch,), -1, device=rgb.device, dtype=torch.int64)
            self._cached_features = torch.zeros((batch, 512), device=rgb.device, dtype=torch.float32)
        assert self._cached_features is not None
        if torch.any((self._cached_frame_ids >= 0) & (frame_ids < self._cached_frame_ids)).item():
            raise RuntimeError("TASK-010 frame IDs decreased without cache reset")
        changed = frame_ids != self._cached_frame_ids
        if changed.any().item():
            rows = torch.nonzero(changed, as_tuple=False).reshape(-1)
            with torch.inference_mode():
                processed = preprocess_task010_rgb(rgb[rows])
                features = self.backbone(processed)
                if features.shape != (len(rows), 512):
                    raise RuntimeError(f"TASK-010 ResNet18 expected [N,512], got {tuple(features.shape)}")
                features = features.to(dtype=torch.float32)
                if not torch.isfinite(features).all().item():
                    raise RuntimeError("TASK-010 visual feature is non-finite")
            self._cached_features[rows] = features
            self._cached_frame_ids[rows] = frame_ids[rows]
            self.forward_image_count += int(len(rows))
        return self._cached_features.detach().clone()
