"""Capsule-endoscope optical post-processing for policy observations.

The RTX camera renders a wide pinhole image that contains every ray needed by
the requested circular field of view.  This module remaps those rays to an
equidistant wide-angle image, masks the inactive rectangular sensor area, and
adds a deterministic lens-shading profile.  Sensor noise and photometric
randomization belong in training events and are intentionally not baked into
this nominal camera model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera


_GRID_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _capsule_sampling_grid(
    height: int,
    width: int,
    intrinsics: torch.Tensor,
    field_of_view_deg: float,
    circle_fill: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an equidistant output-to-pinhole sampling grid.

    The requested FOV is measured across the diameter of the circular image,
    rather than across the full 16:9 renderer buffer.
    """
    # Intrinsics are constant in the nominal task. Reusing the GPU grid avoids
    # evaluating almost one million square roots and trigonometric operations
    # independently for both RGB and depth on every policy step.
    cache_key = (
        intrinsics.device.type,
        intrinsics.device.index,
        intrinsics.data_ptr(),
        intrinsics.shape[0],
        height,
        width,
        float(field_of_view_deg),
        float(circle_fill),
    )
    cached = _GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    device = intrinsics.device
    dtype = torch.float32
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    center_x = 0.5 * (width - 1)
    center_y = 0.5 * (height - 1)
    radius = 0.5 * min(height, width) * circle_fill
    circle_x = (x - center_x) / radius
    circle_y = (y - center_y) / radius
    radial = torch.sqrt(circle_x.square() + circle_y.square())

    # Equidistant lens: output radius is proportional to ray angle.
    theta_max = math.radians(field_of_view_deg * 0.5)
    theta = radial.clamp(max=1.0) * theta_max
    scale = torch.where(radial > 1.0e-6, torch.tan(theta) / radial, torch.ones_like(radial) * theta_max)
    ray_x = circle_x * scale
    ray_y = circle_y * scale

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)
    source_x = fx * ray_x.unsqueeze(0) + cx
    source_y = fy * ray_y.unsqueeze(0) + cy
    grid_x = 2.0 * source_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * source_y / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)

    # A short smooth border prevents a jagged one-pixel circular edge.
    feather_width = 0.025
    alpha = ((1.0 - radial) / feather_width).clamp(0.0, 1.0)
    alpha = alpha.unsqueeze(0).unsqueeze(1)
    # Published capsule optical designs report roughly >=60% relative edge
    # illumination.  Use a conservative 65% nominal lens-shading profile.
    vignetting = (1.0 - 0.35 * radial.clamp(max=1.0).square()).unsqueeze(0).unsqueeze(1)
    result = (grid, alpha, vignetting)
    if len(_GRID_CACHE) >= 8:
        _GRID_CACHE.clear()
    _GRID_CACHE[cache_key] = result
    return result


def _sample_capsule_image(
    image: torch.Tensor,
    intrinsics: torch.Tensor,
    field_of_view_deg: float,
    circle_fill: float,
    *,
    is_depth: bool,
) -> torch.Tensor:
    """Apply the capsule lens model while retaining NHWC tensor layout."""
    batch, height, width, _ = image.shape
    grid, alpha, vignetting = _capsule_sampling_grid(
        height,
        width,
        intrinsics[:batch].float(),
        field_of_view_deg,
        circle_fill,
    )
    source = image.permute(0, 3, 1, 2).float()
    sampled = functional.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    if is_depth:
        # Zero is the explicit invalid-depth value outside the optical circle.
        sampled = torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0) * alpha
    else:
        # Policy RGB is float32 [0, 1], independent of renderer byte format.
        if image.dtype == torch.uint8:
            sampled = sampled / 255.0
        sampled = sampled.clamp(0.0, 1.0) * alpha * vignetting
    return sampled.permute(0, 2, 3, 1).contiguous()


def capsule_rgb(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("capsule_camera"),
    field_of_view_deg: float = 120.0,
    circle_fill: float = 0.98,
    require_new_control_boundary_frame: bool = False,
) -> torch.Tensor:
    """Return circular, wide-angle capsule RGB observations in NHWC layout."""
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    if require_new_control_boundary_frame:
        _synchronize_policy_rgb_at_control_boundary(env, camera)
    return _sample_capsule_image(
        # RTX commonly exposes an alpha channel; the policy interface is true
        # RGB and must match the physical three-channel camera stream.
        camera.data.output["rgb"].torch[..., :3],
        camera.data.intrinsic_matrices.torch,
        field_of_view_deg,
        circle_fill,
        is_depth=False,
    )


def _synchronize_policy_rgb_at_control_boundary(env: ManagerBasedEnv, camera: Camera) -> None:
    """Guarantee one and only one fresh camera frame per formal control boundary.

    Isaac Lab schedules RTX sensors from floating-point simulation timestamps.
    On an exact 0.1 s boundary that comparison can occasionally retain the
    previous frame.  The formal TASK-009B observation path therefore performs
    one internal buffer update only when the renderer has not advanced.  The
    state is attached to the environment so repeated observation reads at the
    same boundary never trigger an additional capture.

    ``Camera._update_buffers_impl`` is an Isaac Lab private API.  Keeping the
    compatibility-sensitive call isolated here makes version audits explicit.
    """
    _ = camera.data.output["rgb"]
    boundary = int(env.common_step_counter)
    frame = int(camera.frame.torch[0].item())
    state = getattr(env, "_task009b_policy_rgb_sync", None)
    if state is None:
        state = {"boundary": boundary, "frame": frame, "forced_capture": False}
    elif boundary == int(state["boundary"]):
        # Observation managers and UI helpers may read the same policy term
        # more than once.  The already-associated frame must be reused.
        frame = int(state["frame"])
    else:
        previous_frame = int(state["frame"])
        forced = False
        if frame <= previous_frame:
            camera._update_buffers_impl(camera._ALL_ENV_MASK)
            _ = camera.data.output["rgb"]
            frame = int(camera.frame.torch[0].item())
            forced = True
        if frame != previous_frame + 1:
            raise RuntimeError(
                "formal policy RGB must advance by exactly one frame per control boundary: "
                f"previous={previous_frame}, current={frame}, boundary={boundary}"
            )
        state = {"boundary": boundary, "frame": frame, "forced_capture": forced}
    env._task009b_policy_rgb_sync = state
    env._task009b_policy_rgb_sync_latest = dict(state)


def capsule_depth(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("capsule_camera"),
    field_of_view_deg: float = 120.0,
    circle_fill: float = 0.98,
) -> torch.Tensor:
    """Return depth aligned pixel-for-pixel with :func:`capsule_rgb`."""
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    return _sample_capsule_image(
        # Euclidean ray distance remains physically meaningful after the
        # wide-angle remap; optical-axis Z depth would not.
        camera.data.output["distance_to_camera"].torch,
        camera.data.intrinsic_matrices.torch,
        field_of_view_deg,
        circle_fill,
        is_depth=True,
    )
