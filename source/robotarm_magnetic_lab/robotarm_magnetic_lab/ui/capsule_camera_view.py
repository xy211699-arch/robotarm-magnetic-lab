"""Optional 30 Hz Kit preview for the capsule camera pose and optics.

The regular viewport is kept interactive. When enabled, Isaac Lab's Kit
visualizer creates a second image panel backed by a dedicated preview sensor.
The preview copies the recorded camera's pose and optical model but runs at
30 Hz. The policy/data camera remains at its task-defined acquisition rate
(1 Hz in the stomach task), so preview frames never enter observations or
training datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any


# The scene config is authored with ``{ENV_REGEX_NS}``, but CameraCfg resolves
# that token before the Kit visualizer searches the runtime sensor registry.
# Therefore this must match Camera.cfg.prim_path after scene construction.
CAMERA_PREVIEW_CONFIG_PATH = (
    "{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet/capsule_camera_preview"
)
CAMERA_PREVIEW_SENSOR_PATH = (
    "/World/envs/env_.*/Scene/MagneticDemo/target_magnet/capsule_camera_preview"
)
CAPSULE_POSE_CAMERA_CONFIG_PATH = "{ENV_REGEX_NS}/capsule_pose_observer"
CAPSULE_POSE_CAMERA_SENSOR_PATH = "/World/envs/env_.*/capsule_pose_observer"
PREVIEW_REFRESH_HZ = 30.0
WINDOW_TITLE_PREFIX = "Capsule Camera | Debug Preview"
POSE_VIEW_TITLE = "Capsule Pose | External Follow | 30 Hz"


def _append_visualizer_cfg(env_cfg: Any, cfg: Any) -> None:
    """Append one visualizer without removing panels configured earlier."""
    existing = env_cfg.sim.visualizer_cfgs
    if existing is None:
        env_cfg.sim.visualizer_cfgs = [cfg]
    elif isinstance(existing, list):
        existing.append(cfg)
    else:
        env_cfg.sim.visualizer_cfgs = [existing, cfg]


def configure_capsule_camera_view(env_cfg: Any) -> None:
    """Add a 30 Hz debug camera and its Kit panel before scene construction."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    recorded_camera = env_cfg.scene.capsule_camera
    recorded_period = float(recorded_camera.update_period)
    recorded_hz = 0.0 if recorded_period <= 0.0 else 1.0 / recorded_period
    env_cfg.scene.capsule_camera_preview = recorded_camera.replace(
        prim_path=CAMERA_PREVIEW_CONFIG_PATH,
        update_period=1.0 / PREVIEW_REFRESH_HZ,
    )
    window_title = (
        f"{WINDOW_TITLE_PREFIX} | {PREVIEW_REFRESH_HZ:g} Hz "
        f"(recorded {recorded_hz:g} Hz)"
    )
    _append_visualizer_cfg(
        env_cfg,
        KitVisualizerCfg(
            eye=tuple(env_cfg.viewer.eye),
            lookat=tuple(env_cfg.viewer.lookat),
            tiled_cam_view=True,
            tiled_cam_num=1,
            tiled_cam_env_indices=[0],
            tiled_cam_prim_path=CAMERA_PREVIEW_SENSOR_PATH,
            viewport_name=window_title,
            create_viewport=False,
            dock_position="RIGHT",
            window_width=960,
            window_height=540,
        ),
    )


def configure_capsule_pose_view(env_cfg: Any) -> None:
    """Add a world-up external camera that follows the passive capsule."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    env_cfg.scene.capsule_pose_camera = CameraCfg(
        prim_path=CAPSULE_POSE_CAMERA_CONFIG_PATH,
        update_period=1.0 / PREVIEW_REFRESH_HZ,
        width=960,
        height=540,
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=36.0,
            focus_distance=0.08,
            clipping_range=(0.005, 2.0),
        ),
    )
    _append_visualizer_cfg(
        env_cfg,
        KitVisualizerCfg(
            eye=tuple(env_cfg.viewer.eye),
            lookat=tuple(env_cfg.viewer.lookat),
            tiled_cam_view=True,
            tiled_cam_num=1,
            tiled_cam_env_indices=[0],
            tiled_cam_prim_path=CAPSULE_POSE_CAMERA_SENSOR_PATH,
            viewport_name=POSE_VIEW_TITLE,
            create_viewport=False,
            dock_position="LEFT",
            window_width=960,
            window_height=540,
        ),
    )


@dataclass
class CapsuleCameraViewHandle:
    """Own the runtime post-processing hook so it can be restored cleanly."""

    visualizer: Any
    original_update: Any

    def close(self) -> None:
        """Restore the stock raw-camera update method."""
        if self.visualizer is not None and self.original_update is not None:
            self.visualizer._update_camera_image_panel = self.original_update
        self.visualizer = None
        self.original_update = None


@dataclass
class CapsulePoseViewHandle:
    """Own the external follow-camera update hook."""

    visualizer: Any
    original_update: Any

    def close(self) -> None:
        """Restore the stock camera panel update method."""
        if self.visualizer is not None and self.original_update is not None:
            self.visualizer._update_camera_image_panel = self.original_update
        self.visualizer = None
        self.original_update = None


def attach_capsule_camera_policy_view(env: Any) -> CapsuleCameraViewHandle:
    """Show circular preview RGB at 30 Hz without changing recorded observations."""
    import torch

    from isaaclab.managers import SceneEntityCfg

    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.vision import capsule_rgb

    base_env = env.unwrapped
    recorded_camera = base_env.scene["capsule_camera"]
    preview_camera = base_env.scene["capsule_camera_preview"]
    visualizer = next(
        (
            item
            for item in base_env.sim.visualizers
            if getattr(getattr(item, "cfg", None), "tiled_cam_prim_path", None)
            == CAMERA_PREVIEW_SENSOR_PATH
            and getattr(item, "_camera_image_provider", None) is not None
        ),
        None,
    )
    if visualizer is None:
        raise RuntimeError(
            "Capsule camera view requested, but its Kit image panel was not created. "
            "Do not combine --capsule_camera_view with a headless visualizer."
        )

    original_update = visualizer._update_camera_image_panel
    last_camera_frame = -1

    def _update_policy_image(self, dt: float) -> None:
        nonlocal last_camera_frame
        if self._camera_image_provider is None:
            return

        # The preview sensor runs at the render cadence. The separately
        # registered capsule_camera remains untouched at the acquisition rate
        # used by observations and datasets.
        _ = preview_camera.data.output["rgb"]
        camera_frame = int(preview_camera.frame.torch[0].item())
        if camera_frame == last_camera_frame:
            return

        with torch.inference_mode():
            policy_rgb = capsule_rgb(
                base_env,
                sensor_cfg=SceneEntityCfg("capsule_camera_preview"),
            )[0]
            display_rgb = (
                policy_rgb.mul(255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8).contiguous()
            )
        self._upload_camera_image_to_panel(display_rgb)
        last_camera_frame = camera_frame

    visualizer._update_camera_image_panel = MethodType(_update_policy_image, visualizer)

    recorded_sensor_hz = 1.0 / float(recorded_camera.cfg.update_period)
    preview_hz = 1.0 / float(preview_camera.cfg.update_period)
    render_hz = 1.0 / float(base_env.sim.get_rendering_dt())
    window_title = visualizer.cfg.viewport_name
    print(
        f"[CAPSULE_CAMERA_VIEW] enabled title={window_title!r} "
        f"source=debug_preview.circular_rgb "
        f"resolution={preview_camera.cfg.width}x{preview_camera.cfg.height} "
        f"preview_hz={preview_hz:.1f} recorded_sensor_hz={recorded_sensor_hz:.1f} "
        f"render_hz={render_hz:.1f} enters_observations=false",
        flush=True,
    )
    return CapsuleCameraViewHandle(visualizer=visualizer, original_update=original_update)


def attach_capsule_pose_view(env: Any) -> CapsulePoseViewHandle:
    """Track the capsule from a fixed world-up diagonal offset at 30 Hz."""
    import torch

    base_env = env.unwrapped
    capsule = base_env.scene["capsule"]
    pose_camera = base_env.scene["capsule_pose_camera"]
    visualizer = next(
        (
            item
            for item in base_env.sim.visualizers
            if getattr(getattr(item, "cfg", None), "tiled_cam_prim_path", None)
            == CAPSULE_POSE_CAMERA_SENSOR_PATH
            and getattr(item, "_camera_image_provider", None) is not None
        ),
        None,
    )
    if visualizer is None:
        raise RuntimeError(
            "Capsule pose view requested, but its Kit image panel was not created. "
            "Use the Kit visualizer and do not run headless."
        )

    # A fixed WORLD-frame offset makes capsule roll, pitch and yaw visible
    # instead of rotating the observer together with the body. The camera is
    # close enough to remain inside the stomach lumen around the lower start.
    # A mostly +Z offset looks down at the capsule resting on the lower wall;
    # the small +X component avoids a singular straight-down look-at frame.
    follow_offset = torch.tensor(
        (0.010, 0.0, 0.040),
        dtype=torch.float32,
        device=base_env.device,
    ).reshape(1, 3)

    def _update_pose() -> tuple[torch.Tensor, torch.Tensor]:
        target = capsule.data.root_pos_w.torch[:, :3]
        eye = target + follow_offset
        pose_camera.set_world_poses_from_view(eye, target)
        return eye, target

    eye, target = _update_pose()
    original_update = visualizer._update_camera_image_panel

    def _update_follow_image(self, dt: float) -> None:
        _update_pose()
        original_update(dt)

    visualizer._update_camera_image_panel = MethodType(
        _update_follow_image, visualizer
    )
    print(
        f"[CAPSULE_POSE_VIEW] enabled title={POSE_VIEW_TITLE!r} "
        f"resolution={pose_camera.cfg.width}x{pose_camera.cfg.height} "
        f"refresh_hz={PREVIEW_REFRESH_HZ:.1f} "
        f"world_offset={follow_offset[0].detach().cpu().tolist()} "
        f"initial_eye={eye[0].detach().cpu().tolist()} "
        f"initial_target={target[0].detach().cpu().tolist()}",
        flush=True,
    )
    return CapsulePoseViewHandle(
        visualizer=visualizer,
        original_update=original_update,
    )
