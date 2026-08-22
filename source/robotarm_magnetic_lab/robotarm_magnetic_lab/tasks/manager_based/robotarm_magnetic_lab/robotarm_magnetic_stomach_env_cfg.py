"""Stomach environment variant of the validated robot-arm magnetic task."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from .robotarm_magnetic_lab_env_cfg import (
    RobotarmMagneticLabEnvCfg,
    RobotarmMagneticLabSceneCfg,
)


STOMACH_SCENE_USD_PATH = os.environ.get(
    "ROBOTARM_STOMACH_SCENE_USD",
    (
        "/mnt/isaac-linux/robotarm_magnetic_lab/assets/"
        "robotarm_magnetic_stomach_training.usda"
    ),
)
STOMACH_ASSET_USD_PATH = os.environ.get(
    "ROBOTARM_STOMACH_ASSET_USD",
    (
        "/mnt/isaac-linux/robotarm_magnetic_lab/assets/stomach/"
        "stomach_environment_lab.usda"
    ),
)
STOMACH_CAMERA_UPDATE_PERIOD_S = 1.0


@configclass
class RobotarmMagneticStomachSceneCfg(RobotarmMagneticLabSceneCfg):
    """Robot, ASM, passive capsule and static textured stomach wall."""

    scene_asset = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=STOMACH_SCENE_USD_PATH),
    )

    stomach = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stomach",
        spawn=sim_utils.UsdFileCfg(usd_path=STOMACH_ASSET_USD_PATH),
    )

    capsule_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet",
        update_period=0.0,
        # One policy step spans 12 physics substeps (20 Hz over 240 Hz).
        # Keep a margin so short impact peaks are not lost between policy
        # samples in the drop tests.
        history_length=16,
        track_pose=True,
        track_air_time=True,
        force_threshold=1.0e-4,
        # Aggregated normal force is intentional. The stomach collider is a
        # static triangle mesh (not a rigid-body prim), while every bench
        # collider is disabled in this task. A reported capsule contact is
        # therefore a stomach-wall contact. The tests reconstruct tangential
        # contact behavior from acceleration instead of relying on a filter.
        debug_vis=False,
    )

    # Bench-only fiducials are deactivated by the scene overlay. Setting these
    # inherited config entries to None prevents Isaac Lab from recreating them.
    fiducial_red = None
    fiducial_green = None
    fiducial_blue = None
    landmark_red = None
    landmark_green = None
    landmark_blue = None


@configclass
class RobotarmMagneticStomachLabEnvCfg(RobotarmMagneticLabEnvCfg):
    """Single-environment stomach bring-up task.

    The first version deliberately treats the stomach as a static textured
    triangle mesh. It validates placement, internal rendering and rigid
    capsule contact before deformable tissue mechanics are introduced.
    """

    scene: RobotarmMagneticStomachSceneCfg = RobotarmMagneticStomachSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # The shared bench task retains its validated 30 Hz camera. Only the
        # stomach-task camera is intentionally reduced to 1 Hz.
        self.scene.capsule_camera.update_period = STOMACH_CAMERA_UPDATE_PERIOD_S
        # Endoscopic illumination calibration. The former 250-intensity LEDs
        # plus a 500-intensity dome clipped the wall at the 1--2 mm working
        # distance. Use a weak ambient term and four warm, extended emitters:
        # the texture supplies the tissue color, while the larger source radius
        # softens specular highlights. Values remain provisional until they are
        # calibrated against the physical DS01 camera and capsule LEDs.
        # Even the previous 8-unit emitters and 15-unit dome clipped close
        # folds. Reduce direct and ambient energy again while keeping a finite
        # wet-tissue highlight, and enlarge the normalized source to broaden
        # its remaining specular shoulder.
        self.scene.dome_light.spawn.intensity = 5.0
        for light_name in (
            "capsule_led_top",
            "capsule_led_bottom",
            "capsule_led_left",
            "capsule_led_right",
        ):
            light = getattr(self.scene, light_name).spawn
            light.intensity = 4.0
            # Approximate the LED/lens diffuser as an extended source so the
            # millimetre-range highlight has a soft shoulder rather than a
            # saturated point.
            light.radius = 0.0030
            light.color_temperature = 4800.0
        self.viewer.eye = (1.28, 0.45, 0.32)
        self.viewer.lookat = (1.06, 0.115, 0.01)
