# Copyright (c) 2026, robotarm magnetic simulation contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-table benchmark for open-loop magnetic capsule motion."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from .robotarm_magnetic_lab_env_cfg import (
    CAPSULE_START_POS,
    RobotarmMagneticLabEnvCfg,
    RobotarmMagneticLabSceneCfg,
)


TABLE_SCENE_USD_PATH = (
    "/mnt/isaac-linux/robotarm_magnetic_lab/assets/"
    "robotarm_magnetic_table_training.usda"
)
TABLE_CAPSULE_SIDE_ROT_XYZW = (
    0.0,
    0.7071067811865476,
    0.0,
    0.7071067811865476,
)


@configclass
class RobotarmMagneticTableSceneCfg(RobotarmMagneticLabSceneCfg):
    """Robot/ASM, passive capsule and calibrated flat contact plane."""

    scene_asset = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_SCENE_USD_PATH),
    )

    capsule = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CAPSULE_START_POS,
            rot=TABLE_CAPSULE_SIDE_ROT_XYZW,
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    capsule_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet",
        update_period=0.0,
        history_length=16,
        track_pose=True,
        track_air_time=True,
        force_threshold=1.0e-4,
        debug_vis=False,
    )


@configclass
class RobotarmMagneticTableLabEnvCfg(RobotarmMagneticLabEnvCfg):
    """Single flat-table environment at 240 Hz physics and 20 Hz actions."""

    scene: RobotarmMagneticTableSceneCfg = RobotarmMagneticTableSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 180.0
        # The table benchmark needs enough arm workspace to create a useful
        # lateral field gradient.  The stomach task deliberately restricts
        # normalized arm actions to +/-0.05 rad; keep its frozen interface
        # untouched and widen only this dedicated benchmark to +/-0.25 rad.
        self.actions.joint_position.scale["j[1-6]"] = 0.25
        self.viewer.eye = (1.32, 0.53, 0.30)
        self.viewer.lookat = (1.0608, 0.1145, 0.035)
