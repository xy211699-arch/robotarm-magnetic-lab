"""Independent stage-one table task for the 11-action SMDP interface."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .robotarm_magnetic_lab_env_cfg import (
    ARM_JOINT_NAMES,
    BALL_JOINT_NAMES,
    RobotarmMagneticLabEnvCfg,
)
from .robotarm_magnetic_table_env_cfg import RobotarmMagneticTableLabEnvCfg


@configclass
class AtomicActionsCfg:
    """One scalar action plus a zero-dimensional physics-rate wrench hook."""

    atomic = mdp.AtomicMagnetActionCfg(asset_name="robot")
    magnetic_physics = mdp.MagneticPhysicsActionCfg(asset_name="capsule")


@configclass
class AtomicObservationsCfg:
    """Deployment signals only; privileged contact and wrench are excluded."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ARM_JOINT_NAMES + BALL_JOINT_NAMES
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ARM_JOINT_NAMES + BALL_JOINT_NAMES
                )
            },
        )
        external_magnet = ObsTerm(func=mdp.external_magnet_state)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RobotarmMagneticAtomicTableLabEnvCfg(RobotarmMagneticTableLabEnvCfg):
    """Scalar atomic-control task kept separate from the legacy 9-D task."""

    actions: AtomicActionsCfg = AtomicActionsCfg()
    observations: AtomicObservationsCfg = AtomicObservationsCfg()

    def __post_init__(self) -> None:
        # Skip RobotarmMagneticTableLabEnvCfg.__post_init__: it widens the
        # legacy 9-D joint_position action, which this task intentionally
        # replaces with one scalar atomic ID.
        RobotarmMagneticLabEnvCfg.__post_init__(self)
        self.episode_length_s = 180.0
        self.viewer.eye = (1.32, 0.53, 0.30)
        self.viewer.lookat = (1.0608, 0.1145, 0.035)
