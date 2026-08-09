"""Dedicated one-environment atomic stomach teleoperation task."""

from isaaclab.utils.configclass import configclass

from . import mdp
from .robotarm_magnetic_atomic_table_env_cfg import (
    AtomicActionsCfg,
    AtomicObservationsCfg,
    AtomicTerminationsCfg,
)
from .robotarm_magnetic_stomach_env_cfg import RobotarmMagneticStomachLabEnvCfg


STOMACH_COLLISION_MESH_PRIM_PATH = (
    "/World/envs/env_0/Stomach/ConvertedSource/Environment/Stomach/"
    "Physics_Collision_Mesh/Stomach"
)


@configclass
class AtomicStomachActionsCfg(AtomicActionsCfg):
    """Atomic actions with robot/ASM-only stomach mesh protection enabled."""

    atomic: mdp.AtomicMagnetActionCfg = mdp.AtomicMagnetActionCfg(
        asset_name="robot",
        environment_collision_mesh_prim_path=STOMACH_COLLISION_MESH_PRIM_PATH,
        environment_collision_clearance_m=0.004,
    )


@configclass
class RobotarmMagneticAtomicStomachTeleopLabEnvCfg(RobotarmMagneticStomachLabEnvCfg):
    """Compose the frozen scalar executor with the existing stomach scene."""

    actions: AtomicStomachActionsCfg = AtomicStomachActionsCfg()
    observations: AtomicObservationsCfg = AtomicObservationsCfg()
    terminations: AtomicTerminationsCfg = AtomicTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1800.0
