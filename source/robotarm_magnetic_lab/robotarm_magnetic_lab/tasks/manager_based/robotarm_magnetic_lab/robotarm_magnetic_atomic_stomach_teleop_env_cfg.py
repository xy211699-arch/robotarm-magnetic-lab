"""Dedicated one-environment atomic stomach teleoperation task."""

from isaaclab.utils.configclass import configclass

from .robotarm_magnetic_atomic_table_env_cfg import (
    AtomicActionsCfg,
    AtomicObservationsCfg,
    AtomicTerminationsCfg,
)
from .robotarm_magnetic_stomach_env_cfg import RobotarmMagneticStomachLabEnvCfg


@configclass
class RobotarmMagneticAtomicStomachTeleopLabEnvCfg(RobotarmMagneticStomachLabEnvCfg):
    """Compose the frozen scalar executor with the existing stomach scene."""

    actions: AtomicActionsCfg = AtomicActionsCfg()
    observations: AtomicObservationsCfg = AtomicObservationsCfg()
    terminations: AtomicTerminationsCfg = AtomicTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1800.0
