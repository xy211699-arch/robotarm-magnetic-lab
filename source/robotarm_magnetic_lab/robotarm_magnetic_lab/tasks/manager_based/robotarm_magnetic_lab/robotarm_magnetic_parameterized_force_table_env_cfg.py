"""Flat-table task for the 240 Hz physics / 10 Hz force interface."""

from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from . import mdp
from .robotarm_magnetic_dynamic_force_macro_table_env_cfg import (
    DynamicForceMacroEventsCfg,
    DynamicForceMacroObservationsCfg,
    DynamicForceMacroRewardsCfg,
    DynamicForceMacroTerminationsCfg,
)
from .robotarm_magnetic_lab_env_cfg import RobotarmMagneticLabEnvCfg
from .robotarm_magnetic_table_env_cfg import RobotarmMagneticTableSceneCfg


@configclass
class ParameterizedForceActionsCfg:
    parameterized_force = mdp.ParameterizedForceActionTermCfg()


@configclass
class RobotarmMagneticParameterizedForceTableLabEnvCfg(RobotarmMagneticLabEnvCfg):
    """One environment step is one exact 0.1 s force-control period."""

    scene: RobotarmMagneticTableSceneCfg = RobotarmMagneticTableSceneCfg(num_envs=1, env_spacing=4.0)
    actions: ParameterizedForceActionsCfg = ParameterizedForceActionsCfg()
    observations: DynamicForceMacroObservationsCfg = DynamicForceMacroObservationsCfg()
    events: DynamicForceMacroEventsCfg = DynamicForceMacroEventsCfg()
    rewards: DynamicForceMacroRewardsCfg = DynamicForceMacroRewardsCfg()
    terminations: DynamicForceMacroTerminationsCfg = DynamicForceMacroTerminationsCfg()

    def __post_init__(self) -> None:
        RobotarmMagneticLabEnvCfg.__post_init__(self)
        self.scene.num_envs = 1
        self.sim.dt = 1.0 / 240.0
        self.decimation = 24
        # External viewport can render at 120 Hz while the policy camera keeps
        # its independent 30 Hz sensor contract.
        self.sim.render_interval = 2
        self.scene.capsule_camera.update_period = 1.0 / 30.0
        self.sim.device = "cpu"
        self.sim.physics = PhysxCfg(enable_ccd=True)
        self.episode_length_s = 1800.0
        self.viewer.eye = (1.18, 0.32, 0.16)
        self.viewer.lookat = (1.0608, 0.1145, 0.015)
