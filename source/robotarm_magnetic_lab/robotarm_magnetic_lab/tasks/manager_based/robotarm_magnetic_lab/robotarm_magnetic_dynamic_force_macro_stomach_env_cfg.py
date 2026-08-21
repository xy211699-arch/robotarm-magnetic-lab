"""TASK-008 stomach viewer using the unchanged table-selected profile."""

from isaaclab.utils.configclass import configclass

from .robotarm_magnetic_dynamic_force_macro_table_env_cfg import (
    DynamicForceMacroActionsCfg,
    DynamicForceMacroEventsCfg,
    DynamicForceMacroObservationsCfg,
    DynamicForceMacroRewardsCfg,
    DynamicForceMacroTerminationsCfg,
)
from .robotarm_magnetic_dynamic_force_stomach_env_cfg import RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg


@configclass
class RobotarmMagneticDynamicForceMacroStomachLabEnvCfg(RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg):
    actions: DynamicForceMacroActionsCfg = DynamicForceMacroActionsCfg()
    observations: DynamicForceMacroObservationsCfg = DynamicForceMacroObservationsCfg()
    events: DynamicForceMacroEventsCfg = DynamicForceMacroEventsCfg()
    rewards: DynamicForceMacroRewardsCfg = DynamicForceMacroRewardsCfg()
    terminations: DynamicForceMacroTerminationsCfg = DynamicForceMacroTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 4
        self.scene.capsule_camera.update_period = 1.0 / 30.0
