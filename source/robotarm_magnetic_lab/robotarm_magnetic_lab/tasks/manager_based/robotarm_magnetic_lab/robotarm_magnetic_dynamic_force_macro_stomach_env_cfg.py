"""TASK-008 direct-force macros migrated into the flipped stomach scene."""

from isaaclab.utils.configclass import configclass

from .robotarm_magnetic_dynamic_force_macro_table_env_cfg import (
    DynamicForceMacroActionsCfg,
    DynamicForceMacroEventsCfg,
    DynamicForceMacroObservationsCfg,
    DynamicForceMacroRewardsCfg,
    DynamicForceMacroTerminationsCfg,
)
from .robotarm_magnetic_dynamic_force_stomach_env_cfg import RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg


# Manually accepted flat-table force levels migrated unchanged into the
# flipped-stomach task.  These are force-to-weight ratios, not magnetic-field
# gains: MOVE is the two-endpoint resultant, VIEW acts at the camera end, and
# UP is the equal/opposite endpoint-couple scale.
TASK008_STOMACH_MOVE_FORCE_RATIO = 0.40
TASK008_STOMACH_VIEW_FORCE_RATIO = 0.25
TASK008_STOMACH_UP_FORCE_RATIO = 0.85


@configclass
class RobotarmMagneticDynamicForceMacroStomachLabEnvCfg(RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg):
    actions: DynamicForceMacroActionsCfg = DynamicForceMacroActionsCfg()
    observations: DynamicForceMacroObservationsCfg = DynamicForceMacroObservationsCfg()
    events: DynamicForceMacroEventsCfg = DynamicForceMacroEventsCfg()
    rewards: DynamicForceMacroRewardsCfg = DynamicForceMacroRewardsCfg()
    terminations: DynamicForceMacroTerminationsCfg = DynamicForceMacroTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.dynamic_force_macro.move_force_ratio = TASK008_STOMACH_MOVE_FORCE_RATIO
        self.actions.dynamic_force_macro.view_force_ratio = TASK008_STOMACH_VIEW_FORCE_RATIO
        self.actions.dynamic_force_macro.up_force_ratio = TASK008_STOMACH_UP_FORCE_RATIO
        self.decimation = 4
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 4
        self.scene.capsule_camera.update_period = 1.0 / 30.0
