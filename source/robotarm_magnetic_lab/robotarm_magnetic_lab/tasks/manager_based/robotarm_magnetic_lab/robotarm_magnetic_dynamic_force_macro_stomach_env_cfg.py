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
# gains: MOVE is the two-endpoint resultant; VIEW and UP act at the camera end.
TASK008_STOMACH_MOVE_FORCE_RATIO = 0.40
TASK008_STOMACH_VIEW_FORCE_RATIO = 0.25
TASK008_STOMACH_UP_FORCE_RATIO = 0.85

# TASK-008-only reset at the opposite (Y-max-side) longitudinal quarter.
# The point was sampled from the flipped stomach's lower visual/collision
# surface at Y=Y_min+0.75*(Y_max-Y_min). Its lumen normal has Z=0.9984; the
# center includes the exact 13 x 25 mm capsule support distance plus 0.2 mm
# clearance while retaining the validated side-lying quaternion.
TASK008_CAPSULE_RIGHT_QUARTER_POS = (
    1.078678615084,
    0.177197008592,
    0.004293035807,
)


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
        self.scene.capsule.init_state.pos = TASK008_CAPSULE_RIGHT_QUARTER_POS
        self.decimation = 4
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 4
        self.scene.capsule_camera.update_period = 1.0 / 30.0
