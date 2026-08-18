"""TASK-004 stomach wrapper using the frozen flat primitive profile unchanged."""

from isaaclab.utils.configclass import configclass

from . import mdp
from .robotarm_magnetic_dynamic_force_stomach_env_cfg import (
    RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg,
)


@configclass
class LocalPrimitiveStomachActionsCfg:
    """The only action in this scene is the shared local-primitive term."""

    local_primitive: mdp.LocalPrimitiveActionTermCfg = mdp.make_local_primitive_action_cfg()


@configclass
class RobotarmMagneticLocalPrimitivesStomachLabEnvCfg(
    RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg
):
    """TASK-003 scene/reset with only the frozen TASK-004 wrench controller."""

    actions: LocalPrimitiveStomachActionsCfg = LocalPrimitiveStomachActionsCfg()

