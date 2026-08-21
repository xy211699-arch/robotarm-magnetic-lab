"""TASK-008 isolated flat-table six-action force task."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from . import mdp
from .robotarm_magnetic_lab_env_cfg import CAPSULE_CAMERA_CIRCULAR_FOV_DEG, RobotarmMagneticLabEnvCfg
from .robotarm_magnetic_table_env_cfg import RobotarmMagneticTableLabEnvCfg, RobotarmMagneticTableSceneCfg


@configclass
class DynamicForceMacroActionsCfg:
    dynamic_force_macro = mdp.DynamicForceMacroActionTermCfg()


@configclass
class DynamicForceMacroObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.capsule_rgb,
            params={"sensor_cfg": SceneEntityCfg("capsule_camera"), "field_of_view_deg": CAPSULE_CAMERA_CIRCULAR_FOV_DEG},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class DynamicForceMacroEventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class DynamicForceMacroRewardsCfg:
    pass


@configclass
class DynamicForceMacroTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RobotarmMagneticDynamicForceMacroTableLabEnvCfg(RobotarmMagneticTableLabEnvCfg):
    scene: RobotarmMagneticTableSceneCfg = RobotarmMagneticTableSceneCfg(num_envs=1, env_spacing=4.0)
    actions: DynamicForceMacroActionsCfg = DynamicForceMacroActionsCfg()
    observations: DynamicForceMacroObservationsCfg = DynamicForceMacroObservationsCfg()
    events: DynamicForceMacroEventsCfg = DynamicForceMacroEventsCfg()
    rewards: DynamicForceMacroRewardsCfg = DynamicForceMacroRewardsCfg()
    terminations: DynamicForceMacroTerminationsCfg = DynamicForceMacroTerminationsCfg()

    def __post_init__(self):
        RobotarmMagneticLabEnvCfg.__post_init__(self)
        self.scene.num_envs = 1
        self.decimation = 4
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 4
        self.scene.capsule_camera.update_period = 1.0 / 30.0
        self.sim.device = "cpu"
        self.sim.physics = PhysxCfg(enable_ccd=True)
        self.episode_length_s = 1800.0


@configclass
class RobotarmMagneticDynamicForceMacroTableProfileCfg(RobotarmMagneticDynamicForceMacroTableLabEnvCfg):
    """Config entry point accepting force ratios through explicit replacement."""
