"""Isolated real-dynamics capsule force task in the delivered stomach."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from . import mdp
from .robotarm_magnetic_lab_env_cfg import CAPSULE_CAMERA_CIRCULAR_FOV_DEG
from .robotarm_magnetic_stomach_env_cfg import RobotarmMagneticStomachLabEnvCfg


# TASK-003-only scene placement.  The shared stomach asset remains untouched so
# the other stomach tasks keep their validated geometry.  AssetBaseCfg poses in
# this Isaac Lab build use XYZW quaternions.  Rotating 180 degrees about world Y
# reverses world Z while preserving the stomach's longitudinal Y alignment.  The
# translation makes the rotation occur about the existing world AABB center, so
# its horizontal footprint and center do not move.
TASK003_STOMACH_FLIP_POS = (2.091607300678508, 0.0, 0.06585919085865222)
TASK003_STOMACH_FLIP_ROT_XYZW = (0.0, 1.0, 0.0, 0.0)

# The delivered stomach spans world Y=[-0.0107347368, 0.2403739599] m.  This
# reset lies one quarter of the way from the Y-min (left) end on the flatter
# post-flip lower wall.  Position and orientation follow the local surface
# normal/tangent and retain 0.2 mm minimum sampled clearance for the 13 x 25 mm
# capsule.  Its local Z/long axis is nearly horizontal (|axis dot world Z|=.080).
TASK003_CAPSULE_LEFT_QUARTER_POS = (
    1.04643745,
    0.05368121,
    0.00190115,
)
TASK003_CAPSULE_SIDE_ROT_XYZW = (
    0.73137642,
    0.07274179,
    0.67346616,
    0.07899674,
)


@configclass
class DynamicForceActionsCfg:
    dynamic_force: mdp.DynamicForceActionTermCfg = mdp.DynamicForceActionTermCfg(
        force_weight_ratio=0.9,
        vertical_force_weight_ratio=1.1,
    )


@configclass
class DynamicForceObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.capsule_rgb,
            params={
                "sensor_cfg": SceneEntityCfg("capsule_camera"),
                "field_of_view_deg": CAPSULE_CAMERA_CIRCULAR_FOV_DEG,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class DynamicForceEventsCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class DynamicForceRewardsCfg:
    pass


@configclass
class DynamicForceTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg(RobotarmMagneticStomachLabEnvCfg):
    """One CPU-PhysX environment with true dynamic capsule force input."""

    actions: DynamicForceActionsCfg = DynamicForceActionsCfg()
    observations: DynamicForceObservationsCfg = DynamicForceObservationsCfg()
    events: DynamicForceEventsCfg = DynamicForceEventsCfg()
    rewards: DynamicForceRewardsCfg = DynamicForceRewardsCfg()
    terminations: DynamicForceTerminationsCfg = DynamicForceTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.stomach.init_state.pos = TASK003_STOMACH_FLIP_POS
        self.scene.stomach.init_state.rot = TASK003_STOMACH_FLIP_ROT_XYZW
        self.scene.capsule.init_state.pos = TASK003_CAPSULE_LEFT_QUARTER_POS
        self.scene.capsule.init_state.rot = TASK003_CAPSULE_SIDE_ROT_XYZW
        self.scene.num_envs = 1
        self.decimation = 4
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 4
        self.scene.capsule_camera.update_period = 1.0 / 30.0
        self.episode_length_s = 1800.0
        # This Isaac Lab version explicitly disables scene CCD under GPU
        # dynamics. TASK-003 therefore uses single-environment CPU PhysX while
        # RTX rendering remains on the GPU, preserving the mandatory CCD gate.
        self.sim.device = "cpu"
        self.sim.physics = PhysxCfg(enable_ccd=True)
