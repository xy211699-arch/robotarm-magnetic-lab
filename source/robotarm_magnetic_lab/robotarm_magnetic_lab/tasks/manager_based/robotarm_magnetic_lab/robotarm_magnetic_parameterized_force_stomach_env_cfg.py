"""TASK-009B stomach environment for the audited 10 Hz force controller."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from . import mdp
from .robotarm_magnetic_lab_env_cfg import CAPSULE_CAMERA_CIRCULAR_FOV_DEG
from .robotarm_magnetic_stomach_env_cfg import (
    RobotarmMagneticStomachLabEnvCfg,
    RobotarmMagneticStomachSceneCfg,
)


# These scene-only transforms reproduce the accepted flipped stomach and the
# non-penetrating opposite-quarter side pose.  The TASK-009B pose library will
# replace the capsule reset after its entrance box is confirmed.
TASK009B_STOMACH_FLIP_POS = (2.091607300678508, 0.0, 0.06585919085865222)
TASK009B_STOMACH_FLIP_ROT_XYZW = (0.0, 1.0, 0.0, 0.0)
TASK009B_INTEGRATION_CAPSULE_POS = (1.078678615084, 0.177197008592, 0.011793035807)
TASK009B_INTEGRATION_CAPSULE_ROT_XYZW = (
    0.73137642,
    0.07274179,
    0.67346616,
    0.07899674,
)


@configclass
class ParameterizedForceStomachActionsCfg:
    parameterized_force = mdp.ParameterizedForceActionTermCfg()


@configclass
class ParameterizedForceStomachObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.capsule_rgb,
            params={
                "sensor_cfg": SceneEntityCfg("capsule_camera"),
                "field_of_view_deg": CAPSULE_CAMERA_CIRCULAR_FOV_DEG,
                "require_new_control_boundary_frame": True,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ParameterizedForceStomachEventsCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class ParameterizedForceStomachRewardsCfg:
    pass


@configclass
class ParameterizedForceStomachTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RobotarmMagneticParameterizedForceStomachCoverageLabEnvCfg(
    RobotarmMagneticStomachLabEnvCfg
):
    """Single stomach environment with one exact 0.1 s action boundary."""

    scene: RobotarmMagneticStomachSceneCfg = RobotarmMagneticStomachSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )
    actions: ParameterizedForceStomachActionsCfg = ParameterizedForceStomachActionsCfg()
    observations: ParameterizedForceStomachObservationsCfg = (
        ParameterizedForceStomachObservationsCfg()
    )
    events: ParameterizedForceStomachEventsCfg = ParameterizedForceStomachEventsCfg()
    rewards: ParameterizedForceStomachRewardsCfg = ParameterizedForceStomachRewardsCfg()
    terminations: ParameterizedForceStomachTerminationsCfg = (
        ParameterizedForceStomachTerminationsCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.stomach.init_state.pos = TASK009B_STOMACH_FLIP_POS
        self.scene.stomach.init_state.rot = TASK009B_STOMACH_FLIP_ROT_XYZW
        self.scene.capsule.init_state.pos = TASK009B_INTEGRATION_CAPSULE_POS
        self.scene.capsule.init_state.rot = TASK009B_INTEGRATION_CAPSULE_ROT_XYZW
        self.scene.num_envs = 1
        self.sim.dt = 1.0 / 240.0
        self.decimation = 24
        # External viewport is 60 FPS. Policy RGB is acquired only at the
        # 10 Hz environment boundary, independently of repeated UI display.
        self.sim.render_interval = 4
        self.scene.capsule_camera.update_period = 1.0 / 10.0
        self.sim.physics = PhysxCfg(enable_ccd=True)
        self.episode_length_s = 1800.0
