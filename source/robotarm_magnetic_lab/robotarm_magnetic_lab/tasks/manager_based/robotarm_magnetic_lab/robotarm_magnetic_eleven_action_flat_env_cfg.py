"""TASK-005 one-environment flat dynamic eleven-action task."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from . import mdp
from .robotarm_magnetic_lab_env_cfg import (
    CAPSULE_CAMERA_CIRCULAR_FOV_DEG,
    RobotarmMagneticLabEnvCfg,
)
from .robotarm_magnetic_table_env_cfg import RobotarmMagneticTableLabEnvCfg


@configclass
class ElevenActionFlatActionsCfg:
    eleven_action: mdp.ElevenActionTermCfg = mdp.ElevenActionTermCfg(surface_kind="flat")


@configclass
class ElevenActionObservationsCfg:
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
class ElevenActionEventsCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class ElevenActionRewardsCfg:
    pass


@configclass
class ElevenActionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    controller_fault = DoneTerm(func=mdp.eleven_action_fault)


@configclass
class RobotarmMagneticElevenActionFlatLabEnvCfg(RobotarmMagneticTableLabEnvCfg):
    actions: ElevenActionFlatActionsCfg = ElevenActionFlatActionsCfg()
    observations: ElevenActionObservationsCfg = ElevenActionObservationsCfg()
    events: ElevenActionEventsCfg = ElevenActionEventsCfg()
    rewards: ElevenActionRewardsCfg = ElevenActionRewardsCfg()
    terminations: ElevenActionTerminationsCfg = ElevenActionTerminationsCfg()

    def __post_init__(self) -> None:
        # Preserve the validated TASK-004 scene, reset, contact and camera.  The
        # caller-selected device remains authoritative; this task does not force CPU.
        RobotarmMagneticLabEnvCfg.__post_init__(self)
        self.scene.num_envs = 1
        self.decimation = 4
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 2
        self.episode_length_s = 180.0
        self.sim.physics = PhysxCfg(enable_ccd=True)
        self.viewer.eye = (1.32, 0.53, 0.30)
        self.viewer.lookat = (1.0608, 0.1145, 0.035)

