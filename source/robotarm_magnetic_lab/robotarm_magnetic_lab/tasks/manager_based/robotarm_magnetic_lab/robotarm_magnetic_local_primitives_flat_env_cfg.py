"""Isolated flat-contact task for TASK-004 local dynamics primitives."""

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
class LocalPrimitiveActionsCfg:
    local_primitive: mdp.LocalPrimitiveActionTermCfg = mdp.make_local_primitive_action_cfg()


@configclass
class LocalPrimitiveObservationsCfg:
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
class LocalPrimitiveEventsCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class LocalPrimitiveRewardsCfg:
    pass


@configclass
class LocalPrimitiveTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RobotarmMagneticLocalPrimitivesFlatLabEnvCfg(RobotarmMagneticTableLabEnvCfg):
    """One CPU-PhysX flat task with only the local-primitive action."""

    actions: LocalPrimitiveActionsCfg = LocalPrimitiveActionsCfg()
    observations: LocalPrimitiveObservationsCfg = LocalPrimitiveObservationsCfg()
    events: LocalPrimitiveEventsCfg = LocalPrimitiveEventsCfg()
    rewards: LocalPrimitiveRewardsCfg = LocalPrimitiveRewardsCfg()
    terminations: LocalPrimitiveTerminationsCfg = LocalPrimitiveTerminationsCfg()

    def __post_init__(self) -> None:
        # Skip the table task's action-specific joint-range edit because this
        # isolated task deliberately has no joint action term.
        RobotarmMagneticLabEnvCfg.__post_init__(self)
        self.scene.num_envs = 1
        self.decimation = 4
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 4
        self.scene.capsule_camera.update_period = 1.0 / 30.0
        self.episode_length_s = 180.0
        self.sim.device = "cpu"
        self.sim.physics = PhysxCfg(enable_ccd=True)
        self.viewer.eye = (1.32, 0.53, 0.30)
        self.viewer.lookat = (1.0608, 0.1145, 0.035)
