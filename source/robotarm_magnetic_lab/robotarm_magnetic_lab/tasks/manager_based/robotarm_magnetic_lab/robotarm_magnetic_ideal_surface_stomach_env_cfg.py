"""Dedicated one-hertz privileged ideal-surface stomach task."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .robotarm_magnetic_lab_env_cfg import CAPSULE_CAMERA_CIRCULAR_FOV_DEG
from .robotarm_magnetic_stomach_env_cfg import RobotarmMagneticStomachLabEnvCfg


@configclass
class IdealSurfaceActionsCfg:
    """The only policy action is one scalar frozen ideal-surface action ID."""

    ideal_surface: mdp.IdealSurfaceActionTermCfg = mdp.IdealSurfaceActionTermCfg()


@configclass
class IdealSurfaceObservationsCfg:
    """Actor channel: deployable RGB only; all geometry remains privileged."""

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
class IdealSurfaceEventsCfg:
    """Reset-only events: no magnetic or legacy interval bridge."""

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class IdealSurfaceTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    ideal_surface_hard_failure = DoneTerm(func=mdp.ideal_surface_hard_failure)


@configclass
class RobotarmMagneticIdealSurfaceStomachTeleopLabEnvCfg(RobotarmMagneticStomachLabEnvCfg):
    """Existing stomach scene with isolated kinematic ideal capsule control."""

    actions: IdealSurfaceActionsCfg = IdealSurfaceActionsCfg()
    observations: IdealSurfaceObservationsCfg = IdealSurfaceObservationsCfg()
    events: IdealSurfaceEventsCfg = IdealSurfaceEventsCfg()
    terminations: IdealSurfaceTerminationsCfg = IdealSurfaceTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.decimation = 240
        self.sim.dt = 1.0 / 240.0
        # Exactly one render per 1 Hz Actor boundary is sufficient for the
        # 1 Hz task camera and avoids 30 redundant 720p renders per action.
        self.sim.render_interval = 240
        self.scene.capsule_camera.update_period = 1.0
        self.episode_length_s = 1800.0
