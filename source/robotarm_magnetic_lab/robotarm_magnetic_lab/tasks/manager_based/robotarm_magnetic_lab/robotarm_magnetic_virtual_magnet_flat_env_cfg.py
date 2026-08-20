"""TASK-007 flat task: dynamic capsule and analytical virtual magnet only."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .robotarm_magnetic_lab_env_cfg import RobotarmMagneticLabEnvCfg
from .robotarm_magnetic_table_env_cfg import (
    RobotarmMagneticTableLabEnvCfg,
    RobotarmMagneticTableSceneCfg,
)


@configclass
class VirtualMagnetActionsCfg:
    """One public scalar plus one internal zero-dimensional physics hook."""

    # ActionTermCfg requires an asset_name even though the request adapter does
    # not actuate that asset. Both terms point only at the dynamic capsule and
    # never resolve or command robot/Ball joints.
    request = mdp.VirtualMagnetRequestActionCfg(asset_name="capsule")
    magnetic_physics = mdp.VirtualMagnetPhysicsActionCfg(asset_name="capsule")


@configclass
class VirtualMagnetObservationsCfg:
    """Exclude privileged capsule/contact/wrench truth from policy observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        action_status = ObsTerm(func=mdp.virtual_magnet_public_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class VirtualMagnetEventsCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    virtual_magnet_bridge = EventTerm(
        func=mdp.VirtualMagnetBridge,
        mode="interval",
        interval_range_s=(1.0, 1.0),
        is_global_time=False,
        resample_interval_on_reset=False,
        params={
            "asset_name": "capsule",
            "contact_sensor_name": "capsule_contact",
            "camera_sensor_name": "capsule_camera",
            "camera_mount_quaternion_capsule_xyzw": (0.0, 1.0, 0.0, 0.0),
            "profile_path": "",
            "debug_xform": True,
        },
    )


@configclass
class VirtualMagnetTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RobotarmMagneticVirtualMagnetFlatLabEnvCfg(RobotarmMagneticTableLabEnvCfg):
    """One action per second, with 240 Hz finite-model wrench feedback."""

    scene: RobotarmMagneticTableSceneCfg = RobotarmMagneticTableSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )
    actions: VirtualMagnetActionsCfg = VirtualMagnetActionsCfg()
    observations: VirtualMagnetObservationsCfg = VirtualMagnetObservationsCfg()
    events: VirtualMagnetEventsCfg = VirtualMagnetEventsCfg()
    terminations: VirtualMagnetTerminationsCfg = VirtualMagnetTerminationsCfg()

    def __post_init__(self) -> None:
        # Bypass the table task's legacy 9-D action edit while retaining its
        # calibrated scene. No manager action commands arm or Ball joints.
        RobotarmMagneticLabEnvCfg.__post_init__(self)
        self.decimation = 240
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 2
        self.episode_length_s = 300.0
        self.viewer.eye = (1.32, 0.53, 0.30)
        self.viewer.lookat = (1.0608, 0.1145, 0.035)
