"""Frozen synchronous vector-training configuration for TASK-009D0."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from robotarm_magnetic_lab.runtime.task009d0_config import (
    TASK009D0_CONFIG_PATH,
    load_task009d0_config,
)

from . import mdp
from .robotarm_magnetic_lab_env_cfg import CAPSULE_CAMERA_CIRCULAR_FOV_DEG
from .robotarm_magnetic_parameterized_force_stomach_env_cfg import (
    ParameterizedForceStomachEventsCfg,
    RobotarmMagneticParameterizedForceStomachCoverageLabEnvCfg,
)
from .robotarm_magnetic_stomach_env_cfg import RobotarmMagneticStomachSceneCfg


_TASK009D0 = load_task009d0_config(TASK009D0_CONFIG_PATH)


@configclass
class Task009D0ActionsCfg:
    parameterized_force = mdp.VectorizedParameterizedForceActionTermCfg()


@configclass
class Task009D0ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.task009d0_rgb,
            params={
                "sensor_cfg": SceneEntityCfg("capsule_camera"),
                "field_of_view_deg": CAPSULE_CAMERA_CIRCULAR_FOV_DEG,
            },
        )
        previous_action = ObsTerm(func=mdp.task009d0_previous_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PrivilegedCfg(ObsGroup):
        capsule_state = ObsTerm(func=mdp.task009d0_privileged_capsule_state)
        area_state = ObsTerm(func=mdp.task009d0_privileged_coverage)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class Task009D0RewardsCfg:
    new_coverage = RewTerm(func=mdp.task009d0_new_coverage, weight=1.0)


@configclass
class Task009D0TerminationsCfg:
    pass


@configclass
class RobotarmMagneticTask009D0EnvCfg(
    RobotarmMagneticParameterizedForceStomachCoverageLabEnvCfg
):
    """Additive vector task; accepted TASK-009B remains untouched."""

    scene: RobotarmMagneticStomachSceneCfg = RobotarmMagneticStomachSceneCfg(
        num_envs=_TASK009D0["num_env_candidates"][0],
        env_spacing=_TASK009D0["env_spacing_m"],
    )
    actions: Task009D0ActionsCfg = Task009D0ActionsCfg()
    observations: Task009D0ObservationsCfg = Task009D0ObservationsCfg()
    events: ParameterizedForceStomachEventsCfg = ParameterizedForceStomachEventsCfg()
    rewards: Task009D0RewardsCfg = Task009D0RewardsCfg()
    terminations: Task009D0TerminationsCfg = Task009D0TerminationsCfg()
    pose_split: str = "train"
    explicit_pose_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = _TASK009D0["num_env_candidates"][0]
        self.scene.env_spacing = _TASK009D0["env_spacing_m"]
        self.sim.dt = 1.0 / _TASK009D0["clocks"]["physics_hz"]
        self.decimation = _TASK009D0["clocks"]["physics_steps_per_action"]
        self.scene.capsule_camera.update_period = 1.0 / _TASK009D0["camera"]["hz"]
        self.scene.capsule_camera.width = _TASK009D0["camera"]["width"]
        self.scene.capsule_camera.height = _TASK009D0["camera"]["height"]
        self.episode_length_s = _TASK009D0["episode"]["duration_s"]
        if self.pose_split not in ("train", "validation", "test"):
            raise ValueError("TASK-009D0 pose_split must be train, validation, or test")
        if self.pose_split == "train" and self.explicit_pose_ids is not None:
            raise ValueError("training mode rejects explicit validation/test pose IDs")
