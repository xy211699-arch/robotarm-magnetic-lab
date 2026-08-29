"""Independent TASK-010 environment configuration."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from robotarm_magnetic_lab.runtime.task010_config import TASK010_CONFIG_PATH, load_task010_config

from .robotarm_magnetic_task009d0_env_cfg import RobotarmMagneticTask009D0EnvCfg
from . import mdp


_TASK010 = load_task010_config(TASK010_CONFIG_PATH)


@configclass
class Task010ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actor = ObsTerm(func=mdp.task010_actor_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        critic = ObsTerm(func=mdp.task010_privileged_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class Task010RewardsCfg:
    task_reward = RewTerm(func=mdp.task010_total_reward, weight=1.0)


@configclass
class RobotarmMagneticTask010EnvCfg(RobotarmMagneticTask009D0EnvCfg):
    """TASK-010 uses the D0 scene and controller with an independent lifecycle."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = _TASK010.training.num_envs
        self.sim.dt = 1.0 / _TASK010.clocks.physics_hz
        self.decimation = _TASK010.clocks.physics_steps_per_action
        self.scene.capsule_camera.update_period = 1.0 / _TASK010.camera.hz
        self.scene.capsule_camera.width = _TASK010.camera.width
        self.scene.capsule_camera.height = _TASK010.camera.height
        self.episode_length_s = _TASK010.episode.duration_s
    observations: Task010ObservationsCfg = Task010ObservationsCfg()
    rewards: Task010RewardsCfg = Task010RewardsCfg()
