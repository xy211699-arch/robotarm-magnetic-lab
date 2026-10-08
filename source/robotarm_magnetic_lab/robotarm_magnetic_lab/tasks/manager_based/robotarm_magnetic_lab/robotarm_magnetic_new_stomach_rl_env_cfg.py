"""Independent new-stomach RL configuration; archived task remains unchanged."""
from isaaclab.utils.configclass import configclass
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from .robotarm_magnetic_new_stomach_vector_env_cfg import RobotarmMagneticNewStomachVectorEnvCfg, NEW_STOMACH_COLLIDER
from .mdp.new_stomach_rl_action import NewStomachRLActionCfg
from .mdp.new_stomach_rl_terms import actor_observation, privileged_critic_observation, summed_four_term_reward


@configclass
class NewStomachRLRewardsCfg:
    four_terms = RewardTermCfg(func=summed_four_term_reward, weight=1.)


@configclass
class NewStomachRLObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        allowed_inputs = ObservationTermCfg(func=actor_observation)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class Critic(ObservationGroupCfg):
        privileged_inputs = ObservationTermCfg(func=privileged_critic_observation)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy = Policy()
    critic = Critic()


@configclass
class RobotarmMagneticNewStomachRLPreflightCfg(RobotarmMagneticNewStomachVectorEnvCfg):
    group: str = 'C'
    observations = NewStomachRLObservationsCfg()
    rewards = NewStomachRLRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.group not in ('A','B','C','D'):
            raise ValueError('preflight group A/B/C/D required')
        self.actions.magnet = NewStomachRLActionCfg(asset_name='robot', world_mesh_path=NEW_STOMACH_COLLIDER,
            mode='single' if self.group in ('A','B') else 'chunk')
        self.decimation = self.sim.render_interval = 240
        self.scene.capsule_camera.update_period = 1.
        self.scene.capsule_camera_preview = None
        self.compute_final_obs = True
        self.episode_length_s = 120.
