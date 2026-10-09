"""Separate vector task; physics/assets/safety and original task are preserved."""
from isaaclab.utils.configclass import configclass
from .robotarm_magnetic_new_stomach_rl_env_cfg import RobotarmMagneticNewStomachRLPreflightCfg
from .robotarm_magnetic_new_stomach_env_cfg import NEW_STOMACH_GEOMETRY
from .mdp.new_stomach_rl_vector_action import NewStomachRLVectorActionCfg
from .mdp.new_stomach_rl_vector_magnetic import NewStomachRLVectorMagnetic
from .mdp.new_stomach_rl_vector_terms import actor_observation,privileged_critic_observation,audited_vector_collision


@configclass
class RobotarmMagneticNewStomachRLVectorCfg(RobotarmMagneticNewStomachRLPreflightCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.magnet = NewStomachRLVectorActionCfg(asset_name='robot',
            mode='single' if self.group in ('A','B') else 'chunk',
            collision_mesh_suffix=NEW_STOMACH_GEOMETRY.collision_mesh_suffix)
        self.events.magnetic_collision_bridge.func = NewStomachRLVectorMagnetic
        self.terminations.collision.func = audited_vector_collision
        self.observations.policy.allowed_inputs.func = actor_observation
        self.observations.critic.privileged_inputs.func = privileged_critic_observation
        # The new step adapter calls each old reward accumulator explicitly,
        # avoiding synchronous reward/automatic reset in ManagerBasedRLEnv.
        self.rewards = None
