"""Class-name entry configuration for the project-local TASK-010 learning stack."""

from isaaclab.utils.configclass import configclass


@configclass
class Task010RslRlPpoCfg:
    class_name: str = "robotarm_magnetic_lab.learning.task010_ppo:Task010PPO"
    runner_class_name: str = "robotarm_magnetic_lab.learning.task010_runner:Task010OnPolicyRunner"
    actor_class_name: str = "robotarm_magnetic_lab.learning.task010_actor:Task010Actor"
    critic_class_name: str = "robotarm_magnetic_lab.learning.task010_critic:Task010Critic"
    num_steps_per_env: int = 64
    max_iterations: int = 1000
    save_interval: int = 50
    seed: int = 991000
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    learning_rate: float = 3.0e-4
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    gamma: float = 0.999
    lam: float = 0.95
    entropy_coef: float = 0.005
