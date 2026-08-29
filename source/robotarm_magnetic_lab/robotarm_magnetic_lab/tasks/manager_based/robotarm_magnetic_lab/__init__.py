# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Robotarm-Magnetic-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotarm_magnetic_lab_env_cfg:RobotarmMagneticLabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Stomach-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_stomach_env_cfg:"
            "RobotarmMagneticStomachLabEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Table-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_table_env_cfg:"
            "RobotarmMagneticTableLabEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Atomic-Table-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_atomic_table_env_cfg:"
            "RobotarmMagneticAtomicTableLabEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_atomic_stomach_teleop_env_cfg:"
            "RobotarmMagneticAtomicStomachTeleopLabEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_ideal_surface_stomach_env_cfg:"
            "RobotarmMagneticIdealSurfaceStomachTeleopLabEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_dynamic_force_stomach_env_cfg:"
            "RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotarm_magnetic_dynamic_force_macro_table_env_cfg:RobotarmMagneticDynamicForceMacroTableLabEnvCfg",
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Dynamic-Force-Macro-Stomach-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotarm_magnetic_dynamic_force_macro_stomach_env_cfg:RobotarmMagneticDynamicForceMacroStomachLabEnvCfg",
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Parameterized-Force-Table-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_parameterized_force_table_env_cfg:"
            "RobotarmMagneticParameterizedForceTableLabEnvCfg"
        ),
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0",
    entry_point=(
        f"{__name__}.task009b_training_env:Task009BTrainingEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_parameterized_force_stomach_env_cfg:"
            "RobotarmMagneticParameterizedForceStomachCoverageLabEnvCfg"
        ),
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0",
    entry_point=f"{__name__}.task009d0_vector_env:Task009D0VectorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_task009d0_env_cfg:"
            "RobotarmMagneticTask009D0EnvCfg"
        ),
    },
)

gym.register(
    id="Template-Robotarm-Magnetic-Task010-CNN-GRU-Coverage-Lab-v0",
    entry_point=f"{__name__}.task010_vector_env:Task010VectorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robotarm_magnetic_task010_env_cfg:"
            "RobotarmMagneticTask010EnvCfg"
        ),
    },
)
