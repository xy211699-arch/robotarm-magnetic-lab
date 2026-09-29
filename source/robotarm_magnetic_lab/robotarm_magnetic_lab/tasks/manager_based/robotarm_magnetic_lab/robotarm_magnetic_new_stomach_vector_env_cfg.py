"""Independent new-stomach task using the existing magnetic actuator controller."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .mdp.magnetic_action import MagneticPhysicsActionCfg
from .mdp.new_stomach_actuator_action import (
    NewStomachActuatorVectorActionCfg,
    audited_new_stomach_collision,
)
from .robotarm_magnetic_new_stomach_env_cfg import (
    NEW_STOMACH_GEOMETRY,
    RobotarmMagneticNewStomachLabEnvCfg,
)


NEW_STOMACH_COLLIDER = (
    "/World/envs/env_0/Stomach" + NEW_STOMACH_GEOMETRY.collision_mesh_suffix
)


@configclass
class NewStomachVectorActionsCfg:
    magnet = NewStomachActuatorVectorActionCfg(
        asset_name="robot",
        world_mesh_path=NEW_STOMACH_COLLIDER,
    )
    magnetic_physics = MagneticPhysicsActionCfg(asset_name="capsule")


@configclass
class RobotarmMagneticNewStomachVectorEnvCfg(RobotarmMagneticNewStomachLabEnvCfg):
    """One-second command boundaries; unchanged 240 Hz physics and magnetic bridge."""

    actions: NewStomachVectorActionsCfg = NewStomachVectorActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 240
        self.sim.render_interval = 240
        self.terminations.collision.func = audited_new_stomach_collision
        self.episode_length_s = 310.0
