"""Bind the existing 4x9 actuator controller to the audited new-stomach mesh.

This module performs only scene-specific identity checks. Trajectory planning,
magnetic force application, and collision algorithms remain in the reused modules.
"""

from __future__ import annotations

import hashlib

from isaaclab.utils.configclass import configclass

from .actuator_vector_action import ActuatorVectorAction, ActuatorVectorActionCfg


class NewStomachActuatorVectorAction(ActuatorVectorAction):
    """Require the confirmed collision mesh before executing any command."""

    def _geometry(self):
        if getattr(self, "_new_stomach_runtime", None) is None:
            import omni.usd

            from robotarm_magnetic_lab.geometry.new_stomach_runtime import NewStomachRuntimeGeometry
            from ..robotarm_magnetic_new_stomach_env_cfg import (
                NEW_STOMACH_ASSET_USD_PATH,
                NEW_STOMACH_GEOMETRY,
            )

            digest = hashlib.sha256(open(NEW_STOMACH_ASSET_USD_PATH, "rb").read()).hexdigest()
            if digest != NEW_STOMACH_GEOMETRY.asset_usd_sha256:
                raise RuntimeError("new-stomach USD asset hash differs from confirmed configuration")
            runtime = NewStomachRuntimeGeometry.from_stage(
                omni.usd.get_context().get_stage(), NEW_STOMACH_GEOMETRY
            )
            if runtime.collision_mesh_path != self.cfg.world_mesh_path:
                raise RuntimeError(
                    f"planner mesh {self.cfg.world_mesh_path} differs from "
                    f"audited collider {runtime.collision_mesh_path}"
                )
            self._new_stomach_runtime = runtime
        return super()._geometry()


@configclass
class NewStomachActuatorVectorActionCfg(ActuatorVectorActionCfg):
    class_type: type = NewStomachActuatorVectorAction

def audited_new_stomach_collision(env):
    """Reuse the old screen's exact current-state clearance decision."""
    import numpy as np
    import torch

    from ..controllers.batched_arm_clearance import (
        minimum_clearance_fast,
        minimum_world_clearance,
    )

    term = env.action_manager.get_term("magnet")
    term._geometry()
    q = term.robot.data.joint_pos.torch[0, term.ids[:6]].detach().cpu().numpy()[None]
    own = float(minimum_clearance_fast(term.kinematics, q)[0])
    wall = float(minimum_world_clearance(term.world_checker, q)[0])
    gap = min(own, wall)
    env._new_stomach_clearance = {
        "self_m": own,
        "world_m": wall,
        "required_m": term.cfg.required_clearance_m,
    }
    return torch.tensor(
        [not np.isfinite([own, wall]).all() or gap < term.cfg.required_clearance_m],
        device=env.device,
    )
