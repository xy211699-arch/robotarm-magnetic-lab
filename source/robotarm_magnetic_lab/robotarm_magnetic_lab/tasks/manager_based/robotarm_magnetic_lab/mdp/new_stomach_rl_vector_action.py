"""N independent bindings of the unchanged RL trajectory and stopping code."""
import hashlib
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_rows import RowEnvironment, row_ids
from .new_stomach_rl_action import NewStomachRLAction, NewStomachRLActionCfg


class ScopedRLRowAction(NewStomachRLAction):
    """Only scene binding changes; inherited planning/servo/stop are untouched."""
    def _geometry(self):
        import omni.usd
        from pxr import UsdGeom
        from robotarm_magnetic_lab.geometry.new_stomach_runtime import NewStomachRuntimeGeometry
        from ..robotarm_magnetic_new_stomach_env_cfg import NEW_STOMACH_GEOMETRY, NEW_STOMACH_ASSET_USD_PATH
        from ..controllers.ball_envelope import audit_stage, install
        from ..controllers.mesh_cover_audit import audit_stage_static_meshes
        from ..controllers.native_mesh_clearance import NativeMeshClearance
        from .world_collision import StomachMeshCollisionChecker
        stage = omni.usd.get_context().get_stage()
        root = f'/World/envs/env_{self._env.row}'
        if self.envelope is None:
            if hashlib.sha256(open(NEW_STOMACH_ASSET_USD_PATH,'rb').read()).hexdigest() != NEW_STOMACH_GEOMETRY.asset_usd_sha256:
                raise RuntimeError('confirmed stomach asset changed')
            transform = np.asarray(UsdGeom.XformCache(0.).GetLocalToWorldTransform(stage.GetPrimAtPath(root)))
            if not np.array_equal(transform[:3,:3], np.eye(3)):
                raise RuntimeError('environment clone must be translation-only')
            origin = transform[3,:3]
            config = replace(NEW_STOMACH_GEOMETRY,
                position_world_m=tuple(np.asarray(NEW_STOMACH_GEOMETRY.position_world_m)+origin),
                world_bounds_min_m=tuple(np.asarray(NEW_STOMACH_GEOMETRY.world_bounds_min_m)+origin),
                world_bounds_max_m=tuple(np.asarray(NEW_STOMACH_GEOMETRY.world_bounds_max_m)+origin),
                opening_centers_world_m=tuple(map(tuple,np.asarray(NEW_STOMACH_GEOMETRY.opening_centers_world_m)+origin)))
            self._new_stomach_runtime = NewStomachRuntimeGeometry.from_stage(stage,config,root+'/Stomach')
            if self._new_stomach_runtime.collision_mesh_path != self.cfg.world_mesh_path:
                raise RuntimeError('row planner/collider binding differs')
            self.env_origin = origin.copy()
            self.mesh_cover_audit = audit_stage_static_meshes(stage,self.kinematics,repair_native=True,env_root=root)
            if not self.mesh_cover_audit['certified']:
                raise RuntimeError(f'row {self._env.row} native mesh cover failed: {self.mesh_cover_audit}')
            audit = audit_stage(stage,env_root=root)
            p,r = self.pose(self.ee)
            center = r.T@(self.pose(self.source)[0]-p)
            install(self.kinematics,center,audit['radius_m'])
            bridge = self._env.parent.event_manager.get_term_cfg('magnetic_collision_bridge').func
            self.kinematics._native_narrow = NativeMeshClearance(stage,self.kinematics,str(self._env.device),
                asm_mount=bridge.config['planning']['asm_mount_l6_from_asm'],
                ball_envelope=(center,audit['radius_m']),env_root=root)
            self.envelope = dict(audit,center_l6_m=center.tolist(),env_root=root)
            if str(self._env.device).startswith('cuda'):
                self.kinematics._clearance_device = str(self._env.device)
        if self.world_checker is None:
            p,r = self.pose(self.base)
            self.world_checker = StomachMeshCollisionChecker(self.kinematics,
                mesh_prim_path=self.cfg.world_mesh_path,robot_base_position_world_m=p,
                robot_base_rotation_world=r,device=str(self._env.device),
                required_clearance_m=self.cfg.required_clearance_m,trajectory_samples=25)


class NewStomachRLVectorAction(ActionTerm):
    """Per-row old one-second action, with explicit simulator index mapping."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.rows = []
        for index in range(env.num_envs):
            local_cfg = cfg.copy()
            local_cfg.world_mesh_path = f'/World/envs/env_{index}/Stomach'+cfg.collision_mesh_suffix
            self.rows.append(ScopedRLRowAction(local_cfg,RowEnvironment(env,index)))

    @property
    def action_dim(self): return 9 if self.cfg.mode == 'single' else 36
    @property
    def raw_actions(self): return torch.cat([row.raw_actions for row in self.rows])
    @property
    def processed_actions(self): return torch.cat([row.processed_actions for row in self.rows])
    @property
    def executed_history(self): return torch.cat([row.executed_history for row in self.rows])

    def process_actions(self,actions):
        if tuple(actions.shape) != (len(self.rows),self.action_dim) or not torch.isfinite(actions).all():
            raise ValueError('finite [N,D] vector actions required')
        for index,row in enumerate(self.rows):
            row.process_actions(actions[index:index+1])

    def apply_actions(self):
        for row in self.rows: row.apply_actions()

    def reset(self,env_ids=None):
        for index in row_ids(env_ids,len(self.rows)): self.rows[index].reset()


@configclass
class NewStomachRLVectorActionCfg(NewStomachRLActionCfg):
    class_type: type = NewStomachRLVectorAction
    collision_mesh_suffix: str = ''
