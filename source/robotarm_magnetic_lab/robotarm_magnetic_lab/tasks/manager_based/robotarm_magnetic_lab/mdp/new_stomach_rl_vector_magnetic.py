"""Private legacy bridge per row; the original physics_step is inherited.

This is isolation, not a new magnetic algorithm or a claimed speed-up.
"""
import math
import numpy as np
import torch
from isaaclab.managers import ManagerTermBase
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_rows import RowEnvironment, row_ids
from .legacy_bridge import LegacyMagneticCollisionBridge, XrdfSphereCollisionModel, LEGACY_EXTENSION_ROOT, _load_source_module, _empty_state


class ScopedMagneticRow(LegacyMagneticCollisionBridge):
    def __init__(self,cfg,env,modules):
        # Reproduce object wiring only. Force evaluation, clamp, drag, release,
        # float32 filter and wrench writes remain the original physics_step.
        ManagerTermBase.__init__(self,cfg,env)
        config_module,field_module,streamline_module,visual_module = modules
        self.config = config_module.load_config(LEGACY_EXTENSION_ROOT)
        self.model = field_module.FiniteMagnetSystem(self.config)
        self.robot,self.capsule = env.scene['robot'],env.scene['capsule']
        self.collision_model = XrdfSphereCollisionModel(self.robot,LEGACY_EXTENSION_ROOT/self.config['planning']['robot_xrdf'])
        self.arm_indices = [self.robot.data.joint_names.index(n) for n in self.config['robot']['arm_joint_names']]
        self.magnet_body_index = self.robot.data.body_names.index('magl')
        self.base_body_index = self.robot.data.body_names.index('base_link')
        self.elapsed = torch.zeros(1,device=env.device)
        self.frame_count = 0
        self._field_last_position = np.full(3,np.nan)
        self.state = _empty_state(1,env.device)
        self._filtered_wrench = torch.zeros((1,12),device=env.device)
        self._trace_streamlines = streamline_module.trace_streamlines
        self._create_or_update_streamlines = visual_module.create_or_update_streamlines
        self._local_streamlines = self._build_local_streamlines()

    def _update_field_visualization(self,env_id,position,rotation):
        if env_id != 0: raise ValueError('row magnetic bridge escaped binding')
        super()._update_field_visualization(self._env.row,position,rotation)

    def _log(self,message):
        # Full wrench evidence comes from the isolation tape, not stdout spam.
        if not message.startswith('MAGNETIC_COLLISION_STATUS'):
            print(f'VECTOR_MAGNETIC_ROW row={self._env.row} {message}',flush=True)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.frame_count = 0
        self._field_last_position[:] = np.nan


class NewStomachRLVectorMagnetic(ManagerTermBase):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        modules = tuple(_load_source_module('_new_vector_'+name,path) for name,path in (
            ('config','robotarm/magnetic_sim/config.py'),
            ('field','robotarm/magnetic_sim/magnetics/field_models.py'),
            ('stream','robotarm/magnetic_sim/magnetics/streamlines.py'),
            ('visual','robotarm/magnetic_sim/visualization/magnetic_field.py')))
        self.rows = [ScopedMagneticRow(cfg,RowEnvironment(env,index),modules) for index in range(env.num_envs)]
        self.config = self.rows[0].config

    def __call__(self,env,env_ids): pass

    def physics_step(self,env,env_ids=None):
        for index in row_ids(env_ids,len(self.rows)):
            row = self.rows[index]
            row.physics_step(row._env)

    def reset(self,env_ids=None):
        for index in row_ids(env_ids,len(self.rows)): self.rows[index].reset()
