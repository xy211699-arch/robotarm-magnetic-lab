"""Formal single-env step/reset adapter; no changes to archived core.

Native step generates ten live geometric rewards before returning one fresh
RGB observation. Reset-only pose writes and unbudgeted HOLD are isolated from
Actor steps. Same-Step terminal observations/rewards are saved before warmup.
"""
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from robotarm_magnetic_lab.runtime.new_stomach_rl_library import FrozenPoseLibrary
from robotarm_magnetic_lab.runtime.new_stomach_rl_runtime import NewStomachRLRuntime


def clone_tree(value):
    if isinstance(value,torch.Tensor):
        return value.clone()
    if isinstance(value,dict):
        return {k:clone_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(clone_tree(v) for v in value)
    return value


class NewStomachRLPreflightEnv(ManagerBasedRLEnv):
    def __init__(self,cfg,pose_manifest,mask_path,pose_split='train',pose_id=None,record_physics=False,**kwargs):
        self._pending_bootstrap = False
        self._bootstrap = False
        self._new_stomach_rl_runtime = None
        self.library = FrozenPoseLibrary(pose_manifest,pose_split,cfg.seed,pose_id)
        self.last_episode_records = None
        self._recording_step = False
        self.trace = None
        super().__init__(cfg,**kwargs)
        self.runtime = NewStomachRLRuntime(self,mask_path)
        if cfg.group in ('B','D'):
            from robotarm_magnetic_lab.runtime.task010_visual_encoder import FrozenResNet18Encoder
            self.runtime.encoder = FrozenResNet18Encoder().to(self.device)
        if record_physics:
            from robotarm_magnetic_lab.runtime.new_stomach_rl_trace import PhysicalStepRecorder
            self.trace = PhysicalStepRecorder(self.scene['capsule'],self.action_manager.get_term('magnet'),
                self.event_manager.get_term_cfg('magnetic_collision_bridge').func,self.device)
            self._original_scene_update = self.scene.update
            def update_and_record(dt):
                self._original_scene_update(dt)
                if self._recording_step and abs(dt-self.physics_dt)<1e-12:
                    self.trace.append()
            self.scene.update = update_and_record

    def _reset_idx(self,env_ids):
        runtime = self._new_stomach_rl_runtime
        if runtime is not None:
            if runtime.ready:
                self._terminal_pose_id = self.reset_pose_id
            runtime.ready = False
        super()._reset_idx(env_ids)
        self.event_manager.get_term_cfg('magnetic_collision_bridge').func.reset()
        self.reset_pose_id, pose = self.library.sample()
        capsule = self.scene['capsule']
        capsule.permanent_wrench_composer.reset()
        capsule.write_root_pose_to_sim_index(root_pose=torch.tensor([pose],dtype=torch.float32,device=self.device))
        capsule.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6),device=self.device))
        self.sim.forward(); self.scene.update(0.)
        self.action_manager.get_term('magnet').reset()
        self._pending_bootstrap = True

    def _settle_and_initialize(self):
        self._bootstrap = True
        self.runtime.ready = False
        self._pending_bootstrap = False
        policy_counter = self.common_step_counter
        try:
            _,reward,terminated,truncated,_ = super().step(torch.zeros((1,self.action_manager.total_action_dim),device=self.device))
            if terminated.any() or truncated.any() or reward.abs().max() != 0:
                raise RuntimeError('unbudgeted reset HOLD failed or generated policy reward')
            self.episode_length_buf.zero_()
            self.common_step_counter = policy_counter
            self.action_manager.get_term('magnet').reset()
            self.runtime.initialize()
            if self.action_manager.get_term('magnet').executed_history.count_nonzero():
                raise RuntimeError('reset inherited issued action history')
            self.obs_buf = self.observation_manager.compute(update_history=True)
            return self.obs_buf
        finally:
            self._bootstrap = False

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            self.library.rng = np.random.default_rng(seed)
        self.runtime.ready = False
        super().reset(seed=seed,options=options)
        return self._settle_and_initialize(),self.extras

    def step(self,action):
        if self._bootstrap or not self.runtime.ready:
            raise RuntimeError('Actor step requires initialized C0 and completed reset HOLD')
        if self.trace is not None:
            self.trace.begin()
        self._recording_step = True
        try:
            observations,reward,terminated,truncated,extras = super().step(action)
        finally:
            self._recording_step = False
        self.last_policy_physics_end_tick = self._sim_step_counter
        if self.trace is not None:
            self.last_physics_tensor = self.trace.finish()
        term = self.action_manager.get_term('magnet')
        self.last_policy_telemetry = clone_tree(term.telemetry)
        self.last_policy_history = (extras['final_obs']['policy'][:,512:].reshape(1,4,9).clone()
            if self._pending_bootstrap else term.executed_history)
        if self._pending_bootstrap:
            saved = clone_tree((reward,terminated,truncated,extras))
            self.last_episode_records = dict(ten_hz=list(self.runtime.ten_hz_records),
                one_hz=list(self.runtime.one_hz_records),pose_id=self._terminal_pose_id)
            observations = self._settle_and_initialize()
            reward,terminated,truncated,extras = saved
        return observations,reward,terminated,truncated,extras

    def close(self):
        if self.trace is not None:
            self.scene.update = self._original_scene_update
        super().close()


def register_preflight():
    import gymnasium as gym
    name = 'Template-Robotarm-Magnetic-New-Stomach-RL-Preflight-v0'
    if name not in gym.registry:
        gym.register(id=name,entry_point=NewStomachRLPreflightEnv,disable_env_checker=True)
    return name
