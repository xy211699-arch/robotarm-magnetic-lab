"""One global physics batch; partial reset never invokes an extra step."""
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from robotarm_magnetic_lab.runtime.new_stomach_rl_library import FrozenPoseLibrary
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_lifecycle import VectorLifecycle
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_rows import row_ids,translate_pose
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_runtime import NewStomachRLVectorRuntime


class NewStomachRLVectorEnv(ManagerBasedRLEnv):
    def __init__(self,cfg,pose_manifest,mask_path,pose_split='train',pose_id=None,**kwargs):
        self.runtime = None
        self.libraries = [FrozenPoseLibrary(pose_manifest,pose_split,cfg.seed+i,pose_id)
            for i in range(cfg.scene.num_envs)]
        self.reset_pose_ids = [None]*cfg.scene.num_envs
        with torch.inference_mode(False):
            super().__init__(cfg,**kwargs)
        if self._physics_handles_decimation or cfg.decimation != 240 or abs(self.physics_dt-1/240)>1e-12:
            raise RuntimeError('240 explicit physics substeps and one-second boundary required')
        self.lifecycle = VectorLifecycle(self.num_envs,self.device)
        with torch.inference_mode(False):
            self.runtime = NewStomachRLVectorRuntime(self,mask_path)

    def reset_rows(self,ids):
        if self.lifecycle._in_batch: raise RuntimeError('reset inside physics batch forbidden')
        ids = row_ids(ids,self.num_envs)
        if not ids: return
        rows = torch.tensor(ids,dtype=torch.int64,device=self.device)
        # Base reset uses env_ids for scene, event, manager and sensor reset.
        # It does not step physics. Never call the old one-env HOLD helper.
        ManagerBasedRLEnv._reset_idx(self,rows)
        bridge = self.event_manager.get_term_cfg('magnetic_collision_bridge').func
        bridge.reset(rows)
        capsule = self.scene['capsule']
        poses = []
        for index in ids:
            name,pose = self.libraries[index].sample()
            self.reset_pose_ids[index] = name
            origin = self.action_manager.get_term('magnet').rows[index].env_origin
            poses.append(translate_pose(pose,[0,0,0],origin))
        capsule.write_root_pose_to_sim_index(root_pose=torch.tensor(np.stack(poses),dtype=torch.float32,device=self.device),env_ids=rows)
        capsule.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids),6),device=self.device),env_ids=rows)
        self.sim.forward(); self.scene.update(0.)
        self.action_manager.get_term('magnet').reset(rows)
        self.runtime.reset_rows(ids)
        self.lifecycle.reset_rows(ids)

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            self.seed(seed)
            for index,library in enumerate(self.libraries): library.rng = np.random.default_rng(seed+index)
        self.reset_rows(list(range(self.num_envs)))
        # Explicit full reset has only WARMUP rows: one normal batch, not an
        # unobserved step while some other episode remains ACTIVE.
        observation,_,_,_,extras = self.step(torch.zeros((self.num_envs,self.action_manager.total_action_dim),device=self.device))
        return observation,extras

    def step(self,action):
        issued = self.lifecycle.begin(action.to(self.device))
        valid = self.lifecycle._valid.clone()
        before = self._sim_step_counter
        self.action_manager.process_action(issued)
        self.recorder_manager.record_pre_step()
        for _ in range(240):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and self.sim.is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(self.physics_dt)
        reward = self.runtime.finish_batch(valid)
        boundary = self.lifecycle.finish(self._sim_step_counter-before)
        self.common_step_counter += 1
        self.episode_length_buf[:] = self.lifecycle.episode_seconds
        self.reset_buf = self.termination_manager.compute()
        terminated = self.termination_manager.terminated.clone()
        truncated = self.termination_manager.time_outs.clone() & valid
        if terminated.any():
            raise RuntimeError('vector safety termination: retain evidence, do not hide with autoreset')
        if not torch.isfinite(reward).all(): raise RuntimeError('nonfinite vector reward')
        observations = self.runtime.observations()
        if any(not torch.isfinite(v).all() for v in observations.values()): raise RuntimeError('nonfinite vector observation')
        final = {name:value.clone() for name,value in observations.items()}
        extras = dict(boundary,final_obs=final,final_mask=truncated.clone(),
            physical_substeps=self._sim_step_counter-before,issued_actions=issued.clone(),
            pose_ids=list(self.reset_pose_ids))
        reset_ids = truncated.nonzero(as_tuple=False).flatten()
        if reset_ids.numel():
            self.reset_rows(reset_ids)
            observations = self.runtime.observations()
        self.obs_buf = observations
        self.reward_buf = reward
        self.extras = extras
        return observations,reward,terminated,truncated,extras


def register_vector():
    import gymnasium as gym
    name = 'Template-Robotarm-Magnetic-New-Stomach-RL-Vector-v0'
    if name not in gym.registry:
        gym.register(id=name,entry_point=NewStomachRLVectorEnv,disable_env_checker=True)
    return name
