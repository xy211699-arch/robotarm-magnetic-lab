"""Independent old coverage/reward/RGB state on a shared physical clock."""
from types import SimpleNamespace
import numpy as np
import torch
from .new_stomach_rl_runtime import NewStomachRLRuntime
from .new_stomach_rl_reward import NewStomachReward
from .new_stomach_rl_optics import LiveMountedOptics
from .new_stomach_rl_rgb import PolicyRGBBoundary
from .new_stomach_rl_vector_lifecycle import prime_zero_compatible_reward
from robotarm_magnetic_lab.coverage.new_stomach_rl_coverage import frozen_visibility, DualRateCoverage


class ZeroCompatibleReward(NewStomachReward):
    def reset_reward(self,env_ids,c0):
        value = torch.as_tensor(c0,device=self.device).reshape(-1)
        if list(env_ids) == [0] and value.numel() == 1 and bool(value[0] == 0):
            prime_zero_compatible_reward(self,value)
        else:
            super().reset_reward(env_ids,c0)


def canonical_reference(stage,row):
    """Keep frozen topology/hash in canonical coordinates, never relabel SHA."""
    from pxr import UsdGeom
    from robotarm_magnetic_lab.geometry.new_stomach_runtime import NEW_STOMACH_WELD_TOLERANCE_M
    from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput,preprocess_reference_mesh
    root = stage.GetPrimAtPath(f'/World/envs/env_{row._env.row}')
    path = row._new_stomach_runtime.collision_mesh_path
    mesh_prim = stage.GetPrimAtPath(path)
    matrix,_ = UsdGeom.XformCache(0.).ComputeRelativeTransform(mesh_prim,root)
    canonical_path = '/World/envs/env_0'+path[len(str(root.GetPath())):]
    mesh = UsdGeom.Mesh(mesh_prim)
    item = MeshInput(canonical_path,np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64),
        np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=np.int64),
        np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int64),np.asarray(matrix,dtype=np.float64),
        str(mesh.GetOrientationAttr().Get() or 'rightHanded'))
    return preprocess_reference_mesh([item],[canonical_path],NEW_STOMACH_WELD_TOLERANCE_M)


class CloneVisibility:
    """Query the canonical exact BVH using only this clone's camera pose."""
    def __init__(self,visibility,origin):
        self.original = visibility
        self.device,self.weights,self.reference = visibility.device,visibility.weights,visibility.reference
        self.origin = torch.tensor(origin,dtype=torch.float64,device=self.device)[None]

    def __call__(self,optical):
        positions,axes = optical
        return self.original((positions-self.origin,axes))


class SharedFrameRow:
    """One actual full sensor capture; row clocks only associate that frame."""
    def __init__(self,owner,index):
        self.owner,self.index,self.last_second = owner,index,-1
        self.sync = owner.rgb_clock.sync

    def sample(self,second,tick,camera):
        if second != self.last_second+1 or tick != second*240:
            raise ValueError('row RGB policy clock must advance once')
        self.last_second = second
        return self.owner.rgb[self.index:self.index+1].clone(),self.owner.frames[self.index:self.index+1].clone()


class VectorRowRuntime(NewStomachRLRuntime):
    """Reuse old boundary/observation methods with private one-row buffers."""
    def __init__(self,row,mask_path,owner):
        env = row._env
        env.bind_camera()
        self.env,self.term,self.ready = env,row,False
        self.owner = owner
        env.action_manager = SimpleNamespace(get_term=lambda name: row)
        self.camera,self.capsule = env.scene['capsule_camera'],env.scene['capsule']
        import omni.usd
        row._geometry()
        reference = canonical_reference(omni.usd.get_context().get_stage(),row)
        self.visibility = CloneVisibility(frozen_visibility(reference,mask_path,env.device),row.env_origin)
        self.optics = LiveMountedOptics(self.camera,env.device,env.sim.physics_manager.forward)
        self.coverage = DualRateCoverage(self.visibility,1,geometry_optics=self.optics)
        self.reward = ZeroCompatibleReward(1,env.device)
        self.policy_second = 0
        self.visual_features = torch.zeros((1,512),device=env.device)
        self.encoder = None
        self.ten_hz_records,self.one_hz_records = [],[]
        row.geometry_tick_callback = self.on_physics_boundary
        env._new_stomach_rl_runtime = self

    def _encode_visual(self):
        # Mutable encoder frame caches must remain normal tensors, including
        # when an inference caller steps and later resets outside that scope.
        with torch.inference_mode(False),torch.no_grad():
            super()._encode_visual()

    def initialize(self):
        self.coverage.reset([0]); self.optics.records.clear()
        self.policy_second = 0
        self.rgb_clock = SharedFrameRow(self.owner,self.env.row)
        self.rgb,self.rgb_frames = self.rgb_clock.sample(0,0,self.camera)
        self.coverage.initialize(self.optics(self.capsule_pose()))
        if not torch.equal(self.visibility(self.camera_optics()),self.coverage._boundary_visible):
            raise RuntimeError('reset C0 optical/RGB mismatch')
        self.reward.reset_reward([0],self.coverage.c10_records[0]['coverage'])
        self.ten_hz_records = [dict(self.coverage.c10_records[0],**self.optics.records[0],initialization=True)]
        self.one_hz_records = [self.rgb_record(0,self.coverage.c1_records[0]['coverage'])]
        self.visual_features.zero_()
        if self.encoder is not None: self.encoder.reset()
        self._encode_visual()
        self.ready = True


class NewStomachRLVectorRuntime:
    def __init__(self,env,mask_path):
        self.env = env
        self.rgb_clock = PolicyRGBBoundary(env.num_envs,env.device)
        self.global_second = -1
        self.rows = [VectorRowRuntime(row,mask_path,self) for row in env.action_manager.get_term('magnet').rows]
        if env.cfg.group in ('B','D'):
            from .task010_visual_encoder import FrozenResNet18Encoder
            # Immutable backbone is shared; frame/feature caches remain private.
            first = FrozenResNet18Encoder().to(env.device)
            self.rows[0].encoder = first
            for row in self.rows[1:]:
                row.encoder = FrozenResNet18Encoder(backbone=first.backbone).to(env.device)

    def reset_rows(self,ids):
        self.rgb_clock.sync.reset_rows(torch.tensor(ids,dtype=torch.int64,device=self.env.device))
        for index in ids:
            row = self.rows[index]
            row.ready = False
            row.coverage.reset([0])
            row.ten_hz_records,row.one_hz_records = [],[]
            row.reward.trackers[0].reset()
            row.reward._pending.zero_(); row.reward._counts.zero_()
            row.reward._initialized.zero_()
            row.visual_features.zero_()
            if row.encoder is not None: row.encoder.reset()

    def finish_batch(self,valid):
        # Account for tick-zero baseline only inside the first completed HOLD.
        # Thereafter exactly one full-camera capture at each global boundary.
        self.global_second += 1
        self.rgb,self.frames = self.rgb_clock.sample(self.global_second,self.global_second*240,self.env.scene['capsule_camera'])
        reward = torch.zeros(self.env.num_envs,device=self.env.device)
        for index,row in enumerate(self.rows):
            if bool(valid[index]):
                reward[index] = row.finish_second()[0]
            else:
                row.term.reset()
                row.initialize()
        return reward

    def observations(self):
        policy,critic = [],[]
        for row in self.rows:
            if row.ready:
                policy.append(row.actor_observation());critic.append(row.critic_observation())
            else:
                policy.append(torch.zeros((1,548),device=self.env.device))
                critic.append(torch.zeros((1,28),device=self.env.device))
        return {'policy':torch.cat(policy),'critic':torch.cat(critic)}
