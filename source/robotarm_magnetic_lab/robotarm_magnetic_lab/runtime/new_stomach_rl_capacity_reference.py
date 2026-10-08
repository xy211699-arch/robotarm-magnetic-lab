"""P0 read-only original bridge tape; no force/model/action changes."""
import numpy as np
import torch


def canonical_actions(mode):
    """Frozen 20 raw actions: HOLD, +/-9 axes, mixed; no parameter search."""
    actions = np.zeros((20,9), dtype=np.float64)
    for axis in range(9):
        actions[1+2*axis,axis]=.2
        actions[2+2*axis,axis]=-.2
    actions[-1]=np.asarray([1,-1,1,-1,1,-1,1,-1,1])*.1
    if mode=='single':
        return actions
    if mode=='chunk':
        # Actual four 0.25s segments; physical scaling remains in original term.
        return (actions[:,None,:]*np.asarray([1.,.5,-.5,-1.])[None,:,None]).reshape(20,36)
    raise ValueError('single or chunk required')


def tensor(value):
    return getattr(value,'torch',value)


class MagneticInputTape:
    """Inputs BEFORE original model call, outputs AFTER that same call.

    Never infer raw magnetic input from a post-integration capsule pose.
    Single-reference-only; does not masquerade as vector magnetic adapter.
    """
    def __init__(self,bridge,device):
        self.bridge=bridge
        self.original=bridge.physics_step
        self.inputs=torch.empty((240,33),device=device)
        self.outputs=torch.empty((240,25),device=device)
        self.active=False
        self.count=0

    def begin(self):
        self.active=True
        self.count=0

    def physics_step(self,env,env_ids=None):
        if env.num_envs!=1:
            raise ValueError('reference tape requires original single environment')
        if self.active:
            if self.count>=240:
                raise RuntimeError('more than 240 magnetic calls in a policy second')
            b=self.bridge
            self.inputs[self.count].copy_(torch.cat((
                tensor(b.robot.data.body_pos_w)[0,b.magnet_body_index],
                tensor(b.robot.data.body_quat_w)[0,b.magnet_body_index],
                tensor(b.capsule.data.root_pos_w)[0],tensor(b.capsule.data.root_quat_w)[0],
                tensor(b.capsule.data.root_lin_vel_w)[0],tensor(b.capsule.data.root_ang_vel_w)[0],
                b._filtered_wrench[0],b.elapsed[:1])))
        self.original(env,env_ids)
        if self.active:
            b=self.bridge
            self.outputs[self.count].copy_(torch.cat((b.state['raw_wrench'][0],
                b.state['wrench'][0],b.elapsed[:1])))
            self.count+=1

    def finish(self):
        self.active=False
        if self.count!=240:
            raise RuntimeError('exactly 240 original magnetic calls required')
        if not torch.isfinite(self.inputs).all() or not torch.isfinite(self.outputs).all():
            raise RuntimeError('nonfinite reference evidence')
        return self.inputs.clone(),self.outputs.clone()
