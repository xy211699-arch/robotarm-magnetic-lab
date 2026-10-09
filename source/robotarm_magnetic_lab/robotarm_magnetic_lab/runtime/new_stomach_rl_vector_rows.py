"""Explicit row bindings for reusing validated one-row algorithms.

No global monkey-patches and no modification of legacy single-row guards.
Every tensor read and every simulator write is bound to one actual scene row.
"""
from types import SimpleNamespace
import numpy as np
import torch


def row_ids(ids, count):
    if ids is None or isinstance(ids, slice):
        if ids is not None and ids != slice(None):
            raise ValueError('only the full slice is supported')
        return list(range(count))
    values = torch.as_tensor(ids).detach().cpu().reshape(-1).tolist()
    if any(isinstance(i, bool) or not np.isfinite(i) or int(i) != i or i < 0 or i >= count for i in values):
        raise ValueError('invalid environment row')
    result = [int(i) for i in values]
    if len(set(result)) != len(result):
        raise ValueError('duplicate environment row')
    return result


class RowData:
    def __init__(self, data, row, count):
        self._data, self.row, self.count = data, int(row), int(count)

    def __getattr__(self, name):
        value = getattr(self._data, name)
        tensor = getattr(value, 'torch', value)
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 0 or tensor.shape[0] != self.count:
                raise ValueError(f'{name}: expected an explicit scene-row tensor')
            selected = tensor[self.row:self.row+1]
            return selected if isinstance(value, torch.Tensor) else SimpleNamespace(torch=selected)
        if isinstance(value, dict):
            return {key: (getattr(item, 'torch', item)[self.row:self.row+1]
                if isinstance(getattr(item, 'torch', item), torch.Tensor) else item)
                for key, item in value.items()}
        return value


class RowWriter:
    def __init__(self, owner, row, device):
        self._owner, self.row = owner, int(row)
        self.ids = torch.tensor([row], dtype=torch.int64, device=device)

    def _write(self, method, **kwargs):
        local = kwargs.pop('env_ids', None)
        if row_ids(local, 1) != [0]:
            raise ValueError('row facade requires its sole local row')
        return getattr(self._owner, method)(env_ids=self.ids, **kwargs)

    def reset(self, env_ids=None):
        return self._write('reset', env_ids=env_ids)

    def set_forces_and_torques_index(self, **kwargs):
        return self._write('set_forces_and_torques_index', **kwargs)


class RowAsset(RowWriter):
    def __init__(self, asset, row, count, device):
        super().__init__(asset, row, device)
        self.data = RowData(asset.data, row, count)
        self.permanent_wrench_composer = RowWriter(asset.permanent_wrench_composer, row, device)

    def __getattr__(self, name):
        if name.startswith(('write_', 'set_')) or name == 'reset':
            raise AttributeError(f'unscoped simulator write is forbidden: {name}')
        return getattr(self._owner, name)

    def set_joint_position_target_index(self, **kwargs):
        return self._write('set_joint_position_target_index', **kwargs)

    def set_joint_velocity_target_index(self, **kwargs):
        return self._write('set_joint_velocity_target_index', **kwargs)

    def set_joint_effort_target_index(self, **kwargs):
        return self._write('set_joint_effort_target_index', **kwargs)


class RowCamera:
    def __init__(self, camera, row, count, device):
        self._camera, self.row = camera, row
        self.cfg, self.num_instances = camera.cfg, 1
        self.data = RowData(camera.data, row, count)
        import warp as wp
        mask = np.zeros(count,dtype=bool); mask[row] = True
        self._ALL_ENV_MASK = wp.array(mask,dtype=wp.bool,device=device)
        self._view = SimpleNamespace(get_world_poses=self._poses,
            _fabric_hierarchy=getattr(camera._view, '_fabric_hierarchy', None))

    @property
    def frame(self):
        tensor = getattr(self._camera.frame, 'torch', self._camera.frame)
        return tensor[self.row:self.row+1]

    def _poses(self):
        positions, quats = self._camera._view.get_world_poses()
        return (SimpleNamespace(torch=positions.torch[self.row:self.row+1]),
                SimpleNamespace(torch=quats.torch[self.row:self.row+1]))

    def _update_buffers_impl(self, ids):
        if ids is not self._ALL_ENV_MASK:
            raise ValueError('camera capture escaped its row')
        return self._camera._update_buffers_impl(ids)


class RowEnvironment:
    def __init__(self, env, row):
        row_ids([row], env.num_envs)
        self.parent, self.row, self.num_envs = env, row, 1
        self.device, self.cfg = env.device, env.cfg
        self.step_dt, self.physics_dt, self.sim = env.step_dt, env.physics_dt, env.sim
        self.event_manager = env.event_manager
        self.scene = {name: RowAsset(env.scene[name], row, env.num_envs, env.device)
            for name in ('robot', 'capsule')}

    def bind_camera(self):
        self.scene['capsule_camera'] = RowCamera(self.parent.scene['capsule_camera'],
            self.row, self.parent.num_envs, self.device)


def translate_pose(pose, original_origin, environment_origin):
    """Canonical world -> local -> clone world; quaternion is not reordered."""
    value = np.asarray(pose, dtype=np.float64).copy()
    source = np.asarray(original_origin, dtype=np.float64)
    target = np.asarray(environment_origin, dtype=np.float64)
    if value.shape != (7,) or source.shape != (3,) or target.shape != (3,):
        raise ValueError('pose/origin shape mismatch')
    if not all(np.isfinite(x).all() for x in (value, source, target)):
        raise ValueError('nonfinite pose/origin')
    if abs(np.linalg.norm(value[3:])-1) > 1e-5:
        raise ValueError('normalized XYZW pose required')
    value[:3] = value[:3] - source + target
    return value
