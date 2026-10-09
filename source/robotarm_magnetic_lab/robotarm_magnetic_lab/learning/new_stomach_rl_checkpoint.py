"""Training-state continuation, explicitly NOT PhysX trajectory restoration."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_state(value):
    if isinstance(value,torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value,float): return bool(np.isfinite(value))
    if isinstance(value,dict): return all(finite_state(v) for v in value.values())
    if isinstance(value,(tuple,list)): return all(finite_state(v) for v in value)
    return True


def configuration_sha(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_identity(identity,config):
    for key in ('config_sha256','code_sha256','assets_sha256','weights_sha256'):
        value=identity.get(key,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'missing SHA-256 identity: {key}')
    if identity['config_sha256']!=configuration_sha(config): raise ValueError('config identity mismatch')


def sampler_identity(library):
    records=json.dumps(library.records,sort_keys=True,separators=(',',':'),allow_nan=False)
    return dict(records_sha256=hashlib.sha256(records.encode()).hexdigest(),
        fixed_pose_id=library.fixed['pose_id'] if library.fixed is not None else None)


def save_checkpoint(path,runner,libraries,config,identity,effective_samples,*,at_episode_boundary,
                    statistics=None,allow_weights_only=False):
    check_identity(identity,config)
    if not at_episode_boundary and not allow_weights_only:
        raise ValueError('mid-episode save requires explicit weights-only mode')
    if isinstance(effective_samples,bool) or effective_samples<0 or int(effective_samples)!=effective_samples: raise ValueError('invalid sample count')
    path=Path(path)
    if path.exists(): raise FileExistsError('never overwrite a checkpoint')
    state=np.random.get_state()
    payload=dict(schema=1,kind='training_boundary' if at_episode_boundary else 'weights_only',
        trajectory_continuous=False,requires_new_reset=True,dimension=runner.dimension,
        gamma=runner.gamma,gae_lambda=runner.gae_lambda,config=config,identity=identity,
        actor=runner.actor.state_dict(),critic=runner.critic.state_dict(),
        optimizer=runner.optimizer.state_dict() if at_episode_boundary else None,
        update_count=runner.update_count,effective_samples=int(effective_samples),
        python_rng=random.getstate(),numpy_rng=(state[0],state[1].tolist(),*state[2:]),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        samplers=[dict(identity=sampler_identity(lib),rng=lib.rng.bit_generator.state) for lib in libraries],
        statistics={} if statistics is None else statistics)
    if not finite_state(payload): raise ValueError('nonfinite checkpoint state')
    # Exclusive hard-link publication prevents replacing an existing checkpoint.
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temporary.open('xb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest(path),kind=payload['kind'])


def restore_checkpoint(path,sha256,runner,libraries,config,identity,reset_environment,clear_hidden,
                       *,allow_weights_only=False):
    check_identity(identity,config)
    if digest(path)!=sha256: raise ValueError('checkpoint bytes changed')
    data=torch.load(path,map_location='cpu',weights_only=True)
    if not finite_state(data): raise ValueError('nonfinite checkpoint state')
    if (data['schema']!=1 or data['identity']!=identity or data['config']!=config or
        data['dimension']!=runner.dimension or data['gamma']!=runner.gamma or data['gae_lambda']!=runner.gae_lambda):
        raise ValueError('checkpoint identity/model/timebase mismatch')
    if data['kind'] not in ('training_boundary','weights_only'): raise ValueError('unknown restoration kind')
    weights_only=data['kind']=='weights_only'
    if weights_only and not allow_weights_only: raise ValueError('weights-only checkpoint is not strict continuation')
    if not callable(reset_environment) or not callable(clear_hidden): raise ValueError('fresh reset and GRU clear required')
    if len(data['samplers'])!=len(libraries) or any(s['identity']!=sampler_identity(lib) for s,lib in zip(data['samplers'],libraries)):
        raise ValueError('pose sampler identity changed')
    if data['cuda_rng'] is not None and (not torch.cuda.is_available() or len(data['cuda_rng'])!=torch.cuda.device_count()):
        raise ValueError('CUDA RNG device count mismatch')
    runner.actor.load_state_dict(data['actor']);runner.critic.load_state_dict(data['critic'])
    if not weights_only:
        runner.optimizer.load_state_dict(data['optimizer']);runner.update_count=data['update_count']
        random.setstate(data['python_rng'])
        state=data['numpy_rng'];np.random.set_state((state[0],np.array(state[1],dtype=np.uint32),*state[2:]))
        torch.set_rng_state(data['torch_rng'])
        if data['cuda_rng'] is not None: torch.cuda.set_rng_state_all(data['cuda_rng'])
        for lib,sampler in zip(libraries,data['samplers']): lib.rng.bit_generator.state=sampler['rng']
    else:
        # Parameters only: discard Adam moments and iteration counters explicitly.
        runner.optimizer.state.clear();runner.update_count=0
    reset_environment();clear_hidden()
    return dict(kind=data['kind'],update_count=runner.update_count,
        effective_samples=0 if weights_only else data['effective_samples'],
        statistics={} if weights_only else data['statistics'],trajectory_continuous=False,
        reset_performed=True,hidden_cleared=True)
