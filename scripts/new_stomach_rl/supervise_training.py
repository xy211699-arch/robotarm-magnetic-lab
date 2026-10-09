"""R1 bounded single-worker supervision; no automatic restart or seed chaining."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/new_stomach_rl_capacity/training_readiness/jobs'


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def finite(value):
    if isinstance(value,float): return math.isfinite(value)
    if isinstance(value,dict): return all(finite(v) for v in value.values())
    if isinstance(value,list): return all(finite(v) for v in value)
    return True


def capacity_timeouts(summary_path,boundaries_path):
    """Conservative supervision limits from measured N=8, not physics tuning."""
    summary=read(summary_path)
    steps=[json.loads(line)['elapsed_s'] for line in Path(boundaries_path).read_text().splitlines()]
    startup=summary['startup_build_wall_seconds']
    if summary.get('num_envs')!=8 or not steps or not finite([startup,*steps]) or min(startup,*steps)<=0:
        raise ValueError('finite positive eight-env capacity measurements required')
    return dict(startup_timeout_seconds=math.ceil(3*startup),
        heartbeat_timeout_seconds=max(180,math.ceil(10*max(steps))),
        measured_slowest_step_seconds=max(steps),
        evidence_files=[dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
                        for p in (summary_path,boundaries_path)])


def stop_owned(child,grace):
    if child.poll() is not None: return
    try: os.killpg(child.pid,signal.SIGINT)
    except ProcessLookupError: return
    try: child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        child.wait(timeout=2)


def monitor(command,folder,config,poll=.5):
    """Only the Popen-owned process group can be stopped, never an arbitrary PID."""
    folder=Path(folder);started=time.monotonic();last_progress=started
    state=dict(status='running',run_id=folder.name,started_epoch_s=time.time(),
        heartbeat_epoch_s=time.time(),child_pid=None,update_count=0,effective_samples=0,
        stage='startup',latest_checkpoint=None,command=command)
    child=None;last_stamp=None
    try:
        with (folder/'worker.log').open('x') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,cwd=ROOT)
            state['child_pid']=child.pid;write(folder/'status.json',state)
            while child.poll() is None:
                if (folder/'stop.request').exists():
                    state.update(status='interrupted',stop_reason='user requested stop');break
                if time.monotonic()-started>=config['max_wall_seconds']-config['stop_grace_seconds']-2:
                    raise RuntimeError('wall-clock budget exhausted')
                heartbeat=folder/'heartbeat.json'
                if heartbeat.exists():
                    h=read(heartbeat)
                    if not finite(h) or h.get('run_id')!=folder.name: raise RuntimeError('invalid/nonfinite heartbeat')
                    stamp=h['heartbeat_epoch_s']
                    if not isinstance(stamp,(int,float)) or stamp>time.time()+5: raise RuntimeError('invalid heartbeat timestamp')
                    if last_stamp is not None and stamp<last_stamp: raise RuntimeError('heartbeat regressed')
                    if last_stamp!=stamp:
                        last_stamp=stamp;last_progress=time.monotonic()
                        for name in ('update_count','effective_samples'):
                            value=h[name]
                            if isinstance(value,bool) or int(value)!=value or value<state[name]: raise RuntimeError('progress count regressed')
                        if h['update_count']>config['max_updates'] or h['effective_samples']>config['max_updates']*config['rollout_steps']*8:
                            raise RuntimeError('progress exceeds approved budget')
                        if h.get('stage') not in ('startup','sampling','update','checkpoint','completed'):
                            raise RuntimeError('invalid worker stage')
                        state.update({name:h.get(name) for name in ('stage','update_count','effective_samples','latest_checkpoint','metrics','gpu_memory_bytes')})
                        state['worker_heartbeat_epoch_s']=stamp
                        checkpoint=h.get('latest_checkpoint')
                        if checkpoint:
                            p=Path(checkpoint['path']).resolve()
                            if not p.is_relative_to(folder.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=checkpoint['sha256']:
                                raise RuntimeError('checkpoint path/hash mismatch')
                        if h.get('status')=='failed': raise RuntimeError(h.get('error','worker failed'))
                limit=config['startup_timeout_seconds'] if state['stage']=='startup' else config['heartbeat_timeout_seconds']
                if last_stamp is not None and time.time()-last_stamp>limit: raise RuntimeError('worker heartbeat timestamp stale')
                if time.monotonic()-last_progress>limit: raise RuntimeError('worker progress heartbeat stale')
                state.update(heartbeat_epoch_s=time.time(),elapsed_seconds=time.monotonic()-started)
                write(folder/'status.json',state);time.sleep(poll)
            if state['status']=='interrupted': stop_owned(child,config['stop_grace_seconds'])
            elif (folder/'stop.request').exists(): state.update(status='interrupted',stop_reason='user requested stop')
            else:
                if child.returncode!=0: raise RuntimeError(f'worker exit code {child.returncode}; see worker.log')
                summary=read(folder/'summary.json')
                if (not finite(summary) or summary.get('status')!='completed' or summary.get('run_id')!=folder.name or
                    summary.get('update_count')!=config['max_updates'] or summary.get('global_boundaries')!=config['max_updates']*config['rollout_steps'] or
                    not 0<summary.get('effective_samples',0)<=config['max_updates']*config['rollout_steps']*8):
                    raise RuntimeError('missing, failed or incomplete worker result')
                state.update(status='completed',update_count=summary['update_count'],effective_samples=summary['effective_samples'])
    except Exception as error:
        state.update(status='paused_on_error',error=f'{type(error).__name__}: {error}')
        if child is not None: stop_owned(child,config['stop_grace_seconds'])
    finally:
        state.update(child_pid=None,exit_code=child.returncode if child is not None else None,
            ended_epoch_s=time.time(),elapsed_seconds=time.monotonic()-started)
        write(folder/'status.json',state)
    return state


def validate_config(path):
    config=read(path)
    if (config.get('purpose')!='r2_smoke' or config.get('num_envs')!=8 or
        config.get('max_updates')!=3 or config.get('rollout_steps')!=64 or config.get('group') not in ('A','B','C','D')):
        raise ValueError('only fixed eight-env R2 smoke is accepted; formal training not approved')
    for name in ('max_wall_seconds','startup_timeout_seconds','heartbeat_timeout_seconds','stop_grace_seconds'):
        value=config.get(name)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'positive finite {name} required')
    if config['stop_grace_seconds']>30: raise ValueError('stop grace must be <=30 seconds')
    if config['max_wall_seconds']<=config['stop_grace_seconds']+2: raise ValueError('wall budget must include shutdown grace')
    evidence=config.get('evidence_files',[])
    if not evidence: raise ValueError('approved gate/config identity evidence required')
    for item in evidence:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('gate or frozen input bytes changed')
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['start','status','stop','worker'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--run_dir',type=Path)
    parser.add_argument('--kit_args')
    args=parser.parse_args()
    if args.command in ('status','stop','worker') and args.run_dir is None: parser.error('--run_dir required')
    if args.command=='status': print(json.dumps(read(args.run_dir/'status.json'),indent=2));return
    if args.command=='stop':
        state=read(args.run_dir/'status.json')
        if state['status'] in ('queued','running') and not (args.run_dir/'stop.request').exists():
            with (args.run_dir/'stop.request').open('x') as stream: stream.write('user requested stop\n')
        print(json.dumps({'status':'stop_requested','run_dir':str(args.run_dir)}));return
    if args.command=='worker':
        config=read(args.run_dir/'launch.json')['config']
        try:
            monitor([sys.executable,str(ROOT/'scripts/new_stomach_rl/train.py'),'--config',
                str(args.run_dir/'frozen_config.json'),'--run_dir',str(args.run_dir)],args.run_dir,config)
        finally:
            lock=BASE/'active.lock'
            if lock.exists() and read(lock).get('run_dir')==str(args.run_dir.resolve()): lock.unlink()
        return
    if args.config is None: parser.error('--config required')
    config=validate_config(args.config)
    target=ROOT/'scripts/new_stomach_rl/train.py'
    if not target.is_file(): parser.error('R2 train.py not delivered; no GPU job launched')
    for previous in ('capacity_jobs','acceptance_jobs'):
        p=ROOT/f'artifacts/new_stomach_rl_capacity/{previous}/latest/status.json'
        if p.is_file() and read(p)['status'] in ('running','queued'): parser.error('GPU capacity/isolation job still running')
    BASE.mkdir(parents=True,exist_ok=True)
    # Atomic single-launch lock, no concurrent training on this project GPU.
    lock=BASE/'active.lock'
    with lock.open('x') as stream: stream.write('reserved\n')
    try:
        folder=BASE/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');folder.mkdir()
        write(folder/'frozen_config.json',config)
        write(folder/'launch.json',dict(config=config,source_config=str(args.config.resolve()),
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        write(folder/'status.json',dict(status='queued',run_id=folder.name))
        write(lock,dict(run_dir=str(folder.resolve())))
        with (folder/'supervisor.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--run_dir',str(folder)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(status='started',supervisor_pid=child.pid,run_dir=str(folder))))
    except BaseException:
        lock.unlink();raise
    # Worker releases only a lock that belongs to its own run directory.


if __name__=='__main__': main()
