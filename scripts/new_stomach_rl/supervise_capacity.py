"""Sequential provisional capacity ladder; start/status never initialize Kit."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from supervise_isolation_acceptance import write

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/new_stomach_rl_capacity/capacity_jobs'
LEVELS=(1,4,8,12,16,20)
STEPS=((4,'D',True),)+tuple((n,g,False) for n in LEVELS for g in 'ABCD')


def evidence(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def gate(single_path,chunk_path):
    single=json.loads(Path(single_path).read_text());chunk=json.loads(Path(chunk_path).read_text())
    if (single.get('status')!='pass' or single.get('mode')!='single' or
        single.get('devices',{}).get('gpu_dynamics') is not True or
        len(single.get('comparisons',[]))!=19 or not all(x['passed'] for x in single['comparisons']) or
        len(single.get('reset_invariants',[]))!=7 or
        not all(all(x['checks'].values()) for x in single['reset_invariants'])):
        raise ValueError('Single full isolation evidence required')
    if (chunk.get('calibration_only') is not True or chunk.get('mode')!='chunk' or
        chunk.get('error',{}).get('type')!='ValueError' or
        not chunk['error']['message'].startswith('calibration exceeds predefined ceiling: capsule_orientation_rad=') or
        chunk.get('devices',{}).get('gpu_dynamics') is not True or
        len(chunk.get('runs',[]))!=3 or any(x['global_batches']!=20 for x in chunk['runs'])):
        raise ValueError('only the explicitly approved Chunk orientation comparison may be deferred')
    tapes=[a for a in chunk['artifacts'] if Path(a['path']).name=='isolation_tape.npz']
    if len(tapes)!=3 or len({a['sha256'] for a in tapes})!=1:
        raise ValueError('Chunk repeatability must have three identical physical tapes')
    for item in tapes:
        if evidence(item['path'])!=item: raise ValueError('Chunk calibration evidence changed')
    return dict(single=evidence(single_path),chunk=evidence(chunk_path),
        waiver='User 2026-10-09 approved deferring noncritical Chunk/P0 orientation mismatch; not a pass',
        formal_training_allowed=False)


def run_job(command,log,state,folder):
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid;write(folder/'status.json',state)
        while child.poll() is None:
            state['heartbeat_epoch_s']=time.time();write(folder/'status.json',state);time.sleep(5)
        state['child_pid']=None
    events=[json.loads(line.split('CAPACITY_RESULT ',1)[1]) for line in log.read_text(errors='replace').splitlines()
        if line.startswith('CAPACITY_RESULT ')]
    if child.returncode!=0 or len(events)!=1: raise RuntimeError(f'capacity child failed: exit={child.returncode}, log={log}')
    summary=Path(events[0]['output'])/'summary.json'
    data=json.loads(summary.read_text())
    if data['status']!=events[0]['status']: raise RuntimeError('summary/event mismatch')
    return data,summary


def worker(folder,job=run_job):
    config=json.loads((folder/'launch.json').read_text())
    state=dict(status='running',started_epoch_s=time.time(),heartbeat_epoch_s=time.time(),
        current_num_envs=None,current_group=None,child_pid=None,results=[],not_run=[],model_updates=0,
        qualification='provisional capacity; Chunk reference comparison deferred',run_dir=str(folder))
    try:
        for n,g,screening in STEPS:
            state.update(current_num_envs=n,current_group=g,screening_only=screening);write(folder/'status.json',state)
            command=[sys.executable,str(ROOT/'scripts/new_stomach_rl/benchmark_capacity.py'),
                '--num_envs',str(n),'--group',g,'--pose_manifest',config['pose_manifest'],'--mask',config['mask'],
                '--device','cuda:0','--viz','none','--kit_args=--/UJITSO/enabled=false --/UJITSO/geometry=false']
            if screening: command.append('--screening')
            data,summary=job(command,folder/f'{"screening_" if screening else ""}n{n}_{g}.log',state,folder)
            state['results'].append(dict(num_envs=n,group=g,screening_only=screening,status=data['status'],summary=str(summary),command=command))
            if (data['status']!=('screened' if screening else 'pass') or data.get('model_updates')!=0 or data.get('global_batches')!=(8 if screening else 224) or
                data.get('devices',{}).get('gpu_dynamics') is not True or
                (not screening and (len(data.get('timeouts_by_row',[]))!=n or min(data['timeouts_by_row'])<1 or
                (n>1 and (len(data.get('partial_reset_checks',[]))!=1 or
                    data['partial_reset_checks'][0]['unchanged'] is not True))))):
                raise RuntimeError(f'capacity failed or incomplete: {summary}')
            write(folder/'status.json',state)
        state.update(status='completed',current_num_envs=None,current_group=None)
    except Exception as error:
        state.update(status='paused_on_error',error=f'{type(error).__name__}: {error}')
        completed={(r['num_envs'],r['group'],r['screening_only']) for r in state['results']}
        attempted=(state['current_num_envs'],state['current_group'],state.get('screening_only'))
        state['not_run']=[dict(num_envs=n,group=g,screening_only=s) for n,g,s in STEPS if (n,g,s) not in completed and (n,g,s)!=attempted]
    finally:
        state.update(ended_epoch_s=time.time(),heartbeat_epoch_s=time.time());write(folder/'status.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['start','status','worker'])
    parser.add_argument('--pose_manifest',type=Path)
    parser.add_argument('--mask',type=Path)
    parser.add_argument('--single_summary',type=Path)
    parser.add_argument('--chunk_summary',type=Path)
    parser.add_argument('--allow_deferred_chunk_reference',action='store_true')
    parser.add_argument('--run_dir',type=Path,default=BASE/'latest')
    parser.add_argument('--kit_args')
    args=parser.parse_args()
    if args.command=='status': print((args.run_dir.resolve()/'status.json').read_text());return
    if args.command=='worker': worker(args.run_dir.resolve());return
    if not args.allow_deferred_chunk_reference: parser.error('explicit user-approved deferral flag required')
    if any(p is None or not p.is_file() for p in (args.pose_manifest,args.mask,args.single_summary,args.chunk_summary)):
        parser.error('frozen inputs and prior isolation summaries required')
    review=gate(args.single_summary,args.chunk_summary)
    pointer=BASE/'latest'
    if pointer.exists() and json.loads((pointer.resolve()/'status.json').read_text())['status'] in ('queued','running'):
        parser.error('capacity job already running; use status')
    isolation=ROOT/'artifacts/new_stomach_rl_capacity/acceptance_jobs/latest/status.json'
    if isolation.is_file() and json.loads(isolation.read_text())['status'] in ('queued','running'):
        parser.error('isolation GPU job is still running')
    folder=BASE/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');folder.mkdir(parents=True,exist_ok=False)
    write(folder/'launch.json',dict(pose_manifest=str(args.pose_manifest.resolve()),mask=str(args.mask.resolve()),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gate=review,
        ladder=list(LEVELS),groups=list('ABCD'),model_updates=0))
    write(folder/'status.json',dict(status='queued',run_dir=str(folder),model_updates=0))
    if pointer.is_symlink(): pointer.unlink()
    elif pointer.exists(): raise RuntimeError('refuse replacing a non-symlink latest')
    pointer.symlink_to(folder.name,target_is_directory=True)
    environment=os.environ.copy()
    environment.setdefault('ROBOTARM_MAGPYLIB_VENDOR','/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor')
    with (folder/'supervisor.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--run_dir',str(folder)],
            cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(status='started',supervisor_pid=child.pid,run_dir=str(folder),
        ladder=list(LEVELS),groups=list('ABCD'),model_updates=0),indent=2))


if __name__=='__main__': main()
