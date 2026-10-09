"""有限GPU验收串行执行器；不训练、不自动调参、不并发使用GPU。"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/new_stomach_rl_capacity/acceptance_jobs'
STEPS=(('single_verify','single',False),('chunk_calibration','chunk',True),('chunk_verify','chunk',False))


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run_job(command,log,state,folder):
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid;write(folder/'status.json',state)
        while child.poll() is None:
            state['heartbeat_epoch_s']=time.time();write(folder/'status.json',state)
            time.sleep(5)
        state['child_pid']=None;state['child_exit_code']=child.returncode
    lines=log.read_text(errors='replace').splitlines()
    results=[json.loads(x.split('VECTOR_ISOLATION_RESULT ',1)[1])
             for x in lines if x.startswith('VECTOR_ISOLATION_RESULT ')]
    if len(results)!=1: raise RuntimeError(f'没有唯一结果事件: {log}')
    result=results[0]
    summary=Path(result['output'])/'summary.json'
    data=json.loads(summary.read_text())
    # Kit may exit with 0 even on failed validation: summary is authoritative.
    if child.returncode!=0 or data['status']!=result['status']:
        raise RuntimeError(f'子进程失败或状态不一致: exit={child.returncode}, summary={summary}')
    return data,summary


def worker(folder,job=run_job):
    config=json.loads((folder/'launch.json').read_text())
    state=dict(status='running',started_epoch_s=time.time(),heartbeat_epoch_s=time.time(),
        current_stage=None,child_pid=None,stages=[],model_updates=0,run_dir=str(folder))
    manifests={'single':config['single_manifest']}
    try:
        for index,(name,mode,calibration) in enumerate(STEPS):
            state['current_stage']=name;write(folder/'status.json',state)
            command=[sys.executable,str(ROOT/'scripts/new_stomach_rl/validate_vector_isolation.py'),
                '--num_envs','2','--seconds','120','--mode',mode,
                '--pose_manifest',config['pose_manifest'],'--mask',config['mask'],
                '--device','cuda:0','--viz','none',
                '--kit_args=--/UJITSO/enabled=false --/UJITSO/geometry=false']
            command+=['--calibrate_repeatability'] if calibration else ['--acceptance_manifest',manifests[mode]]
            data,summary=job(command,folder/f'{index}_{name}.log',state,folder)
            state['stages'].append(dict(stage=name,status=data['status'],summary=str(summary),command=command))
            expected='calibrated' if calibration else 'pass'
            if data['status']!=expected: raise RuntimeError(f'{name}未通过: {summary}')
            if data.get('model_updates')!=0 or data.get('devices',{}).get('gpu_dynamics') is not True:
                raise RuntimeError('验收设备或零训练边界不符合要求')
            if calibration:
                if len(data['runs'])!=3 or any(x['global_batches']!=20 for x in data['runs']):
                    raise RuntimeError('标定次数/预算不完整')
                manifests[mode]=str(summary.parent/'acceptance_manifest.json')
            else:
                if (len(data['runs'])!=2 or any(x['global_batches']!=121 or
                    x['row_1_valid_transitions']!=120 or x['row_1_timeout_count']!=1 for x in data['runs']) or
                    len(data['comparisons'])!=19 or not all(x['passed'] for x in data['comparisons']) or
                    len(data['reset_invariants'])!=7 or
                    not all(all(x['checks'].values()) for x in data['reset_invariants'])):
                    raise RuntimeError('完整120秒隔离证据不齐全')
            write(folder/'status.json',state)
        state.update(status='completed',current_stage=None,ended_epoch_s=time.time())
    except Exception as error:
        state.update(status='paused_on_error',error=f'{type(error).__name__}: {error}',ended_epoch_s=time.time())
    finally:
        state['heartbeat_epoch_s']=time.time();write(folder/'status.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['start','status','worker'])
    parser.add_argument('--single_manifest',type=Path)
    parser.add_argument('--pose_manifest',type=Path)
    parser.add_argument('--mask',type=Path)
    parser.add_argument('--run_dir',type=Path,default=BASE/'latest')
    parser.add_argument('--kit_args')  # launcher injection; never launch Kit here
    args=parser.parse_args()
    if args.command=='status':
        print((args.run_dir.resolve()/'status.json').read_text());return
    if args.command=='worker':
        worker(args.run_dir.resolve());return
    if any(p is None or not p.is_file() for p in (args.single_manifest,args.pose_manifest,args.mask)):
        parser.error('start需要已通过的Single独立标定清单、已确认的位姿库和掩码')
    # Prevent accidentally launching a second GPU validation pipeline.
    pointer=BASE/'latest'
    if pointer.exists():
        old=json.loads((pointer.resolve()/'status.json').read_text())
        if old['status'] in ('queued','running'):
            parser.error(f'已有验收任务，先查询状态: {pointer.resolve()}')
    folder=BASE/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');folder.mkdir(parents=True,exist_ok=False)
    write(folder/'launch.json',dict(single_manifest=str(args.single_manifest.resolve()),
        pose_manifest=str(args.pose_manifest.resolve()),mask=str(args.mask.resolve()),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        stages=[x[0] for x in STEPS],model_updates=0))
    write(folder/'status.json',dict(status='queued',run_dir=str(folder),model_updates=0))
    if pointer.is_symlink(): pointer.unlink()
    elif pointer.exists(): raise RuntimeError('不覆盖非软链的latest目录')
    pointer.symlink_to(folder.name,target_is_directory=True)
    environment=os.environ.copy()
    environment.setdefault('ROBOTARM_MAGPYLIB_VENDOR','/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor')
    with (folder/'supervisor.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--run_dir',str(folder)],
            cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(status='started',supervisor_pid=child.pid,run_dir=str(folder),
        stages=[x[0] for x in STEPS],model_updates=0),indent=2))


if __name__=='__main__': main()
