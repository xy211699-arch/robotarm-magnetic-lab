"""Create identity-bound finite R2 inputs; never launches Kit or training."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
from supervise_training import ROOT,capacity_timeouts,write
from validate_preflight import audit_assets,inventory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose_manifest',type=Path,required=True)
    p.add_argument('--mask',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--capacity_summary',type=Path,required=True)
    p.add_argument('--capacity_boundaries',type=Path,required=True)
    p.add_argument('--output_dir',type=Path)
    p.add_argument('--wall_seconds',type=float,help='User-requested wall duration including initialization/shutdown; no default limit')
    p.add_argument('--time_limited_probe',action='store_true',help='Requested time expiry is interrupted, never full R2 pass')
    p.add_argument('--kit_args')
    args=p.parse_args()
    if args.time_limited_probe and args.wall_seconds is None:p.error('probe requires explicit --wall_seconds')
    if args.wall_seconds is not None and args.wall_seconds<=22:p.error('duration must include shutdown grace')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        p.error('commit implementation before freezing R2 identity')
    assets=audit_assets(args.pose_manifest,args.mask)
    weights=inventory(args.weights)
    if weights['sha256']!='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec':
        p.error('original frozen ResNet18 weight bytes required; no download/fallback')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    limits=capacity_timeouts(args.capacity_summary,args.capacity_boundaries)
    manifest=json.loads(args.pose_manifest.read_text())
    folder=args.output_dir or ROOT/'artifacts/new_stomach_rl_capacity/training_readiness/configs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder=folder.resolve()
    if not folder.is_relative_to((ROOT/'artifacts').resolve()):p.error('R2 inputs belong under this project artifacts')
    folder.mkdir(parents=True,exist_ok=False)
    evidence=[dict(path=x['path'],sha256=x['sha256']) for x in
        (inventory(args.pose_manifest),inventory(args.mask),weights,inventory(Path(manifest['data_path'])))]
    evidence+=limits.pop('evidence_files')
    # Bind mutable local source/config/asset bytes, not just the Git HEAD label.
    for name in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines():
        if name.startswith(('source/','scripts/new_stomach_rl/','configs/','assets/','dependencies/')):
            item=inventory(ROOT/name);evidence.append(dict(path=item['path'],sha256=item['sha256']))
    for group in 'ABCD':
        config=dict(purpose='r2_smoke',group=group,num_envs=8,max_updates=3,rollout_steps=64,
            seed=1008,device='cuda:0',visualizer='none',max_wall_seconds=args.wall_seconds,stop_grace_seconds=20,
            time_limited_probe=args.time_limited_probe,
            pose_manifest=str(args.pose_manifest.resolve()),mask=str(args.mask.resolve()),
            weights=weights,implementation_head=head,assets=assets,evidence_files=evidence,
            fixed_train_pose=manifest['fixed_live_reload_pose_ids']['train'][0],
            fixed_validation_pose=manifest['fixed_live_reload_pose_ids']['validation'][0],
            fixed_replay_seconds=120,partial_reset_boundary=5,restore_before_update=3,
            gamma=.999**10,gae_lambda=.95,epochs=2,sequence_length=64,
            minibatch='whole_valid_sequence',formal_training_allowed=False,**limits)
        write(folder/f'{group}.json',config)
    print(json.dumps(dict(status='prepared_not_started',config_directory=str(folder),head=head),indent=2))


if __name__=='__main__':main()
