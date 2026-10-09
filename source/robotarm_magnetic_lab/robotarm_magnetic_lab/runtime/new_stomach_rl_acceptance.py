"""Approved clone-repeatability gates; never alter simulation parameters.

Bounds are declared before collecting calibration. A separate, immutable
manifest is required for relaxed verification; the original P0 stays intact.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


# SI component limits, not a single dimensionally-invalid tensor tolerance.
BOUNDS = {
    'capsule_position_m': (.0002, .003),
    'capsule_orientation_rad': (np.deg2rad(.5), np.deg2rad(8)),
    'linear_velocity_m_s': (.005, .05),
    'angular_velocity_rad_s': (.5, 8.),
    'source_position_m': (1e-6, 5e-5),
    'source_orientation_rad': (1e-5, 1e-3),
    'actual_joints_rad': (1e-6, 1e-4),
    'issued_joints_rad': (2e-7, 1e-6),
    'force_n': (1e-5, .001),
    'torque_nm': (1e-5, .0003),
    'elapsed_s': (1e-6, 1e-5),
    'mask_area_fraction': (.002, .04),
}
VERSION = 'clone-repeatability-v1'


def difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('shape/nonfinite mismatch')
    return float(np.abs(a.astype(float)-b.astype(float)).max(initial=0))


def angle(a, b):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    difference(a,b)
    # Never treat zero/non-unit quaternions as valid poses.
    na,nb = np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
    if np.any(np.abs(na-1)>.001) or np.any(np.abs(nb-1)>.001):
        raise ValueError('invalid quaternion norm')
    dot=np.abs(np.sum(a*b,axis=-1)/(na*nb))
    return float((2*np.arccos(np.clip(dot,0,1))).max(initial=0))


def measurements(key,a,b):
    difference(a,b)
    if key=='physics':
        result=dict(capsule_position_m=difference(a[...,:3],b[...,:3]),
            capsule_orientation_rad=angle(a[...,3:7],b[...,3:7]),
            linear_velocity_m_s=difference(a[...,7:10],b[...,7:10]),
            angular_velocity_rad_s=difference(a[...,10:13],b[...,10:13]),
            source_position_m=difference(a[...,25:28],b[...,25:28]),
            source_orientation_rad=angle(a[...,28:32],b[...,28:32]),
            actual_joints_rad=difference(a[...,32:41],b[...,32:41]),
            issued_joints_rad=difference(a[...,41:50],b[...,41:50]))
        wrench=a[...,13:25];other=b[...,13:25]
    elif key=='magnetic_inputs':
        result=dict(source_position_m=difference(a[...,:3],b[...,:3]),
            source_orientation_rad=angle(a[...,3:7],b[...,3:7]),
            capsule_position_m=difference(a[...,7:10],b[...,7:10]),
            capsule_orientation_rad=angle(a[...,10:14],b[...,10:14]),
            linear_velocity_m_s=difference(a[...,14:17],b[...,14:17]),
            angular_velocity_rad_s=difference(a[...,17:20],b[...,17:20]),
            elapsed_s=difference(a[...,32:33],b[...,32:33]))
        wrench=a[...,20:32];other=b[...,20:32]
    elif key=='magnetic_outputs':
        result=dict(elapsed_s=difference(a[...,24:25],b[...,24:25]))
        wrench=np.concatenate((a[...,:12],a[...,12:24]),axis=-1)
        other=np.concatenate((b[...,:12],b[...,12:24]),axis=-1)
    else: raise ValueError('unknown signal')
    force=[i for i in range(wrench.shape[-1]) if i%6<3]
    torque=[i for i in range(wrench.shape[-1]) if i%6>=3]
    result.update(force_n=difference(wrench[...,force],other[...,force]),
                  torque_nm=difference(wrench[...,torque],other[...,torque]))
    return result


def mask_difference(a,b,weights):
    difference(a,b)
    weights=np.asarray(weights,dtype=float)
    if weights.ndim!=1 or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('invalid frozen area weights')
    if a.shape[-1]!=(len(weights)+7)//8: raise ValueError('packed mask size mismatch')
    bits=np.unpackbits(np.bitwise_xor(a,b),axis=-1,bitorder='little')[...,:len(weights)]
    return float(((bits@weights)/weights.sum()).max(initial=0))


def register(maxima, mode, source_sha, artifacts):
    if set(maxima)!=set(BOUNDS): raise ValueError('incomplete repeatability calibration')
    limits={}
    for name,(floor,cap) in BOUNDS.items():
        value=float(maxima[name])
        if not np.isfinite(value) or value<0 or value>cap:
            raise ValueError(f'calibration exceeds predefined ceiling: {name}={value}, cap={cap}')
        limits[name]=min(cap,max(floor,3*value))
    return dict(version=VERSION,mode=mode,original_registration_sha256=source_sha,
        bounds={name:list(v) for name,v in BOUNDS.items()},limits=limits,
        observed_calibration_maxima=maxima,calibration_artifacts=artifacts,
        repetitions=3,seconds_per_repetition=20,model_updates=0)


def load_manifest(path, mode, source_sha):
    item=json.loads(Path(path).read_text())
    if (item.get('version')!=VERSION or item.get('mode')!=mode or
        item.get('original_registration_sha256')!=source_sha or item.get('repetitions')!=3 or
        item.get('seconds_per_repetition')!=20 or item.get('model_updates')!=0 or
        item.get('bounds')!={name:list(v) for name,v in BOUNDS.items()}):
        raise ValueError('acceptance identity/bounds mismatch')
    if set(item['limits'])!=set(BOUNDS): raise ValueError('incomplete limits')
    for name,(floor,cap) in BOUNDS.items():
        value=item['limits'][name]
        if not np.isfinite(value) or not floor<=value<=cap: raise ValueError('invalid acceptance limit')
    for artifact in item['calibration_artifacts']:
        file=Path(artifact['path'])
        if file.stat().st_size!=artifact['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest()!=artifact['sha256']:
            raise ValueError('calibration artifact changed')
    return item


def compare(key,a,b,manifest,label,weights=None):
    try:
        values=({'mask_area_fraction':mask_difference(a,b,weights)} if 'packed' in key
                else measurements(key,a,b))
        checks={name:dict(error=value,limit=manifest['limits'][name],passed=value<=manifest['limits'][name])
            for name,value in values.items()}
        return dict(label=label,passed=all(v['passed'] for v in checks.values()),components=checks)
    except ValueError as error:
        return dict(label=label,passed=False,reason=str(error))
