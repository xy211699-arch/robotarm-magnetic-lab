import numpy as np
import pytest
import hashlib
import json
from robotarm_magnetic_lab.runtime.new_stomach_rl_acceptance import (
    BOUNDS,measurements,mask_difference,register,compare,load_manifest)


def physics():
    value=np.zeros((2,240,50));value[...,6]=1;value[...,31]=1
    return value


def test_units_quaternion_sign_and_nonfinite_are_not_hidden():
    a=physics();b=a.copy();b[...,3:7]*=-1
    assert measurements('physics',a,b)['capsule_orientation_rad']==0
    b=a.copy();b[...,0]=.002;b[...,11]=4.
    m=measurements('physics',a,b)
    assert m['capsule_position_m']==.002 and m['angular_velocity_rad_s']==4.
    b[...,0]=np.nan
    with pytest.raises(ValueError): measurements('physics',a,b)


def test_register_cannot_exceed_predeclared_caps_or_omit_components():
    maxima={k:0. for k in BOUNDS}
    maxima['capsule_position_m']=.001
    result=register(maxima,'single','sha',[])
    assert result['limits']['capsule_position_m']==.003
    assert result['limits']['elapsed_s']==1e-6
    maxima['capsule_position_m']=.0031
    with pytest.raises(ValueError,match='ceiling'): register(maxima,'single','sha',[])
    with pytest.raises(ValueError,match='incomplete'): register({},'single','sha',[])


def test_weighted_mask_disagreement_not_vertex_count_or_padding():
    weights=np.array([9.,1.]);a=np.array([[0]],np.uint8)
    assert mask_difference(a,np.array([[2]],np.uint8),weights)==.1
    assert mask_difference(a,np.array([[1]],np.uint8),weights)==.9
    assert mask_difference(a,np.array([[128]],np.uint8),weights)==0.
    manifest=register({k:0. for k in BOUNDS},'single','sha',[])
    assert not compare('mask_visible_packed',a,np.array([[1]],np.uint8),manifest,'mask',weights)['passed']


def test_gross_pose_error_and_invalid_quaternion_cannot_be_relaxed():
    manifest=register({k:cap for k,(_,cap) in BOUNDS.items()},'single','sha',[])
    a=physics();b=a.copy();b[...,0]=.004
    assert not compare('physics',a,b,manifest,'position')['passed']
    b=a.copy();b[...,3:7]=0
    assert not compare('physics',a,b,manifest,'quaternion')['passed']


def test_manifest_identity_caps_and_evidence_are_checked_before_gpu(tmp_path):
    data=tmp_path/'calibration.npz';np.savez(data,physics=physics())
    artifact=dict(path=str(data),bytes=data.stat().st_size,sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    manifest=register({k:0. for k in BOUNDS},'single','original-sha',[artifact])
    file=tmp_path/'manifest.json';file.write_text(json.dumps(manifest))
    assert load_manifest(file,'single','original-sha')['limits']==manifest['limits']
    with pytest.raises(ValueError,match='identity'): load_manifest(file,'chunk','original-sha')
    with pytest.raises(ValueError,match='identity'): load_manifest(file,'single','changed-sha')
    manifest['limits']['capsule_position_m']=.5;file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='limit'): load_manifest(file,'single','original-sha')
    manifest['limits']['capsule_position_m']=.003;file.write_text(json.dumps(manifest))
    data.write_bytes(b'changed fixture')
    with pytest.raises(ValueError,match='changed'): load_manifest(file,'single','original-sha')
