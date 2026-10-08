"""Gate 0 identity guards, independent of Kit and GPU startup."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'scripts/new_stomach_rl/validate_preflight.py'


def load_audit():
    spec = importlib.util.spec_from_file_location('preflight_audit', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reject_old_stomach_pose_manifest(tmp_path):
    module = load_audit()
    path = tmp_path / 'old.json'
    path.write_text(json.dumps({'stomach_geometry_sha256': 'old-stomach'}))
    with pytest.raises(ValueError, match='new-stomach geometry'):
        module.audit_pose_manifest(path)


def test_mask_geometry_identity(tmp_path):
    module = load_audit()
    np.savez(tmp_path / 'wrong.npz', labels=np.zeros(12, dtype=np.int8))
    with pytest.raises(ValueError, match='mask file SHA'):
        module.audit_mask(tmp_path / 'wrong.npz', ROOT)


def test_release_action_semantics():
    module = load_audit()
    result = module.audit_release_semantics(ROOT)
    assert result['preview_seconds'] == 4
    assert result['committed_ticks'] == 240
    assert result['single_env_guard']
    assert result['required_clearance_m'] == .005


def test_missing_manifest_is_explicit(tmp_path):
    module = load_audit()
    with pytest.raises(FileNotFoundError):
        module.audit_pose_manifest(tmp_path / 'absent.json')


def test_existing_candidate_library_identity():
    module = load_audit()
    result = module.audit_pose_manifest(module.DEFAULT_MANIFEST)
    assert result['split_counts'] == {'train': 1000, 'validation': 100, 'test': 100}
    assert result['unique_poses'] == 1200
    assert result['stomach_geometry_sha256'] == module.RUNTIME_MESH_SHA


def test_existing_mask_identity_and_membership():
    module = load_audit()
    result = module.audit_mask(module.DEFAULT_MASK, ROOT)
    assert result['total_faces'] == 429029
    assert result['excluded_faces'] == 176109
    assert result['target_faces'] == 252920


@pytest.mark.parametrize('status', ['pass', 'fail'])
def test_evidence_is_persisted_before_kit_shutdown(tmp_path, status):
    module = load_audit()
    report = {'status': status}

    def shutdown():
        assert json.loads((tmp_path / 'summary.json').read_text()) == report
        raise SystemExit(0)

    with pytest.raises(SystemExit):
        module.persist_before_shutdown(report, tmp_path, shutdown)


@pytest.mark.parametrize('height', [.5, -.1])
def test_signed_lumen_distance_unpacks_vectorized_closest_points(height):
    module = load_audit()
    vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [10., 0., 0.], [11., 0., 0.], [10., 1., 0.]])
    faces = np.asarray([[0, 1, 2], [3, 4, 5]])
    assert module.signed_lumen_distance(np.asarray([.2, .2, height]), vertices, faces, block_size=1) == pytest.approx(height)
