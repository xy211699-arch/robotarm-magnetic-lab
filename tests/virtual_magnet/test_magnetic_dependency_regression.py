from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "robotarm_magnetic_lab"
LEGACY_ROOT = Path("/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim")
LEGACY_VENDOR = LEGACY_ROOT / "vendor"


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_system():
    sys.path.append(str(LEGACY_VENDOR))
    config_module = _load_file(
        "task007_legacy_config",
        LEGACY_ROOT / "robotarm/magnetic_sim/config.py",
    )
    field_module = _load_file(
        "task007_legacy_field",
        LEGACY_ROOT / "robotarm/magnetic_sim/magnetics/field_models.py",
    )
    return field_module.FiniteMagnetSystem(config_module.load_config(LEGACY_ROOT))


def _poses():
    rng = np.random.default_rng(7007)
    for index in range(32):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        separation = rng.uniform(0.055, 0.14)
        cube_position = direction * separation
        cube_rotation = Rotation.from_rotvec(rng.normal(size=3) * 0.65).as_matrix()
        cylinder_position = rng.uniform(-0.01, 0.01, size=3)
        cylinder_rotation = Rotation.from_rotvec(rng.normal(size=3) * 0.8).as_matrix()
        observers = np.stack(
            [cylinder_position, cylinder_position + np.array([0.003, -0.002, 0.001])]
        )
        yield index, cube_position, cube_rotation, cylinder_position, cylinder_rotation, observers


def test_repository_model_matches_verified_legacy_model(monkeypatch):
    sys.path.insert(0, str(SOURCE_ROOT))
    monkeypatch.setenv("ROBOTARM_MAGPYLIB_VENDOR", str(LEGACY_VENDOR))
    from robotarm_magnetic_lab.magnetics import FiniteMagnetSystem, load_config

    legacy = _legacy_system()
    local = FiniteMagnetSystem(load_config())

    for _, cube_p, cube_r, cylinder_p, cylinder_r, observers in _poses():
        pairs = (
            (
                local.field_tesla(observers, cube_p, cube_r),
                legacy.field_tesla(observers, cube_p, cube_r),
            ),
            (
                local.force_torque_si(cube_p, cube_r, cylinder_p, cylinder_r),
                legacy.force_torque_si(cube_p, cube_r, cylinder_p, cylinder_r),
            ),
            (
                local.force_torque_on_cube_si(cylinder_p, cylinder_r, cube_p, cube_r),
                legacy.force_torque_on_cube_si(cylinder_p, cylinder_r, cube_p, cube_r),
            ),
        )
        for local_values, legacy_values in pairs:
            local_flat = np.concatenate([np.ravel(item) for item in local_values]) if isinstance(local_values, tuple) else np.ravel(local_values)
            legacy_flat = np.concatenate([np.ravel(item) for item in legacy_values]) if isinstance(legacy_values, tuple) else np.ravel(legacy_values)
            assert np.isfinite(local_flat).all()
            np.testing.assert_allclose(local_flat, legacy_flat, rtol=1.0e-9, atol=1.0e-13)


def test_repository_package_has_no_legacy_absolute_path(monkeypatch):
    sys.path.insert(0, str(SOURCE_ROOT))
    monkeypatch.setenv("ROBOTARM_MAGPYLIB_VENDOR", str(LEGACY_VENDOR))
    import robotarm_magnetic_lab.magnetics.config as config_module
    import robotarm_magnetic_lab.magnetics.field_models as field_module

    source = inspect.getsource(config_module) + inspect.getsource(field_module)
    assert "/mnt/isaac-linux/isaacsim/extsUser" not in source
    assert "importlib.resources" in source


def test_legacy_source_hashes_match_authority():
    import hashlib

    expected = {
        LEGACY_ROOT / "robotarm/magnetic_sim/config.py": "5d32740c62a75e06b7b876ed16f0043378ad45b72317b1f99637466b7f71ee07",
        LEGACY_ROOT / "robotarm/magnetic_sim/magnetics/field_models.py": "be2f4d4af8db2e3a04552add61cbbc84d89e2348c08864a0c9cc3e6283265965",
        LEGACY_ROOT / "data/config/default.json": "e38563d558f6945f3041458060965ce6cd4b7044eacce573318c0f0fdcd319a6",
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
