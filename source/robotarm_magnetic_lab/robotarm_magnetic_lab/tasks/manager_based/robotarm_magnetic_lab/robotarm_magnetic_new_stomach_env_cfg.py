"""Isolated environment configuration for the horizontally aligned new stomach."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.configclass import configclass

from .robotarm_magnetic_stomach_env_cfg import (
    RobotarmMagneticStomachLabEnvCfg,
    RobotarmMagneticStomachSceneCfg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[6]
NEW_STOMACH_ASSET_DIR = PROJECT_ROOT / "assets" / "stomach" / "new_stomach_v1"
NEW_STOMACH_ASSET_USD_PATH = str(NEW_STOMACH_ASSET_DIR / "new_stomach_physics.usda")
NEW_STOMACH_ORIENTATION_PATH = PROJECT_ROOT / "configs" / "new_stomach_v1" / "orientation_v1.json"


@dataclass(frozen=True)
class NewStomachGeometryConfig:
    """Validated immutable placement and geometry identity for the new stomach."""

    asset_usd_sha256: str
    position_world_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    collision_mesh_suffix: str
    opening_centers_world_m: tuple[tuple[float, float, float], ...]
    opening_axes_world: tuple[tuple[float, float, float], ...]
    opening_elevation_deg: tuple[float, float]
    world_bounds_min_m: tuple[float, float, float]
    world_bounds_max_m: tuple[float, float, float]
    confirmed: bool
    config_sha256: str

    @property
    def rotation_wxyz(self) -> tuple[float, float, float, float]:
        x, y, z, w = self.rotation_xyzw
        return (w, x, y, z)

    @classmethod
    def from_json(cls, path: str | Path) -> "NewStomachGeometryConfig":
        path = Path(path)
        payload = path.read_bytes()
        record = json.loads(payload)
        if record.get("schema") != "robotarm_magnetic_lab.new_stomach_orientation.v1":
            raise ValueError("unsupported new-stomach orientation schema")

        def vector(name: str, size: int) -> tuple[float, ...]:
            value = np.asarray(record[name], dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite {size}-vector")
            return tuple(float(item) for item in value)

        position = vector("position_world_m", 3)
        rotation = vector("rotation_xyzw", 4)
        scale = vector("scale", 3)
        if not np.allclose(scale, 1.0, atol=0.0):
            raise ValueError("new stomach scale must remain exactly one")
        norm = float(np.linalg.norm(rotation))
        if not np.isclose(norm, 1.0, atol=1.0e-10):
            raise ValueError("new stomach XYZW quaternion must be normalized")

        centers = np.asarray(record["opening_centers_world_m"], dtype=np.float64)
        axes = np.asarray(record["opening_axes_world"], dtype=np.float64)
        elevations = np.asarray(record["opening_elevation_deg"], dtype=np.float64)
        if centers.shape != (2, 3) or axes.shape != (2, 3) or elevations.shape != (2,):
            raise ValueError("orientation config must contain exactly two opening fits")
        if not np.isfinite(centers).all() or not np.isfinite(axes).all() or not np.isfinite(elevations).all():
            raise ValueError("opening fits must be finite")
        axis_norms = np.linalg.norm(axes, axis=1)
        if not np.allclose(axis_norms, 1.0, atol=1.0e-9):
            raise ValueError("opening axes must be normalized")
        calculated = np.degrees(np.arcsin(np.clip(np.abs(axes[:, 2]), 0.0, 1.0)))
        if float(calculated.max()) > 1.0 or not np.allclose(calculated, elevations, atol=1.0e-9):
            raise ValueError("both configured opening axes must be horizontal")

        suffix = str(record["collision_mesh_suffix"])
        if not suffix.startswith("/") or ".." in suffix:
            raise ValueError("collision_mesh_suffix must be an absolute prim suffix")
        bounds_min = vector("world_bounds_min_m", 3)
        bounds_max = vector("world_bounds_max_m", 3)
        if np.any(np.asarray(bounds_max) <= np.asarray(bounds_min)):
            raise ValueError("new stomach world bounds must have positive extent")
        return cls(
            asset_usd_sha256=str(record["asset_usd_sha256"]),
            position_world_m=position,
            rotation_xyzw=rotation,
            scale=scale,
            collision_mesh_suffix=suffix,
            opening_centers_world_m=tuple(tuple(float(item) for item in row) for row in centers),
            opening_axes_world=tuple(tuple(float(item) for item in row) for row in axes),
            opening_elevation_deg=tuple(float(item) for item in elevations),
            world_bounds_min_m=bounds_min,
            world_bounds_max_m=bounds_max,
            confirmed=bool(record.get("confirmed", False)),
            config_sha256=hashlib.sha256(payload).hexdigest(),
        )


NEW_STOMACH_GEOMETRY = NewStomachGeometryConfig.from_json(NEW_STOMACH_ORIENTATION_PATH)


@configclass
class RobotarmMagneticNewStomachSceneCfg(RobotarmMagneticStomachSceneCfg):
    """Existing robot/capsule scene with only the stomach asset replaced."""

    stomach = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stomach",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=NEW_STOMACH_GEOMETRY.position_world_m,
            rot=NEW_STOMACH_GEOMETRY.rotation_wxyz,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=NEW_STOMACH_ASSET_USD_PATH,
            scale=NEW_STOMACH_GEOMETRY.scale,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class RobotarmMagneticNewStomachLabEnvCfg(RobotarmMagneticStomachLabEnvCfg):
    """Single-environment new-stomach migration task."""

    scene: RobotarmMagneticNewStomachSceneCfg = RobotarmMagneticNewStomachSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )

