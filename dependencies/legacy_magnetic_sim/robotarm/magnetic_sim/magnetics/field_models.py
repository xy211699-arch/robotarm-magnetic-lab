"""Finite-size analytical magnetic fields and meshed force integration."""

import os
import sys
from pathlib import Path

import numpy as np


def _load_magpylib():
    extension_root = Path(__file__).resolve().parents[3]
    vendor = str(extension_root / "vendor")
    if vendor not in sys.path:
        # Append so Isaac Sim's tested NumPy/SciPy builds retain precedence.
        sys.path.append(vendor)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/robotarm_magnetic_sim_matplotlib")
    import magpylib as magpy
    from scipy.spatial.transform import Rotation

    return magpy, Rotation


class FiniteMagnetSystem:
    """N52 cuboid source plus a finite cylindrical target.

    Magnetic B-fields use Magpylib's vectorized analytical solutions. Force
    and torque use target-only meshing and finite-difference field gradients.
    All public inputs and outputs use SI units.
    """

    def __init__(self, config):
        self._magpy, self._rotation = _load_magpylib()
        self._cube_cfg = config["magnets"]["main_cube"]
        self._cylinder_cfg = config["magnets"]["target_cylinder"]

    @staticmethod
    def _polarization(config):
        axis = np.asarray(config["polarization_axis_local"], dtype=float)
        axis /= np.linalg.norm(axis)
        return axis * float(config["remanence_t"])

    def _cube(self, position, rotation_matrix, meshing=None):
        kwargs = {}
        if meshing is not None:
            kwargs["meshing"] = int(meshing)
        return self._magpy.magnet.Cuboid(
            position=np.asarray(position, dtype=float),
            orientation=self._rotation.from_matrix(np.asarray(rotation_matrix, dtype=float)),
            dimension=tuple(self._cube_cfg["dimensions_m"]),
            polarization=self._polarization(self._cube_cfg),
            **kwargs,
        )

    def _cylinder(self, position, rotation_matrix):
        return self._magpy.magnet.Cylinder(
            position=np.asarray(position, dtype=float),
            orientation=self._rotation.from_matrix(np.asarray(rotation_matrix, dtype=float)),
            dimension=(
                float(self._cylinder_cfg["diameter_m"]),
                float(self._cylinder_cfg["height_m"]),
            ),
            polarization=self._polarization(self._cylinder_cfg),
            meshing=int(self._cylinder_cfg["force_meshing"]),
        )

    def field_tesla(self, observer_points_world, cube_position_world, cube_rotation_world):
        source = self._cube(cube_position_world, cube_rotation_world)
        points = np.asarray(observer_points_world, dtype=float).reshape(-1, 3)
        return np.asarray(self._magpy.getB(source, points), dtype=float)

    def force_torque_si(
        self,
        cube_position_world,
        cube_rotation_world,
        cylinder_position_world,
        cylinder_rotation_world,
    ):
        source = self._cube(cube_position_world, cube_rotation_world)
        target = self._cylinder(cylinder_position_world, cylinder_rotation_world)
        force, torque = self._magpy.getFT(source, target, squeeze=True)
        return np.asarray(force, dtype=float), np.asarray(torque, dtype=float)

    def force_torque_on_cube_si(
        self,
        cylinder_position_world,
        cylinder_rotation_world,
        cube_position_world,
        cube_rotation_world,
    ):
        """Return force/torque on the ball-mounted cube from the ground magnet."""
        source = self._cylinder(cylinder_position_world, cylinder_rotation_world)
        target = self._cube(
            cube_position_world,
            cube_rotation_world,
            meshing=self._cube_cfg.get("force_meshing", 216),
        )
        force, torque = self._magpy.getFT(source, target, squeeze=True)
        return np.asarray(force, dtype=float), np.asarray(torque, dtype=float)
