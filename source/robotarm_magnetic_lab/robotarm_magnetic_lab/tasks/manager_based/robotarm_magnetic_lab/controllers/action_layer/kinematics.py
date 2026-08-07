"""Small URDF/XRDF kinematic safety model for planned arm targets.

This module intentionally avoids Isaac Sim and capsule state.  It evaluates
the same XRDF self-collision spheres at candidate arm configurations, enabling
PRECHECK to reject unsafe trajectories before they are sent to PhysX or a real
controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    x, y, z = axis
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one = 1.0 - c
    return np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=np.float64,
    )


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _parse_vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in text.split()], dtype=np.float64)


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class UrdfXrdfSafetyModel:
    """Forward kinematics plus ASM-to-arm signed sphere clearance."""

    def __init__(
        self,
        urdf_path: str | Path,
        xrdf_path: str | Path,
        *,
        joint_names: tuple[str, ...] = ("j1", "j2", "j3", "j4", "j5", "j6"),
        asm_frame: str = "l6",
        ignored_frames: tuple[str, ...] = (),
    ) -> None:
        self.urdf_path = Path(urdf_path)
        self.xrdf_path = Path(xrdf_path)
        self.joint_names = tuple(joint_names)
        self.asm_frame = asm_frame
        self.ignored_frames = set(ignored_frames)
        self.joints = self._load_urdf()
        self.spheres = self._load_xrdf()
        if asm_frame not in self.spheres:
            raise ValueError(f"XRDF contains no spheres for ASM frame {asm_frame!r}")

    def _load_urdf(self) -> list[_Joint]:
        root = ET.parse(self.urdf_path).getroot()
        by_name = {element.attrib["name"]: element for element in root.findall("joint")}
        result: list[_Joint] = []
        for name in self.joint_names:
            element = by_name.get(name)
            if element is None:
                raise ValueError(f"URDF joint not found: {name}")
            parent = element.find("parent").attrib["link"]
            child = element.find("child").attrib["link"]
            origin_element = element.find("origin")
            xyz = _parse_vector(
                None if origin_element is None else origin_element.attrib.get("xyz"),
                (0.0, 0.0, 0.0),
            )
            rpy = _parse_vector(
                None if origin_element is None else origin_element.attrib.get("rpy"),
                (0.0, 0.0, 0.0),
            )
            axis_element = element.find("axis")
            axis = _parse_vector(
                None if axis_element is None else axis_element.attrib.get("xyz"),
                (1.0, 0.0, 0.0),
            )
            result.append(
                _Joint(
                    name=name,
                    parent=parent,
                    child=child,
                    origin=_transform(_rpy_matrix(rpy), xyz),
                    axis=axis,
                )
            )
        return result

    def _load_xrdf(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        import yaml

        with self.xrdf_path.open("r", encoding="utf-8") as stream:
            xrdf = yaml.safe_load(stream)
        geometry_name = xrdf["self_collision"]["geometry"]
        source = xrdf["geometry"][geometry_name]["spheres"]
        return {
            frame: (
                np.asarray([item["center"] for item in items], dtype=np.float64),
                np.asarray([item["radius"] for item in items], dtype=np.float64),
            )
            for frame, items in source.items()
        }

    def link_transforms(self, arm_configuration_rad: np.ndarray) -> dict[str, np.ndarray]:
        """Return link transforms in the robot base frame."""
        configuration = np.asarray(arm_configuration_rad, dtype=np.float64).reshape(-1)
        if configuration.size != len(self.joints):
            raise ValueError(
                f"expected {len(self.joints)} arm joints, got {configuration.size}"
            )
        transforms: dict[str, np.ndarray] = {self.joints[0].parent: np.eye(4)}
        for joint, angle in zip(self.joints, configuration, strict=True):
            if joint.parent not in transforms:
                raise ValueError(f"URDF chain is disconnected at {joint.name}")
            motion = _transform(_axis_rotation(joint.axis, float(angle)), np.zeros(3))
            transforms[joint.child] = transforms[joint.parent] @ joint.origin @ motion
        return transforms

    def world_spheres(
        self, arm_configuration_rad: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        transforms = self.link_transforms(arm_configuration_rad)
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for frame, (centers, radii) in self.spheres.items():
            transform = transforms.get(frame)
            if transform is None:
                continue
            world_centers = transform[:3, 3] + centers @ transform[:3, :3].T
            result[frame] = (world_centers, radii)
        return result

    def asm_clearance_by_frame(
        self, arm_configuration_rad: np.ndarray
    ) -> dict[str, float]:
        spheres = self.world_spheres(arm_configuration_rad)
        asm_centers, asm_radii = spheres[self.asm_frame]
        result: dict[str, float] = {}
        for frame, (centers, radii) in spheres.items():
            if frame == self.asm_frame or frame in self.ignored_frames:
                continue
            delta = asm_centers[:, None, :] - centers[None, :, :]
            signed = np.linalg.norm(delta, axis=2) - asm_radii[:, None] - radii[None, :]
            result[frame] = float(np.min(signed))
        return result

    def minimum_asm_clearance(self, arm_configuration_rad: np.ndarray) -> float:
        values = self.asm_clearance_by_frame(arm_configuration_rad)
        return min(values.values()) if values else math.inf

    def minimum_sphere_height(self, arm_configuration_rad: np.ndarray) -> float:
        """Return the lowest robot collision-sphere surface in base coordinates."""
        values = []
        for centers, radii in self.world_spheres(arm_configuration_rad).values():
            values.extend((centers[:, 2] - radii).tolist())
        return min(values) if values else math.inf

    def validate_path(
        self,
        arm_targets_rad: np.ndarray,
        *,
        required_asm_clearance_m: float = 0.0,
        ground_height_m: float | None = None,
    ) -> dict[str, float | int | str | bool]:
        """Validate every sampled arm target before execution."""
        targets = np.asarray(arm_targets_rad, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != len(self.joints):
            raise ValueError("arm_targets_rad has an invalid shape")
        minimum_clearance = math.inf
        minimum_height = math.inf
        for index, configuration in enumerate(targets):
            clearance = self.minimum_asm_clearance(configuration)
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance < required_asm_clearance_m:
                return {
                    "ok": False,
                    "kind": "ASM_CLEARANCE",
                    "sample_index": index,
                    "minimum_asm_clearance_m": clearance,
                }
            height = self.minimum_sphere_height(configuration)
            minimum_height = min(minimum_height, height)
            if ground_height_m is not None and height < ground_height_m:
                return {
                    "ok": False,
                    "kind": "ENVIRONMENT_COLLISION",
                    "sample_index": index,
                    "minimum_sphere_height_m": height,
                }
        return {
            "ok": True,
            "kind": "CLEAR",
            "sample_index": len(targets) - 1,
            "minimum_asm_clearance_m": minimum_clearance,
            "minimum_sphere_height_m": minimum_height,
        }
