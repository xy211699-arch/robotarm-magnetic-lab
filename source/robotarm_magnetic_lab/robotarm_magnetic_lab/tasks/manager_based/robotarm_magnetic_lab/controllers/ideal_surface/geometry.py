"""Small deterministic Euclidean and rotation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def normalized(value: np.ndarray, *, name: str = "vector") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or norm <= 1.0e-12:
        raise ValueError(f"{name} must be finite and non-zero")
    return result / norm


def quintic(tau: float) -> float:
    value = float(np.clip(float(tau), 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = normalized(axis, name="rotation axis")
    cosine, sine = math.cos(float(angle_rad)), math.sin(float(angle_rad))
    one_minus = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
            [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
            [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
        ],
        dtype=np.float64,
    )


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray(
            [0.25 * scale, (value[2, 1] - value[1, 2]) / scale,
             (value[0, 2] - value[2, 0]) / scale, (value[1, 0] - value[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(value)))
        if index == 0:
            scale = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
            q = np.asarray([(value[2, 1] - value[1, 2]) / scale, 0.25 * scale,
                            (value[0, 1] + value[1, 0]) / scale, (value[0, 2] + value[2, 0]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
            q = np.asarray([(value[0, 2] - value[2, 0]) / scale,
                            (value[0, 1] + value[1, 0]) / scale, 0.25 * scale,
                            (value[1, 2] + value[2, 1]) / scale])
        else:
            scale = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
            q = np.asarray([(value[1, 0] - value[0, 1]) / scale,
                            (value[0, 2] + value[2, 0]) / scale,
                            (value[1, 2] + value[2, 1]) / scale, 0.25 * scale])
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def orientation_from_axis_and_image_up(axis_world: np.ndarray, image_up_world: np.ndarray) -> np.ndarray:
    """Return wxyz orientation whose local +Z/+Y match the supplied axes."""
    z_axis = normalized(axis_world, name="capsule axis")
    y_candidate = np.asarray(image_up_world, dtype=np.float64).reshape(3)
    y_candidate -= float(y_candidate @ z_axis) * z_axis
    if float(np.linalg.norm(y_candidate)) <= 1.0e-12:
        fallback = np.asarray([0.0, 1.0, 0.0])
        if abs(float(fallback @ z_axis)) > 0.9:
            fallback = np.asarray([1.0, 0.0, 0.0])
        y_candidate = fallback - float(fallback @ z_axis) * z_axis
    y_axis = normalized(y_candidate, name="image up")
    x_axis = normalized(np.cross(y_axis, z_axis), name="capsule local x")
    y_axis = normalized(np.cross(z_axis, x_axis), name="capsule local y")
    return quaternion_wxyz_from_matrix(np.column_stack((x_axis, y_axis, z_axis)))


@dataclass(frozen=True)
class LocalFrame:
    point_world: np.ndarray
    normal_world: np.ndarray
    image_up_tangent_world: np.ndarray

    def __post_init__(self) -> None:
        point = np.asarray(self.point_world, dtype=np.float64).reshape(3)
        normal = normalized(self.normal_world, name="surface normal")
        first = np.asarray(self.image_up_tangent_world, dtype=np.float64).reshape(3)
        first -= float(first @ normal) * normal
        first = normalized(first, name="projected image up")
        object.__setattr__(self, "point_world", point.copy())
        object.__setattr__(self, "normal_world", normal)
        object.__setattr__(self, "image_up_tangent_world", first)

    @property
    def e1(self) -> np.ndarray:
        return self.image_up_tangent_world.copy()

    @property
    def e2(self) -> np.ndarray:
        return normalized(np.cross(self.normal_world, self.e1), name="surface e2")

    def direction(self, phi_rad: float) -> np.ndarray:
        return normalized(math.cos(float(phi_rad)) * self.e1 + math.sin(float(phi_rad)) * self.e2)


def closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return closest point and barycentric coordinates (Ericson regions)."""
    point = np.asarray(point, dtype=np.float64).reshape(3)
    a, b, c = np.asarray(triangle, dtype=np.float64).reshape(3, 3)
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), np.asarray([1.0, 0.0, 0.0])
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), np.asarray([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.asarray([1.0 - v, v, 0.0])
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), np.asarray([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.asarray([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.asarray([0.0, 1.0 - w, w])
    denominator = 1.0 / (va + vb + vc)
    v, w = vb * denominator, vc * denominator
    return a + v * ab + w * ac, np.asarray([1.0 - v - w, v, w])

