"""Explicit quaternion conversions at Isaac Lab runtime boundaries.

Isaac Lab 3.0 rigid-body tensors and pose writers use ``xyzw``.  Some
controller-only geometry in this project predates that API and intentionally
uses ``wxyz``.  Keeping all conversions here prevents silent component
reordering from leaking into simulation code.
"""

from __future__ import annotations

import numpy as np


ISAAC_QUATERNION_ORDER = "xyzw"


def normalized_xyzw(quaternion) -> np.ndarray:
    """Return a finite unit quaternion in Isaac Lab's ``xyzw`` order."""
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm <= 1.0e-12:
        raise ValueError("Isaac Lab xyzw quaternion must be finite and non-zero")
    return value / norm


def rotation_matrix_from_xyzw(quaternion) -> np.ndarray:
    """Convert an Isaac Lab ``xyzw`` quaternion to a rotation matrix."""
    x, y, z, w = normalized_xyzw(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def xyzw_to_wxyz(quaternion) -> np.ndarray:
    """Convert an Isaac Lab quaternion to an internal legacy ``wxyz`` value."""
    x, y, z, w = normalized_xyzw(quaternion)
    return np.asarray([w, x, y, z], dtype=np.float64)


def wxyz_to_xyzw(quaternion) -> np.ndarray:
    """Convert an internal legacy ``wxyz`` value to Isaac Lab ``xyzw``."""
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm <= 1.0e-12:
        raise ValueError("internal wxyz quaternion must be finite and non-zero")
    w, x, y, z = value / norm
    return np.asarray([x, y, z, w], dtype=np.float64)
