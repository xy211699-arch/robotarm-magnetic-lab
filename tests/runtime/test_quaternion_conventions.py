import math

import numpy as np

from robotarm_magnetic_lab.runtime.quaternion_conventions import (
    ISAAC_QUATERNION_ORDER,
    rotation_matrix_from_xyzw,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def test_isaac_contract_is_xyzw():
    assert ISAAC_QUATERNION_ORDER == "xyzw"


def test_xyzw_identity_and_quarter_turn_about_z():
    assert np.allclose(rotation_matrix_from_xyzw([0.0, 0.0, 0.0, 1.0]), np.eye(3))
    half = math.sqrt(0.5)
    rotation = rotation_matrix_from_xyzw([0.0, 0.0, half, half])
    assert np.allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1.0e-12)


def test_legacy_boundary_round_trip_preserves_rotation():
    xyzw = np.asarray([0.2, -0.3, 0.1, 0.9], dtype=np.float64)
    xyzw /= np.linalg.norm(xyzw)
    assert np.allclose(wxyz_to_xyzw(xyzw_to_wxyz(xyzw)), xyzw)


def test_wrong_identity_order_is_not_silently_accepted_as_identity():
    # [1,0,0,0] is wxyz identity, but in Isaac xyzw it is a 180-degree X turn.
    rotation = rotation_matrix_from_xyzw([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(rotation @ [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], atol=1.0e-12)
