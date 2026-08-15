"""Minimal state-dependent mask for the fifteen ideal actions."""

from __future__ import annotations

import numpy as np

from .config import IdealSurfaceConfig
from .types import IdealSurfaceAction, SurfaceFlags


def compute_action_mask(flags: SurfaceFlags, cfg: IdealSurfaceConfig) -> np.ndarray:
    """Return the frozen 15-entry mask without consulting coverage state."""
    del cfg  # Contract is state based in v1; retain the argument for versioning.
    mask = np.zeros(len(IdealSurfaceAction), dtype=np.bool_)
    mask[int(IdealSurfaceAction.HOLD)] = True
    if flags.upright:
        mask[1:9] = True
        return mask
    for action in (
        IdealSurfaceAction.TILT_MORE,
        IdealSurfaceAction.RISE,
        IdealSurfaceAction.PRECESS_POS,
        IdealSurfaceAction.PRECESS_NEG,
    ):
        mask[int(action)] = True
    if flags.side_contact:
        mask[int(IdealSurfaceAction.ROLL_POS)] = True
        mask[int(IdealSurfaceAction.ROLL_NEG)] = True
    if flags.contact_limited:
        mask[int(IdealSurfaceAction.TILT_MORE)] = False
    return mask

