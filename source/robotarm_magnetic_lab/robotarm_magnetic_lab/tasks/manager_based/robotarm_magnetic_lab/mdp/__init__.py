# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""

from isaaclab.utils.module import lazy_export

lazy_export()

from .legacy_bridge import (  # noqa: F401
    LegacyMagneticCollisionBridge,
    asm_clearance,
    collision_detected,
    collision_penalty,
    magnetic_wrench,
)
from .magnetic_action import MagneticPhysicsAction, MagneticPhysicsActionCfg  # noqa: F401
from .atomic_action import (  # noqa: F401
    AtomicMagnetAction,
    AtomicMagnetActionCfg,
    external_magnet_state,
)
from .vision import capsule_depth, capsule_rgb  # noqa: F401
