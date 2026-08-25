"""Runtime helpers shared by TASK-008 scripts and future policy adapters."""

from .dynamic_force_macro_runner import MacroTransition, SynchronousMacroRunner
from .quaternion_conventions import (
    ISAAC_QUATERNION_ORDER,
    normalized_xyzw,
    rotation_matrix_from_xyzw,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

__all__ = [
    "ISAAC_QUATERNION_ORDER",
    "MacroTransition",
    "SynchronousMacroRunner",
    "normalized_xyzw",
    "rotation_matrix_from_xyzw",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
