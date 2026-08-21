# Copyright (c) 2026, robotarm magnetic simulation contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop controllers for the simplified magnetic table benchmark."""

from .table_motion import (
    ArmGradientPlan,
    BallFieldPlanner,
    MotionMode,
    arm_gradient_plan,
    axis_from_tilt_azimuth,
    capsule_support_height,
    quaternion_from_axis,
    quintic_smoothstep,
    quaternion_from_y_rotation,
)
from .action_layer import AtomicAction, AtomicActionExecutor, ActionLayerConfig
from .dynamic_force_macro import (
    DynamicForceMacroActionId,
    DynamicForceMacroConfig,
    MacroPhase,
    NumericalContractError,
    PointForce,
    equivalent_com_wrench,
    lateral_direction_world,
    phase_for_substep,
    point_forces_for_action,
)

__all__ = [
    "ArmGradientPlan",
    "BallFieldPlanner",
    "MotionMode",
    "arm_gradient_plan",
    "axis_from_tilt_azimuth",
    "capsule_support_height",
    "quaternion_from_axis",
    "quaternion_from_y_rotation",
    "quintic_smoothstep",
    "ActionLayerConfig",
    "AtomicAction",
    "AtomicActionExecutor",
    "DynamicForceMacroActionId",
    "DynamicForceMacroConfig",
    "MacroPhase",
    "NumericalContractError",
    "PointForce",
    "equivalent_com_wrench",
    "lateral_direction_world",
    "phase_for_substep",
    "point_forces_for_action",
]
