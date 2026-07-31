# Copyright (c) 2026, robotarm magnetic simulation contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Visual and headless acceptance harness for the flat-table benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_DIR = Path("/mnt/isaac-linux/robotarm_magnetic_lab")
DEFAULT_TASK = "Template-Robotarm-Magnetic-Table-Lab-v0"
RESULT_ROOT = PROJECT_DIR / "logs" / "table_motion"
BALL_ACTION_INDICES = (6, 7, 8)
ARM_ACTION_INDICES = (0, 1, 2, 3, 4, 5)
CAPSULE_POS_XY = (1.0608155, 0.1145374)
CAPSULE_RADIUS_M = 0.0065
CAPSULE_LENGTH_M = 0.025
CONTACT_THRESHOLD_N = 1.0e-4
COMPOSITE_ARM_ACTION_SCALE_RAD = 0.45
COMPOSITE_BALL_ACTION_SCALE_RAD = math.pi
# The camera/optical head is capsule local -Z, opposite the axially-magnetized
# local +Z axis used by the field model.  A magnetic-axis elevation of -45 deg
# therefore makes the camera head point +45 deg above world XY.  The trajectory
# helper accepts polar tilt from world +Z, so the equivalent input is 135 deg.
AZIMUTH_AXIS_ELEVATION_DEG = -45.0
AZIMUTH_HEAD_ELEVATION_DEG = -AZIMUTH_AXIS_ELEVATION_DEG
AZIMUTH_POLAR_TILT_DEG = 90.0 - AZIMUTH_AXIS_ELEVATION_DEG

SCENARIO_LABELS = {
    "baseline": "No-magnet table contact and friction baseline",
    "field_scan": "Axial magnetization and fixed-point field scan",
    "tilt_azimuth": "Fixed-tilt azimuth sweep",
    "upright_to_side": "Upright-to-side posture transition",
    "long_axis_roll": "Long-axis rolling by arm-generated field gradient",
    "composite_motion": (
        "Side-upright-45deg revolution-side-100mm roll sequence"
    ),
}


def _is_headless(args) -> bool:
    if bool(getattr(args, "headless", False)):
        return True
    if "--visualizer" in sys.argv:
        index = sys.argv.index("--visualizer")
        if index + 1 < len(sys.argv) and sys.argv[index + 1].lower() == "none":
            return True
    value = getattr(args, "visualizer", None)
    if isinstance(value, (list, tuple)):
        return any(str(item).lower() == "none" for item in value)
    return str(value).lower() == "none"


def _tolist(value, digits: int = 7) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return [
        round(float(item), digits)
        for item in np.asarray(value).reshape(-1)
    ]


def _quat_xyzw_to_matrix(quaternion) -> np.ndarray:
    x, y, z, w = np.asarray(_tolist(quaternion, 12), dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_world(quaternion) -> np.ndarray:
    return _quat_xyzw_to_matrix(quaternion)[:, 2]


def _angle_deg(first, second, *, unsigned_axis: bool = False) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if unsigned_axis:
        dot = abs(dot)
    return math.degrees(math.acos(dot))


class TableTestHarness:
    """Single-environment passive-capsule test driver."""

    def __init__(
        self,
        env,
        args,
        scenario: str,
        *,
        result_root: Path = RESULT_ROOT,
        environment_label: str = "TABLE",
        dry_surface: bool = True,
    ):
        import torch

        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import (
            BallFieldPlanner,
        )

        self.BallFieldPlanner = BallFieldPlanner
        self.env = env
        self.base_env = env.unwrapped
        self.scene = self.base_env.scene
        self.robot = self.scene["robot"]
        self.capsule = self.scene["capsule"]
        self.contact = self.scene["capsule_contact"]
        self.bridge = self.base_env.event_manager.get_term_cfg(
            "magnetic_collision_bridge"
        ).func
        # The flat benchmark is dry, while the stomach migration must retain
        # the configured gastric-fluid drag.  Both environments otherwise use
        # exactly the same open-loop magnetic controller.
        if dry_surface:
            self.bridge.config["external_magnet"]["capsule"][
                "angular_drag_nm_per_rad_s"
            ] = 0.0
        self.args = args
        self.scenario = scenario
        self.environment_label = environment_label
        self.ball_action_scale_rad = (
            COMPOSITE_BALL_ACTION_SCALE_RAD
            if scenario == "composite_motion"
            else math.pi / 2.0
        )
        self.device = self.base_env.device
        self.step_dt = float(self.base_env.step_dt)
        self.action = torch.zeros(env.action_space.shape, device=self.device)
        self.arm_indices = [
            self.robot.data.joint_names.index(name)
            for name in ("j1", "j2", "j3", "j4", "j5", "j6")
        ]
        self.ball_indices = [
            self.robot.data.joint_names.index(name)
            for name in ("ballxj", "ballyj", "ballzj")
        ]
        self.magnet_body_index = self.robot.data.body_names.index("magl")
        self._magnetic_enabled = True
        self._roll_angle_rad = 0.0
        self._previous_position = None
        self.records: list[dict[str, Any]] = []
        self.global_step = 0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = result_root / scenario / stamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.output_dir / "telemetry.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self._stream = self.telemetry_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._stream.close()

    def set_magnetic_forces(self, enabled: bool) -> None:
        """Enable or disable only the analytical magnetic wrench."""
        self._magnetic_enabled = bool(enabled)
        self.bridge.config["simulation"]["apply_forces"] = bool(enabled)
        if not enabled:
            self.bridge._filtered_wrench.zero_()
            self.bridge.robot.permanent_wrench_composer.reset()
            self.bridge.capsule.permanent_wrench_composer.reset()

    def reset(
        self,
        axis_world: np.ndarray,
        *,
        magnetic: bool,
        clearance_m: float = 0.00025,
    ) -> None:
        """Reset and place the capsule at the plane support height."""
        import torch

        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import (
            capsule_support_height,
            quaternion_from_axis,
        )

        self.env.reset()
        self.action.zero_()
        self.set_magnetic_forces(False)
        axis = np.asarray(axis_world, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        quat = quaternion_from_axis(axis)
        pose = self.capsule.data.root_pose_w.torch.clone()
        pose[:, :3] = torch.tensor(
            [
                CAPSULE_POS_XY[0],
                CAPSULE_POS_XY[1],
                capsule_support_height(axis) + clearance_m,
            ],
            device=self.device,
        )
        pose[:, 3:7] = torch.tensor(quat, device=self.device)
        self.capsule.write_root_pose_to_sim_index(root_pose=pose)
        self.capsule.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=self.device)
        )
        self._roll_angle_rad = 0.0
        self._previous_position = pose[0, :3].detach().clone()
        self.set_magnetic_forces(magnetic)

    def replace_capsule_pose(
        self,
        axis_world: np.ndarray,
        *,
        clearance_m: float = 0.00025,
    ) -> None:
        """Place the capsule without resetting robot/ball pre-positioning."""
        import torch

        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import (
            capsule_support_height,
            quaternion_from_axis,
        )

        axis = np.asarray(axis_world, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        pose = self.capsule.data.root_pose_w.torch.clone()
        pose[:, :3] = torch.tensor(
            [
                CAPSULE_POS_XY[0],
                CAPSULE_POS_XY[1],
                capsule_support_height(axis) + clearance_m,
            ],
            device=self.device,
        )
        pose[:, 3:7] = torch.tensor(
            quaternion_from_axis(axis), device=self.device
        )
        self.capsule.write_root_pose_to_sim_index(root_pose=pose)
        self.capsule.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=self.device)
        )
        self._roll_angle_rad = 0.0
        self._previous_position = pose[0, :3].detach().clone()

    def make_field_planner(self):
        """Build a finite-field inverse at the current nominal reset geometry."""
        main_position = _tolist(
            self.robot.data.body_pos_w.torch[0, self.magnet_body_index], 12
        )
        main_rotation = _quat_xyzw_to_matrix(
            self.robot.data.body_quat_w.torch[0, self.magnet_body_index]
        )
        ball_position = _tolist(
            self.robot.data.joint_pos.torch[0, self.ball_indices], 12
        )
        ball_default = _tolist(
            self.robot.data.default_joint_pos.torch[0, self.ball_indices], 12
        )
        capsule_position = np.asarray(
            _tolist(self.capsule.data.root_pos_w.torch[0], 12), dtype=np.float64
        )
        capsule_rotation = _quat_xyzw_to_matrix(
            self.capsule.data.root_quat_w.torch[0]
        )
        offset = np.array(
            [
                0.0,
                0.0,
                float(
                    self.bridge.config["magnets"]["target_cylinder"][
                        "center_offset_axis_m"
                    ]
                ),
            ]
        )
        nominal_target = capsule_position + capsule_rotation @ offset
        return self.BallFieldPlanner(
            self.bridge.model,
            np.asarray(main_position),
            main_rotation,
            np.asarray(ball_position),
            nominal_target,
            np.asarray(ball_default),
            action_scale_rad=self.ball_action_scale_rad,
        )

    def set_ball_action(self, values) -> None:
        for index, value in zip(BALL_ACTION_INDICES, values, strict=True):
            self.action[:, index] = float(value)

    def set_arm_action(self, values) -> None:
        for index, value in zip(ARM_ACTION_INDICES, values, strict=True):
            self.action[:, index] = float(value)

    def run_phase(
        self,
        phase: str,
        duration_s: float,
        profile: Callable[[float], tuple[np.ndarray, np.ndarray] | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run one phase with optional arm/ball normalized action profile."""
        import torch

        steps = max(int(round(duration_s / self.step_dt)), 1)
        result = []
        for local_step in range(steps):
            fraction = local_step / max(steps - 1, 1)
            if profile is not None:
                command = profile(fraction)
                if command is not None:
                    arm, ball = command
                    self.set_arm_action(arm)
                    self.set_ball_action(ball)
            started = time.perf_counter()
            with torch.inference_mode():
                _, _, terminated, truncated, _ = self.env.step(self.action)
            record = self.capture(phase, local_step, terminated, truncated)
            self.records.append(record)
            result.append(record)
            self._stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            if self.global_step % self.args.log_every == 0:
                print(
                    f"[{self.environment_label}_TEST] "
                    f"scenario={self.scenario} phase={phase} "
                    f"t={record['sim_time_s']:.2f}s pos={record['position_m']} "
                    f"axis={record['capsule_axis_world']} "
                    f"cone_deg={record['rotation_axis_to_capsule_deg']:.1f} "
                    f"head_elev_deg={record['camera_head_elevation_deg']:.1f} "
                    f"omega_axis_deg={record['angular_velocity_axis_to_capsule_deg']:.1f} "
                    f"field_mT={1000.0 * record['field_magnitude_T']:.3f} "
                    f"contact={record['in_contact']} "
                    f"roll={record['roll_angle_rad']:.3f}",
                    flush=True,
                )
            self.global_step += 1
            if self.args.realtime and not _is_headless(self.args):
                remaining = self.step_dt - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        return result

    def capture(self, phase, local_step, terminated, truncated) -> dict[str, Any]:
        """Capture policy-rate physics and magnetic diagnostics."""
        import torch

        position = self.capsule.data.root_pos_w.torch[0]
        quaternion = self.capsule.data.root_quat_w.torch[0]
        velocity = self.capsule.data.root_lin_vel_w.torch[0]
        angular_velocity = self.capsule.data.root_ang_vel_w.torch[0]
        axis = _axis_world(quaternion)
        roll_rate = float(
            torch.dot(
                angular_velocity,
                torch.tensor(axis, device=self.device, dtype=torch.float32),
            ).item()
        )
        angular_speed = float(torch.linalg.norm(angular_velocity).item())
        if angular_speed > 1.0e-8:
            angular_axis_to_capsule_deg = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            abs(roll_rate) / angular_speed,
                            0.0,
                            1.0,
                        )
                    )
                )
            )
        else:
            angular_axis_to_capsule_deg = 0.0
        # The azimuth trajectory revolves about world +Z. This geometric cone
        # angle is distinct from the body's instantaneous angular-velocity axis.
        rotation_axis_to_capsule_deg = math.degrees(
            math.acos(float(np.clip(abs(float(axis[2])), 0.0, 1.0)))
        )
        capsule_axis_elevation_deg = math.degrees(
            math.asin(float(np.clip(float(axis[2]), -1.0, 1.0)))
        )
        camera_head_elevation_deg = -capsule_axis_elevation_deg
        self._roll_angle_rad += roll_rate * self.step_dt
        main_position_t = self.robot.data.body_pos_w.torch[
            0, self.magnet_body_index
        ]
        main_quaternion = self.robot.data.body_quat_w.torch[
            0, self.magnet_body_index
        ]
        main_position = np.asarray(_tolist(main_position_t, 12))
        main_rotation = _quat_xyzw_to_matrix(main_quaternion)
        capsule_position = np.asarray(_tolist(position, 12))
        capsule_rotation = _quat_xyzw_to_matrix(quaternion)
        cylinder_offset = np.array(
            [
                0.0,
                0.0,
                float(
                    self.bridge.config["magnets"]["target_cylinder"][
                        "center_offset_axis_m"
                    ]
                ),
            ]
        )
        observer = capsule_position + capsule_rotation @ cylinder_offset
        field = self.bridge.model.field_tesla(
            observer, main_position, main_rotation
        ).reshape(-1, 3)[0]
        field_magnitude = float(np.linalg.norm(field))
        field_direction = field / max(field_magnitude, 1.0e-12)
        main_axis = main_rotation[:, 2]
        contact_force = self.contact.data.net_forces_w.torch[0, 0]
        contact_norm = float(torch.linalg.norm(contact_force).item())
        wrench = self.bridge.state["wrench"][0]
        applied_wrench = wrench if self._magnetic_enabled else torch.zeros_like(wrench)
        clearance = float(self.bridge.state["asm_clearance"][0, 0].item())
        support_height = CAPSULE_RADIUS_M + (
            0.5 * CAPSULE_LENGTH_M - CAPSULE_RADIUS_M
        ) * abs(float(axis[2]))
        ground_gap = float(position[2].item()) - support_height

        return {
            "scenario": self.scenario,
            "phase": phase,
            "step": self.global_step,
            "phase_step": local_step,
            "sim_time_s": round(self.global_step * self.step_dt, 7),
            "position_m": _tolist(position),
            "quaternion_xyzw": _tolist(quaternion),
            "capsule_axis_world": _tolist(axis),
            "linear_velocity_mps": _tolist(velocity),
            "angular_velocity_radps": _tolist(angular_velocity),
            "linear_speed_mps": float(torch.linalg.norm(velocity).item()),
            "angular_speed_radps": angular_speed,
            "rotation_axis_to_capsule_deg": rotation_axis_to_capsule_deg,
            "capsule_axis_elevation_deg": capsule_axis_elevation_deg,
            "camera_head_elevation_deg": camera_head_elevation_deg,
            "angular_velocity_axis_to_capsule_deg": (
                angular_axis_to_capsule_deg
            ),
            "roll_rate_radps": roll_rate,
            "roll_angle_rad": self._roll_angle_rad,
            "ground_gap_m": ground_gap,
            "contact_force_N": _tolist(contact_force),
            "contact_force_norm_N": contact_norm,
            "in_contact": contact_norm >= CONTACT_THRESHOLD_N,
            "main_magnet_position_m": _tolist(main_position_t),
            "main_magnet_axis_world": _tolist(main_axis),
            "field_T": _tolist(field),
            "field_direction_world": _tolist(field_direction),
            "field_magnitude_T": field_magnitude,
            "magnetic_force_N": _tolist(applied_wrench[6:9]),
            "magnetic_torque_Nm": _tolist(applied_wrench[9:12]),
            "computed_magnetic_force_N": _tolist(wrench[6:9]),
            "computed_magnetic_torque_Nm": _tolist(wrench[9:12]),
            "ball_joint_pos_rad": _tolist(
                self.robot.data.joint_pos.torch[0, self.ball_indices]
            ),
            "arm_joint_pos_rad": _tolist(
                self.robot.data.joint_pos.torch[0, self.arm_indices]
            ),
            "action": _tolist(self.action[0]),
            "asm_clearance_m": clearance,
            "asm_collision": bool(self.bridge.state["collision"][0, 0].item()),
            "magnetic_enabled": self._magnetic_enabled,
            "terminated": bool(terminated[0].item()),
            "truncated": bool(truncated[0].item()),
        }

    def save_summary(
        self,
        checks: dict[str, bool],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Write objective acceptance checks."""
        checks = {name: bool(value) for name, value in checks.items()}
        summary = {
            "schema_version": "1.0.0",
            "scenario": self.scenario,
            "label": SCENARIO_LABELS[self.scenario],
            "passed": all(checks.values()),
            "checks": checks,
            "metrics": metrics,
            "physics_dt_s": float(self.base_env.physics_dt),
            "policy_dt_s": self.step_dt,
            "telemetry_path": str(self.telemetry_path),
        }
        self.summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
                default=lambda value: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value.item()
                    if isinstance(value, np.generic)
                    else str(value)
                ),
            ),
            encoding="utf-8",
        )
        print(
            f"[{self.environment_label}_TEST_RESULT] "
            f"scenario={self.scenario} "
            f"result={'PASS' if summary['passed'] else 'FAIL'} "
            f"summary={self.summary_path} checks={json.dumps(checks)}",
            flush=True,
        )
        return summary


def _contact_ratio(records) -> float:
    return sum(record["in_contact"] for record in records) / max(len(records), 1)


def _interpolate_knots(knots: list[np.ndarray], fraction: float) -> np.ndarray:
    if len(knots) == 1:
        return knots[0]
    coordinate = min(max(fraction, 0.0), 1.0) * (len(knots) - 1)
    index = min(int(math.floor(coordinate)), len(knots) - 2)
    local = coordinate - index
    return (1.0 - local) * knots[index] + local * knots[index + 1]


def _prepare_ball_plan(
    harness: TableTestHarness,
    desired_axes: list[np.ndarray],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    planner = harness.make_field_planner()
    actions = []
    diagnostics = []
    for index, axis in enumerate(desired_axes):
        position, diagnostic = planner.solve(axis, global_search=index == 0)
        actions.append(planner.action_from_positions(position))
        diagnostics.append(diagnostic)
    return actions, diagnostics


def run_baseline(h: TableTestHarness) -> dict[str, Any]:
    import torch

    side_axis = np.array([1.0, 0.0, 0.0])
    h.reset(side_axis, magnetic=False)
    resting = h.run_phase("no_magnet_rest", 4.0)

    h.replace_capsule_pose(side_axis, clearance_m=0.015)
    drop = h.run_phase("drop_15mm", 4.0)

    h.replace_capsule_pose(side_axis)
    root_velocity = torch.zeros((1, 6), device=h.device)
    root_velocity[:, 1] = 0.03
    h.capsule.write_root_velocity_to_sim_index(root_velocity=root_velocity)
    slide = h.run_phase("friction_decay", 3.0)

    tail = slide[-20:]
    displacement = abs(slide[-1]["position_m"][1] - slide[0]["position_m"][1])
    checks = {
        "rest_contact": _contact_ratio(resting[-40:]) >= 0.8,
        "rest_settles": resting[-1]["linear_speed_mps"] < 0.01,
        "drop_contacts": any(record["in_contact"] for record in drop),
        "drop_no_tunneling": min(record["ground_gap_m"] for record in drop) > -0.002,
        "friction_decelerates": tail[-1]["linear_speed_mps"] < slide[0]["linear_speed_mps"],
        "friction_finite": all(
            math.isfinite(record["contact_force_norm_N"]) for record in slide
        ),
        "magnetic_wrench_zero": max(
            np.linalg.norm(record["magnetic_force_N"])
            for record in resting + drop + slide
        )
        < 1.0e-8,
    }
    return h.save_summary(
        checks,
        {
            "rest_final_speed_mps": resting[-1]["linear_speed_mps"],
            "drop_minimum_gap_m": min(record["ground_gap_m"] for record in drop),
            "slide_displacement_m": displacement,
            "slide_initial_speed_mps": slide[0]["linear_speed_mps"],
            "slide_final_speed_mps": slide[-1]["linear_speed_mps"],
        },
    )


def run_field_scan(h: TableTestHarness) -> dict[str, Any]:
    side_axis = np.array([1.0, 0.0, 0.0])
    h.reset(side_axis, magnetic=False)
    settle = h.run_phase("settle", 1.0)
    positions = (
        (-0.55, -0.30, 0.0),
        (-0.55, 0.0, 0.0),
        (-0.55, 0.30, 0.0),
        (0.0, -0.30, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.30, 0.0),
        (0.55, -0.30, 0.0),
        (0.55, 0.0, 0.0),
        (0.55, 0.30, 0.0),
    )
    scans = []
    zero_arm = np.zeros(6)
    for index, target in enumerate(positions):
        start = h.action[0, list(BALL_ACTION_INDICES)].detach().cpu().numpy().copy()
        target_np = np.asarray(target)
        records = h.run_phase(
            f"scan_{index:02d}",
            0.8,
            lambda u, a=start, b=target_np: (
                zero_arm,
                a + (b - a) * (10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5),
            ),
        )
        scans.append(records[-1])

    axis_cfg = np.asarray(
        h.bridge.config["magnets"]["target_cylinder"]["polarization_axis_local"],
        dtype=np.float64,
    )
    synthetic_fields = np.eye(3)
    moment = axis_cfg / np.linalg.norm(axis_cfg)
    axial_torque_components = [
        abs(float(np.dot(np.cross(moment, field), moment)))
        for field in synthetic_fields
    ]
    directions = np.asarray([record["field_direction_world"] for record in scans])
    spread = max(
        _angle_deg(first, second)
        for first in directions
        for second in directions
    )
    checks = {
        "target_magnet_axis_is_local_z": np.allclose(axis_cfg, [0.0, 0.0, 1.0]),
        "long_axis_torque_identity": max(axial_torque_components) < 1.0e-12,
        "all_fields_finite": all(
            math.isfinite(record["field_magnitude_T"]) for record in scans
        ),
        "field_direction_changes": spread >= 20.0,
        "capsule_stays_passive": max(
            np.linalg.norm(record["magnetic_force_N"]) for record in settle + scans
        )
        < 1.0e-8,
    }
    return h.save_summary(
        checks,
        {
            "configured_polarization_axis_local": axis_cfg.tolist(),
            "maximum_axial_torque_component": max(axial_torque_components),
            "field_direction_spread_deg": spread,
            "field_scan": [
                {
                    "ball_action": record["action"][6:9],
                    "field_T": record["field_T"],
                    "field_magnitude_T": record["field_magnitude_T"],
                }
                for record in scans
            ],
        },
    )


def run_tilt_azimuth(h: TableTestHarness) -> dict[str, Any]:
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import (
        axis_from_tilt_azimuth,
        quintic_smoothstep,
    )

    tilt = math.radians(AZIMUTH_POLAR_TILT_DEG)
    start_azimuth = math.radians(-175.0)
    azimuth_span = math.radians(80.0)
    desired_axes = [
        axis_from_tilt_azimuth(tilt, start_azimuth + azimuth_span * i / 8.0)
        for i in range(9)
    ]
    h.reset(desired_axes[0], magnetic=False)
    actions, plan_diagnostics = _prepare_ball_plan(h, desired_axes)
    zero_arm = np.zeros(6)
    h.run_phase(
        "preposition_ball_no_magnet",
        2.5,
        lambda u: (zero_arm, actions[0] * quintic_smoothstep(u)),
    )
    h.replace_capsule_pose(desired_axes[0])
    h.set_magnetic_forces(True)
    hold = h.run_phase("field_ramp_and_hold", 2.0)
    active = h.run_phase(
        "fixed_tilt_azimuth_sweep",
        10.0,
        lambda u: (zero_arm, _interpolate_knots(actions, quintic_smoothstep(u))),
    )
    final_hold = h.run_phase("final_hold", 2.0)

    start_axis = np.asarray(hold[-1]["capsule_axis_world"])
    final_axis = np.asarray(final_hold[-1]["capsule_axis_world"])
    start_azimuth = math.atan2(start_axis[1], start_axis[0])
    final_azimuth = math.atan2(final_axis[1], final_axis[0])
    azimuth_change = abs(math.degrees(math.atan2(
        math.sin(final_azimuth - start_azimuth),
        math.cos(final_azimuth - start_azimuth),
    )))
    # Keep polarity here: local -Z is the optical head, so treating +axis and
    # -axis as interchangeable would accept the exact head-down failure.
    final_error = _angle_deg(final_axis, desired_axes[-1], unsigned_axis=False)
    final_head_elevation = float(final_hold[-1]["camera_head_elevation_deg"])
    checks = {
        "finite_field_inverse": max(
            item["direction_error_deg"] for item in plan_diagnostics
        )
        < 8.0,
        "capsule_azimuth_changes": azimuth_change >= 20.0,
        "final_axis_bounded_error": final_error <= 35.0,
        "camera_head_points_up": final_head_elevation >= 20.0,
        "contact_retained": _contact_ratio(active) >= 0.70,
        "no_launch": max(record["ground_gap_m"] for record in active) < 0.003,
        "asm_clearance_safe": min(
            record["asm_clearance_m"] for record in active
        )
        >= 0.0,
    }
    return h.save_summary(
        checks,
        {
            "requested_capsule_axis_elevation_deg": (
                AZIMUTH_AXIS_ELEVATION_DEG
            ),
            "requested_camera_head_elevation_deg": (
                AZIMUTH_HEAD_ELEVATION_DEG
            ),
            "requested_polar_tilt_deg": AZIMUTH_POLAR_TILT_DEG,
            "measured_final_camera_head_elevation_deg": (
                final_head_elevation
            ),
            "requested_azimuth_change_deg": 80.0,
            "measured_azimuth_change_deg": azimuth_change,
            "final_axis_error_deg": final_error,
            "contact_ratio": _contact_ratio(active),
            "field_inverse_max_error_deg": max(
                item["direction_error_deg"] for item in plan_diagnostics
            ),
        },
    )


def run_upright_to_side(h: TableTestHarness) -> dict[str, Any]:
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import (
        axis_from_tilt_azimuth,
        quintic_smoothstep,
    )

    target_azimuth = math.radians(-135.0)
    desired_axes = [
        axis_from_tilt_azimuth(
            math.radians(88.0) * i / 10.0, target_azimuth
        )
        for i in range(11)
    ]
    h.reset(desired_axes[0], magnetic=False)
    actions, plan_diagnostics = _prepare_ball_plan(h, desired_axes)
    zero_arm = np.zeros(6)
    h.run_phase(
        "preposition_vertical_field_no_magnet",
        3.0,
        lambda u: (zero_arm, actions[0] * quintic_smoothstep(u)),
    )
    h.replace_capsule_pose(desired_axes[0])
    h.set_magnetic_forces(True)
    start_hold = h.run_phase("upright_field_hold", 1.5)
    active = h.run_phase(
        "smooth_tip_to_side",
        10.0,
        lambda u: (zero_arm, _interpolate_knots(actions, quintic_smoothstep(u))),
    )
    final_hold = h.run_phase("side_settle", 3.0)

    start_axis = np.asarray(start_hold[-1]["capsule_axis_world"])
    end_axis = np.asarray(final_hold[-1]["capsule_axis_world"])
    start_tilt = _angle_deg(start_axis, [0.0, 0.0, 1.0], unsigned_axis=True)
    end_tilt = _angle_deg(end_axis, [0.0, 0.0, 1.0], unsigned_axis=True)
    checks = {
        "finite_field_inverse": max(
            item["direction_error_deg"] for item in plan_diagnostics
        )
        < 8.0,
        "starts_near_upright": start_tilt <= 30.0,
        "ends_near_side": end_tilt >= 55.0,
        "tilt_progresses": end_tilt - start_tilt >= 35.0,
        "contact_during_transition": _contact_ratio(active) >= 0.65,
        "no_launch": max(record["ground_gap_m"] for record in active) < 0.004,
        "settles": final_hold[-1]["angular_speed_radps"] < 0.20,
    }
    return h.save_summary(
        checks,
        {
            "start_tilt_from_vertical_deg": start_tilt,
            "end_tilt_from_vertical_deg": end_tilt,
            "contact_ratio": _contact_ratio(active),
            "peak_gap_m": max(record["ground_gap_m"] for record in active),
            "final_angular_speed_radps": final_hold[-1]["angular_speed_radps"],
        },
    )


def run_long_axis_roll(h: TableTestHarness) -> dict[str, Any]:
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import (
        arm_gradient_plan,
        quintic_smoothstep,
    )

    axis = np.array(
        [-math.sqrt(0.5), -math.sqrt(0.5), 0.0], dtype=np.float64
    )
    h.reset(axis, magnetic=False)
    actions, plan_diagnostics = _prepare_ball_plan(h, [axis])
    zero_arm = np.zeros(6)
    h.run_phase(
        "preposition_axial_field_no_magnet",
        2.5,
        lambda u: (zero_arm, actions[0] * quintic_smoothstep(u)),
    )
    h.replace_capsule_pose(axis)
    # The legacy stomach scene limits the entire magnetic force vector to
    # 5 mN.  At the table reset separation this leaves less than 0.5 mN of
    # lateral force, below the calibrated static-friction threshold.  The
    # table benchmark uses a 40 mN cap: still below capsule weight (56 mN), so
    # even a fully vertical attractive force cannot lift the capsule.
    h.bridge.config["simulation"]["max_force_n"] = 0.040
    h.set_magnetic_forces(True)
    settle = h.run_phase("side_contact_field_hold", 2.0)

    body_index = h.magnet_body_index
    jacobian_index = body_index - 1 if h.robot.is_fixed_base else body_index
    jacobian = (
        h.robot.data.body_link_jacobian_w.torch[
            0, jacobian_index, :, h.arm_indices
        ]
        .detach()
        .cpu()
        .numpy()
    )
    # Shift the source magnet sideways and slightly toward the table.  The
    # lateral gradient pulls/pushes the capsule while static friction supplies
    # the rolling torque.  No capsule state enters this open-loop reference.
    roll_direction = np.cross(np.array([0.0, 0.0, 1.0]), axis)
    requested = 0.070 * roll_direction + np.array([0.0, 0.0, -0.050])
    gradient = arm_gradient_plan(
        jacobian,
        requested,
        action_scale_rad=0.25,
        max_joint_delta_rad=0.22,
    )
    active = h.run_phase(
        "arm_gradient_roll",
        10.0,
        lambda u: (
            gradient.normalized_action * quintic_smoothstep(u),
            actions[0],
        ),
    )
    final_hold = h.run_phase("gradient_hold", 2.0)

    start = np.asarray(settle[-1]["position_m"])
    end = np.asarray(final_hold[-1]["position_m"])
    displacement_roll = float(np.dot(end - start, roll_direction))
    roll_angle = float(
        final_hold[-1]["roll_angle_rad"] - settle[-1]["roll_angle_rad"]
    )
    rolling_distance = CAPSULE_RADIUS_M * abs(roll_angle)
    travel = abs(displacement_roll)
    slip_ratio = abs(travel - rolling_distance) / max(
        travel, rolling_distance, 1.0e-6
    )
    final_axis_error = _angle_deg(
        final_hold[-1]["capsule_axis_world"], axis, unsigned_axis=True
    )
    checks = {
        "field_inverse_valid": plan_diagnostics[0]["direction_error_deg"] < 8.0,
        "arm_gradient_nonzero": float(
            np.linalg.norm(gradient.predicted_displacement_world_m)
        )
        >= 0.005,
        "translation_observed": travel >= 0.0005,
        "long_axis_rotation_observed": abs(roll_angle) >= 0.10,
        "rolling_not_pure_sliding": slip_ratio <= 0.65,
        "axis_maintained": final_axis_error <= 25.0,
        "contact_retained": _contact_ratio(active) >= 0.75,
        "no_launch": max(record["ground_gap_m"] for record in active) < 0.003,
        "asm_clearance_safe": min(
            record["asm_clearance_m"] for record in active
        )
        >= 0.0,
    }
    return h.save_summary(
        checks,
        {
            "requested_magnet_displacement_m": requested.tolist(),
            "table_force_safety_cap_N": 0.040,
            "predicted_magnet_displacement_m": (
                gradient.predicted_displacement_world_m.tolist()
            ),
            "joint_delta_rad": gradient.joint_delta_rad.tolist(),
            "capsule_lateral_travel_m": displacement_roll,
            "roll_direction_world": roll_direction.tolist(),
            "integrated_long_axis_roll_rad": roll_angle,
            "rolling_arc_length_m": rolling_distance,
            "slip_ratio": slip_ratio,
            "final_axis_error_deg": final_axis_error,
            "contact_ratio": _contact_ratio(active),
        },
    )


def _unwrapped_azimuth_change_deg(records: list[dict[str, Any]]) -> float:
    """Return directed accumulated capsule-axis azimuth change."""
    azimuth = np.asarray(
        [
            math.atan2(
                record["capsule_axis_world"][1],
                record["capsule_axis_world"][0],
            )
            for record in records
        ],
        dtype=np.float64,
    )
    return math.degrees(float(np.unwrap(azimuth)[-1] - np.unwrap(azimuth)[0]))


def run_composite_motion(h: TableTestHarness) -> dict[str, Any]:
    """Run the complete passive-capsule open-loop motion sequence."""
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers import (
        arm_gradient_plan,
        axis_from_tilt_azimuth,
        quintic_smoothstep,
    )

    zero_arm = np.zeros(6)
    azimuth = math.radians(-90.0)
    side_axis = axis_from_tilt_azimuth(math.radians(90.0), azimuth)

    side_to_upright_axes = [
        axis_from_tilt_azimuth(math.radians(90.0 + 9.0 * index), azimuth)
        for index in range(11)
    ]
    tilt_out_axes = [
        axis_from_tilt_azimuth(
            math.radians(
                180.0
                + (AZIMUTH_POLAR_TILT_DEG - 180.0) * index / 6.0
            ),
            azimuth,
        )
        for index in range(7)
    ]
    revolution_axes = [
        axis_from_tilt_azimuth(
            math.radians(AZIMUTH_POLAR_TILT_DEG),
            azimuth + 2.0 * math.pi * index / 48.0,
        )
        for index in range(49)
    ]
    tilt_back_axes = [
        axis_from_tilt_azimuth(
            math.radians(
                AZIMUTH_POLAR_TILT_DEG
                + (180.0 - AZIMUTH_POLAR_TILT_DEG) * index / 6.0
            ),
            azimuth,
        )
        for index in range(7)
    ]
    upright_to_side_axes = [
        axis_from_tilt_azimuth(math.radians(180.0 - 9.0 * index), azimuth)
        for index in range(11)
    ]

    # Solve all segments with one continuous finite-field inverse. Duplicate
    # endpoint directions are retained deliberately so each phase starts and
    # ends at a zero-discontinuity command.
    axis_segments = (
        side_to_upright_axes,
        tilt_out_axes,
        revolution_axes,
        tilt_back_axes,
        upright_to_side_axes,
    )
    full_axes: list[np.ndarray] = []
    segment_slices: list[slice] = []
    for segment in axis_segments:
        start = len(full_axes)
        full_axes.extend(segment)
        segment_slices.append(slice(start, len(full_axes)))

    h.reset(side_axis, magnetic=False)
    full_actions, plan_diagnostics = _prepare_ball_plan(h, full_axes)
    side_to_upright_actions = full_actions[segment_slices[0]]
    tilt_out_actions = full_actions[segment_slices[1]]
    revolution_actions = full_actions[segment_slices[2]]
    tilt_back_actions = full_actions[segment_slices[3]]
    upright_to_side_actions = full_actions[segment_slices[4]]

    h.run_phase(
        "initialize_side_field_no_magnet",
        3.0,
        lambda u: (
            zero_arm,
            side_to_upright_actions[0] * quintic_smoothstep(u),
        ),
    )
    h.replace_capsule_pose(side_axis)
    h.set_magnetic_forces(True)
    initial_side_hold = h.run_phase("initial_side_hold", 2.0)

    side_to_upright = h.run_phase(
        "side_to_upright",
        10.0,
        lambda u: (
            zero_arm,
            _interpolate_knots(
                side_to_upright_actions, quintic_smoothstep(u)
            ),
        ),
    )
    upright_hold_1 = h.run_phase("upright_hold_before_revolution", 2.0)

    tilt_out = h.run_phase(
        "tilt_to_45deg",
        5.0,
        lambda u: (
            zero_arm,
            _interpolate_knots(tilt_out_actions, quintic_smoothstep(u)),
        ),
    )
    revolution = h.run_phase(
        "45deg_full_azimuth_revolution",
        24.0,
        lambda u: (
            zero_arm,
            _interpolate_knots(revolution_actions, quintic_smoothstep(u)),
        ),
    )
    tilt_back = h.run_phase(
        "return_to_upright",
        5.0,
        lambda u: (
            zero_arm,
            _interpolate_knots(tilt_back_actions, quintic_smoothstep(u)),
        ),
    )
    upright_hold_2 = h.run_phase("upright_hold_after_revolution", 2.0)

    upright_to_side = h.run_phase(
        "upright_to_side",
        10.0,
        lambda u: (
            zero_arm,
            _interpolate_knots(
                upright_to_side_actions, quintic_smoothstep(u)
            ),
        ),
    )
    side_hold = h.run_phase("side_hold_before_roll", 2.0)

    body_index = h.magnet_body_index
    jacobian_index = body_index - 1 if h.robot.is_fixed_base else body_index
    jacobian = (
        h.robot.data.body_link_jacobian_w.torch[
            0, jacobian_index, :, h.arm_indices
        ]
        .detach()
        .cpu()
        .numpy()
    )
    # For a side axis along world -Y, z x axis points along world +X.
    roll_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # The first collision-free j6-locked run measured 121.73 mm capsule
    # travel from a 125 mm source request. Apply that open-loop bench gain so
    # the commanded sequence lands close to the requested 100 mm travel.
    requested = np.array([0.103, 0.0, -0.050], dtype=np.float64)
    gradient = arm_gradient_plan(
        jacobian,
        requested,
        action_scale_rad=COMPOSITE_ARM_ACTION_SCALE_RAD,
        max_joint_delta_rad=0.38,
        # j6=-0.28688 rad is the collision-selected wrist roll in the saved
        # initialization. Letting a full-pose Cartesian inverse move it toward
        # zero folds the mounted ASM into l4 and triggers automatic reset.
        locked_joint_indices=(5,),
        # Ball supplies the magnetic direction; the arm is needed primarily
        # for the field-gradient translation. A small orientation weight gives
        # j1..j5 enough freedom to realize the requested +X stroke while j6
        # remains reserved for collision clearance.
        orientation_weight=0.10,
    )
    h.bridge.config["simulation"]["max_force_n"] = 0.040
    roll_start = side_hold[-1]
    roll = h.run_phase(
        "roll_world_positive_x",
        18.0,
        lambda u: (
            gradient.normalized_action * quintic_smoothstep(u),
            upright_to_side_actions[-1],
        ),
    )
    final_hold = h.run_phase("final_roll_hold", 6.0)

    initial_side_tilt = _angle_deg(
        initial_side_hold[-1]["capsule_axis_world"],
        [0.0, 0.0, 1.0],
        unsigned_axis=True,
    )
    upright_1_tilt = _angle_deg(
        upright_hold_1[-1]["capsule_axis_world"],
        [0.0, 0.0, 1.0],
        unsigned_axis=True,
    )
    revolution_tilts = [
        _angle_deg(
            record["capsule_axis_world"],
            [0.0, 0.0, 1.0],
            unsigned_axis=True,
        )
        for record in revolution
    ]
    revolution_omega_axis_angles = [
        record["angular_velocity_axis_to_capsule_deg"]
        for record in revolution
        if record["angular_speed_radps"] > 1.0e-4
    ]
    revolution_head_elevations = [
        record["camera_head_elevation_deg"] for record in revolution
    ]
    measured_revolution = _unwrapped_azimuth_change_deg(revolution)
    upright_2_tilt = _angle_deg(
        upright_hold_2[-1]["capsule_axis_world"],
        [0.0, 0.0, 1.0],
        unsigned_axis=True,
    )
    final_side_tilt = _angle_deg(
        side_hold[-1]["capsule_axis_world"],
        [0.0, 0.0, 1.0],
        unsigned_axis=True,
    )

    roll_end = final_hold[-1]
    displacement = (
        np.asarray(roll_end["position_m"])
        - np.asarray(roll_start["position_m"])
    )
    x_travel = float(displacement[0])
    transverse_drift = float(np.linalg.norm(displacement[1:2]))
    roll_angle = float(
        roll_end["roll_angle_rad"] - roll_start["roll_angle_rad"]
    )
    rolling_arc = CAPSULE_RADIUS_M * abs(roll_angle)
    slip_ratio = abs(abs(x_travel) - rolling_arc) / max(
        abs(x_travel), rolling_arc, 1.0e-6
    )
    dynamic_records = (
        side_to_upright
        + tilt_out
        + revolution
        + tilt_back
        + upright_to_side
        + roll
        + final_hold
    )

    checks = {
        "finite_field_inverse": max(
            item["direction_error_deg"] for item in plan_diagnostics
        )
        < 8.0,
        "starts_side_lying": initial_side_tilt >= 70.0,
        "side_to_upright_completed": upright_1_tilt <= 20.0,
        "45deg_tilt_maintained": (
            30.0 <= float(np.median(revolution_tilts)) <= 60.0
        ),
        "camera_head_points_up_during_revolution": (
            float(np.median(revolution_head_elevations)) >= 20.0
        ),
        "full_azimuth_revolution_completed": abs(measured_revolution) >= 300.0,
        "returns_upright": upright_2_tilt <= 20.0,
        "returns_side_lying": final_side_tilt >= 70.0,
        "roll_positive_world_x": x_travel >= 0.080,
        "roll_distance_near_100mm": abs(x_travel - 0.100) <= 0.015,
        "roll_tracks_world_x": transverse_drift <= 0.020,
        "roll_not_pure_sliding": slip_ratio <= 0.65,
        "contact_retained": _contact_ratio(dynamic_records) >= 0.80,
        "no_launch": max(
            record["ground_gap_m"] for record in dynamic_records
        )
        < 0.004,
        "asm_clearance_safe": min(
            record["asm_clearance_m"] for record in dynamic_records
        )
        >= 0.0,
        "no_early_termination": not any(
            record["terminated"] for record in dynamic_records
        ),
    }
    return h.save_summary(
        checks,
        {
            "requested_sequence": [
                "side",
                "upright",
                "45deg_full_azimuth_revolution",
                "upright",
                "side",
                "roll_world_positive_x_100mm",
            ],
            "initial_side_tilt_deg": initial_side_tilt,
            "first_upright_tilt_deg": upright_1_tilt,
            "revolution_median_tilt_deg": float(
                np.median(revolution_tilts)
            ),
            "revolution_tilt_range_deg": [
                min(revolution_tilts),
                max(revolution_tilts),
            ],
            "requested_rotation_axis_to_capsule_deg": 45.0,
            "requested_capsule_axis_elevation_deg": (
                AZIMUTH_AXIS_ELEVATION_DEG
            ),
            "requested_camera_head_elevation_deg": (
                AZIMUTH_HEAD_ELEVATION_DEG
            ),
            "revolution_camera_head_elevation_median_deg": float(
                np.median(revolution_head_elevations)
            ),
            "measured_angular_velocity_axis_to_capsule_median_deg": (
                float(np.median(revolution_omega_axis_angles))
                if revolution_omega_axis_angles
                else 0.0
            ),
            "measured_azimuth_revolution_deg": measured_revolution,
            "second_upright_tilt_deg": upright_2_tilt,
            "final_side_tilt_deg": final_side_tilt,
            "requested_magnet_displacement_m": requested.tolist(),
            "predicted_magnet_displacement_m": (
                gradient.predicted_displacement_world_m.tolist()
            ),
            "joint_delta_rad": gradient.joint_delta_rad.tolist(),
            "capsule_displacement_world_m": displacement.tolist(),
            "capsule_x_travel_m": x_travel,
            "transverse_drift_m": transverse_drift,
            "integrated_long_axis_roll_rad": roll_angle,
            "rolling_arc_length_m": rolling_arc,
            "slip_ratio": slip_ratio,
            "contact_ratio": _contact_ratio(dynamic_records),
            "minimum_asm_clearance_m": min(
                record["asm_clearance_m"] for record in dynamic_records
            ),
            "field_inverse_max_error_deg": max(
                item["direction_error_deg"] for item in plan_diagnostics
            ),
        },
    )


SCENARIO_RUNNERS = {
    "baseline": run_baseline,
    "field_scan": run_field_scan,
    "tilt_azimuth": run_tilt_azimuth,
    "upright_to_side": run_upright_to_side,
    "long_axis_roll": run_long_axis_roll,
    "composite_motion": run_composite_motion,
}


def run_cli(
    scenario: str,
    *,
    default_task: str = DEFAULT_TASK,
    result_root: Path = RESULT_ROOT,
    environment_label: str = "TABLE",
    dry_surface: bool = True,
) -> None:
    """Launch one passive-capsule motion scenario."""
    if scenario not in SCENARIO_RUNNERS:
        raise ValueError(f"Unknown table scenario: {scenario}")

    parser = argparse.ArgumentParser(
        description=f"Table benchmark: {SCENARIO_LABELS[scenario]}"
    )
    parser.add_argument("--task", default=default_task)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--capsule_camera_view",
        action="store_true",
        help="Open the separate circular capsule-camera window.",
    )
    parser.add_argument(
        "--capsule_pose_view",
        action="store_true",
        help="Open a 30 Hz world-up external follow view of the capsule.",
    )
    parser.add_argument(
        "--contact_debug",
        action="store_true",
        help="Show the 2 mm contact marker.",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace visible runs close to simulated time.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args = parser.parse_args()
    if args.num_envs != 1:
        parser.error("Table acceptance scripts require --num_envs 1")
    args.enable_cameras = True
    if (args.capsule_camera_view or args.capsule_pose_view) and _is_headless(args):
        parser.error("Capsule views require the Kit visualizer")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import torch

        import isaaclab_tasks  # noqa: F401
        import robotarm_magnetic_lab.tasks  # noqa: F401

        from isaaclab.app import launch_simulation
        from isaaclab_tasks.utils import parse_env_cfg
        from robotarm_magnetic_lab.ui import (
            attach_capsule_camera_policy_view,
            attach_capsule_pose_view,
            configure_capsule_camera_view,
            configure_capsule_pose_view,
        )

        torch.manual_seed(42)
        env_cfg = parse_env_cfg(
            args.task, device=args.device, num_envs=1, use_fabric=True
        )
        env_cfg.episode_length_s = 180.0
        if scenario == "composite_motion":
            # The composite 100 mm roll needs more arm workspace than the
            # individual 55 mm acceptance test. This changes only this script's
            # environment instance and leaves the task/training interface
            # untouched.
            env_cfg.actions.joint_position.scale[
                "j[1-6]"
            ] = COMPOSITE_ARM_ACTION_SCALE_RAD
            # ballxj must span both magnetic hemispheres for a continuous
            # 30-degree-cone revolution.  The joints are continuous and their
            # configured 0.8 rad/s velocity limit remains unchanged.
            env_cfg.actions.joint_position.scale[
                "ball.*j"
            ] = COMPOSITE_BALL_ACTION_SCALE_RAD
        if _is_headless(args):
            # Acceptance metrics do not consume images. Removing the 720p
            # camera sensor keeps the physics-only verification several times
            # faster without changing visible runs or the registered task.
            env_cfg.scene.capsule_camera = None
            env_cfg.observations.vision = None
        env_cfg.scene.capsule_contact.debug_vis = args.contact_debug
        if args.contact_debug:
            env_cfg.scene.capsule_contact.visualizer_cfg.markers[
                "contact"
            ].radius = 0.002
        if args.capsule_camera_view:
            configure_capsule_camera_view(env_cfg)
        if args.capsule_pose_view:
            configure_capsule_pose_view(env_cfg)

        with launch_simulation(env_cfg, args):
            env = gym.make(args.task, cfg=env_cfg)
            camera_view = (
                attach_capsule_camera_policy_view(env)
                if args.capsule_camera_view
                else None
            )
            pose_view = (
                attach_capsule_pose_view(env)
                if args.capsule_pose_view
                else None
            )
            harness = TableTestHarness(
                env,
                args,
                scenario,
                result_root=result_root,
                environment_label=environment_label,
                dry_surface=dry_surface,
            )
            try:
                SCENARIO_RUNNERS[scenario](harness)
            finally:
                harness.close()
                if camera_view is not None:
                    camera_view.close()
                if pose_view is not None:
                    pose_view.close()
                env.close()
    finally:
        simulation_app.close()
