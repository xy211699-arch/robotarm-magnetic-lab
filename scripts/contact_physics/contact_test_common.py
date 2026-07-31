"""Shared visual/contact-physics acceptance harness for the stomach task.

Each small ``test_*.py`` file in this directory selects one scenario.  This
module owns Isaac Lab startup, passive-capsule state injection, magnetic-force
gating, contact telemetry, JSONL logging and objective PASS/FAIL checks.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
import json
import math
from pathlib import Path
import os
import time
from typing import Any

from isaaclab.app import AppLauncher


PROJECT_DIR = Path("/mnt/isaac-linux/robotarm_magnetic_lab")
DEFAULT_TASK = "Template-Robotarm-Magnetic-Stomach-Lab-v0"
RESULT_ROOT = Path(
    os.environ.get(
        "ROBOTARM_CONTACT_RESULT_ROOT",
        str(PROJECT_DIR / "logs" / "contact_physics"),
    )
)
CONTACT_THRESHOLD_N = 1.0e-4
BALL_ACTION_INDICES = (6, 7, 8)
BASE_CONTACT_POS = (1.0608155, 0.1145374, 0.0065)

SCENARIO_LABELS = {
    "resting": "No-magnet resting",
    "drop": "Drops from multiple heights",
    "incline_slide": "Free sliding on inclined stomach regions",
    "magnetic_attraction": "Magnetic attraction to stomach wall",
    "wall_roll_turn": "Magnetic rolling and turning along stomach wall",
    "ball_pose_contact": "Ball-pose changes while capsule remains wall-contacting",
    "multi_start": "Repeated stomach contact from multiple initial points",
}


def _is_headless(args) -> bool:
    value = getattr(args, "visualizer", None)
    if isinstance(value, (list, tuple)):
        return any(str(item).lower() == "none" for item in value)
    return str(value).lower() == "none"


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _norm(vector) -> float:
    return float(vector.norm().item())


def _tolist(tensor, digits: int = 7) -> list[float]:
    return [round(float(value), digits) for value in tensor.detach().cpu().reshape(-1)]


def _local_z_axis_world(quaternion_xyzw):
    import torch

    x, y, z, w = quaternion_xyzw.unbind(-1)
    return torch.stack(
        (
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    )


class ContactTestHarness:
    """One-environment deterministic contact test driver."""

    def __init__(self, env, args, scenario: str):
        import torch

        self.env = env
        self.base_env = env.unwrapped
        self.scene = self.base_env.scene
        self.robot = self.scene["robot"]
        self.capsule = self.scene["capsule"]
        self.contact = self.scene["capsule_contact"]
        self.bridge = self.base_env.event_manager.get_term_cfg(
            "magnetic_collision_bridge"
        ).func
        self.args = args
        self.scenario = scenario
        self.step_dt = float(self.base_env.step_dt)
        self.device = self.base_env.device
        self.action = torch.zeros(env.action_space.shape, device=self.device)
        self.ball_indices = [
            self.robot.data.joint_names.index(name)
            for name in ("ballxj", "ballyj", "ballzj")
        ]
        self.records: list[dict[str, Any]] = []
        self.trials: list[dict[str, Any]] = []
        self.global_step = 0
        self._last_velocity = None
        self._trial_start_position = None
        self._magnetic_enabled = True
        self._observations = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = RESULT_ROOT / scenario / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "telemetry.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self._jsonl = self.jsonl_path.open("w", encoding="utf-8")

        masses = self.capsule.root_physx_view.get_masses().numpy().reshape(-1)
        self.mass_kg = float(masses[0])
        self.gravity = torch.tensor(
            self.base_env.sim.cfg.gravity, device=self.device, dtype=torch.float32
        )

    def close(self) -> None:
        self._jsonl.close()

    def set_magnetic_forces(self, enabled: bool) -> None:
        """Gate analytical magnetic force without changing gravity/contact."""
        self._magnetic_enabled = enabled
        self.bridge.config["simulation"]["apply_forces"] = enabled
        if not enabled:
            self.bridge._filtered_wrench.zero_()
            self.bridge.robot.permanent_wrench_composer.reset()
            self.bridge.capsule.permanent_wrench_composer.reset()

    def reset_trial(
        self,
        name: str,
        position: tuple[float, float, float] | None = None,
        magnetic: bool = False,
    ) -> int:
        """Reset the whole scene, then place the passive capsule if requested."""
        import torch

        self._observations, _ = self.env.reset()
        self.action.zero_()
        self.set_magnetic_forces(magnetic)
        if position is not None:
            pose = self.capsule.data.root_pose_w.torch.clone()
            pose[:, :3] = torch.tensor(position, device=self.device)
            self.capsule.write_root_pose_to_sim_index(root_pose=pose)
            self.capsule.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=self.device)
            )
        self._last_velocity = None
        self._trial_start_position = (
            self.capsule.data.root_pos_w.torch[0].detach().clone()
        )
        trial_id = len(self.trials)
        self.trials.append(
            {
                "trial_id": trial_id,
                "name": name,
                "magnetic_initially_enabled": magnetic,
                "start_position_m": _tolist(self._trial_start_position),
                "record_start": len(self.records),
            }
        )
        print(
            f"[CONTACT_TEST] scenario={self.scenario} trial={trial_id} "
            f"name={name} start={_tolist(self._trial_start_position)} "
            f"magnetic={magnetic}",
            flush=True,
        )
        return trial_id

    def set_capsule_position(self, position: tuple[float, float, float]) -> None:
        import torch

        pose = self.capsule.data.root_pose_w.torch.clone()
        pose[:, :3] = torch.tensor(position, device=self.device)
        self.capsule.write_root_pose_to_sim_index(root_pose=pose)
        self.capsule.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=self.device)
        )
        self._last_velocity = None
        self._trial_start_position = pose[0, :3].detach().clone()

    def set_ball_action(self, values: tuple[float, float, float]) -> None:
        for index, value in zip(BALL_ACTION_INDICES, values, strict=True):
            self.action[:, index] = value

    def run_phase(
        self,
        trial_id: int,
        phase: str,
        duration_s: float,
        action_profile: Callable[[float], tuple[float, float, float]] | None = None,
        on_start: Callable[[], None] | None = None,
    ) -> list[dict[str, Any]]:
        if on_start is not None:
            on_start()
        steps = max(int(round(duration_s / self.step_dt)), 1)
        phase_records = []
        for local_step in range(steps):
            fraction = local_step / max(steps - 1, 1)
            if action_profile is not None:
                self.set_ball_action(action_profile(fraction))
            started = time.perf_counter()
            with __import__("torch").inference_mode():
                self._observations, _, terminated, truncated, _ = self.env.step(
                    self.action
                )
            record = self._capture(trial_id, phase, local_step, terminated, truncated)
            self.records.append(record)
            phase_records.append(record)
            self._jsonl.write(json.dumps(record, separators=(",", ":")) + "\n")
            if self.global_step % self.args.log_every == 0:
                print(
                    f"[CONTACT] scenario={self.scenario} trial={trial_id} "
                    f"phase={phase} t={record['sim_time_s']:.2f}s "
                    f"pos={record['position_m']} speed={record['linear_speed_mps']:.4f} "
                    f"normal_N={record['normal_force_norm_N']:.5f} "
                    f"friction_est_N={record['friction_estimate_norm_N']:.5f} "
                    f"contact={record['in_contact']} "
                    f"mag_N={record['magnetic_force_norm_N']:.5f}",
                    flush=True,
                )
            self.global_step += 1
            if self.args.realtime and not _is_headless(self.args):
                remaining = self.step_dt - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
        return phase_records

    def _capture(self, trial_id, phase, local_step, terminated, truncated):
        import torch

        position = self.capsule.data.root_pos_w.torch[0]
        quaternion = self.capsule.data.root_quat_w.torch[0]
        velocity = self.capsule.data.root_lin_vel_w.torch[0]
        angular_velocity = self.capsule.data.root_ang_vel_w.torch[0]
        normal_force = self.contact.data.net_forces_w.torch[0, 0]
        contact_norm = _norm(normal_force)
        normal_force_history = self.contact.data.net_forces_w_history.torch[0, :, 0]
        history_norms = torch.linalg.vector_norm(normal_force_history, dim=-1)
        history_peak_normal_force = float(history_norms.max().item())
        current_contact_time = float(
            self.contact.data.current_contact_time.torch[0, 0].item()
        )
        current_air_time = float(self.contact.data.current_air_time.torch[0, 0].item())

        wrench = getattr(self.base_env, "_legacy_bridge_state", {}).get("wrench")
        if wrench is None:
            magnetic_force = torch.zeros(3, device=self.device)
            magnetic_torque = torch.zeros(3, device=self.device)
        else:
            magnetic_force = wrench[0, 6:9]
            magnetic_torque = wrench[0, 9:12]
            if not self._magnetic_enabled:
                magnetic_force = torch.zeros_like(magnetic_force)
                magnetic_torque = torch.zeros_like(magnetic_torque)

        if self._last_velocity is None:
            acceleration = torch.zeros(3, device=self.device)
        else:
            acceleration = (velocity - self._last_velocity) / self.step_dt
        self._last_velocity = velocity.detach().clone()
        gravity_force = self.mass_kg * self.gravity
        # Newton's second law: contact total = ma - gravity - magnetic.
        # The sensor supplies only the normal component. Their residual is an
        # effective tangential/friction force estimate at policy rate.
        total_contact_estimate = (
            self.mass_kg * acceleration - gravity_force - magnetic_force
        )
        friction_estimate = total_contact_estimate - normal_force
        if contact_norm < CONTACT_THRESHOLD_N:
            friction_estimate = torch.zeros_like(friction_estimate)
        axis_world = _local_z_axis_world(quaternion)
        displacement = position - self._trial_start_position
        ball_pos = self.robot.data.joint_pos.torch[0, self.ball_indices]

        return {
            "scenario": self.scenario,
            "trial_id": trial_id,
            "phase": phase,
            "step": self.global_step,
            "phase_step": local_step,
            "sim_time_s": round(self.global_step * self.step_dt, 7),
            "position_m": _tolist(position),
            "quaternion_xyzw": _tolist(quaternion),
            "axis_world": _tolist(axis_world),
            "linear_velocity_mps": _tolist(velocity),
            "angular_velocity_radps": _tolist(angular_velocity),
            "linear_speed_mps": _norm(velocity),
            "angular_speed_radps": _norm(angular_velocity),
            "displacement_m": _tolist(displacement),
            "normal_force_N": _tolist(normal_force),
            "normal_force_norm_N": contact_norm,
            "normal_force_history_peak_N": history_peak_normal_force,
            "friction_estimate_N": _tolist(friction_estimate),
            "friction_estimate_norm_N": _norm(friction_estimate),
            "magnetic_force_N": _tolist(magnetic_force),
            "magnetic_force_norm_N": _norm(magnetic_force),
            "magnetic_torque_Nm": _tolist(magnetic_torque),
            "magnetic_torque_norm_Nm": _norm(magnetic_torque),
            "ball_joint_pos_rad": _tolist(ball_pos),
            "ball_action": _tolist(self.action[0, list(BALL_ACTION_INDICES)]),
            "in_contact": contact_norm >= CONTACT_THRESHOLD_N,
            "contact_event_in_policy_step": (
                history_peak_normal_force >= CONTACT_THRESHOLD_N
            ),
            "current_contact_time_s": current_contact_time,
            "current_air_time_s": current_air_time,
            "magnetic_enabled": self._magnetic_enabled,
            "terminated": bool(terminated[0].item()),
            "truncated": bool(truncated[0].item()),
        }

    def finish_trial(self, trial_id: int) -> list[dict[str, Any]]:
        trial = self.trials[trial_id]
        trial["record_end"] = len(self.records)
        return self.records[trial["record_start"] : trial["record_end"]]

    def save_summary(self, checks: dict[str, bool], metrics: dict[str, Any]) -> dict:
        passed = all(checks.values())
        summary = {
            "schema_version": "1.0.0",
            "scenario": self.scenario,
            "label": SCENARIO_LABELS[self.scenario],
            "passed": passed,
            "checks": checks,
            "metrics": metrics,
            "mass_kg": self.mass_kg,
            "contact_threshold_N": CONTACT_THRESHOLD_N,
            "physics_dt_s": float(self.base_env.physics_dt),
            "policy_dt_s": self.step_dt,
            "telemetry_path": str(self.jsonl_path),
            "trials": self.trials,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[CONTACT_TEST_RESULT] scenario={self.scenario} "
            f"result={'PASS' if passed else 'FAIL'} "
            f"checks={json.dumps(checks)} summary={self.summary_path}",
            flush=True,
        )
        return summary


def _contact_ratio(records) -> float:
    return sum(record["in_contact"] for record in records) / max(len(records), 1)


def _distance(first, last, axes=(0, 1, 2)) -> float:
    return math.sqrt(
        sum(
            (last["position_m"][axis] - first["position_m"][axis]) ** 2
            for axis in axes
        )
    )


def run_resting(h: ContactTestHarness) -> dict:
    trial = h.reset_trial("rest_on_lower_wall", magnetic=False)
    records = h.run_phase(trial, "rest", 8.0)
    h.finish_trial(trial)
    tail = records[-max(int(3.0 / h.step_dt), 1) :]
    drift = _distance(records[0], records[-1])
    checks = {
        "contact_detected": any(r["in_contact"] for r in records),
        "contact_maintained_last_3s": _contact_ratio(tail) >= 0.80,
        "settled_linear_speed": max(r["linear_speed_mps"] for r in tail[-10:]) < 0.02,
        "bounded_drift": drift < 0.015,
        "magnetic_force_disabled": max(r["magnetic_force_norm_N"] for r in records) < 1.0e-8,
    }
    return h.save_summary(
        checks,
        {
            "contact_ratio_last_3s": _contact_ratio(tail),
            "drift_m": drift,
            "final_speed_mps": records[-1]["linear_speed_mps"],
            "mean_support_force_N": sum(r["normal_force_norm_N"] for r in tail)
            / len(tail),
        },
    )


def run_drop(h: ContactTestHarness) -> dict:
    outcomes = []
    base = BASE_CONTACT_POS
    for height_mm in (5.0, 15.0, 30.0):
        start = (base[0], base[1], base[2] + height_mm * 1.0e-3)
        trial = h.reset_trial(f"drop_{height_mm:.0f}mm", start, magnetic=False)
        records = h.run_phase(trial, "free_fall_and_settle", 5.0)
        h.finish_trial(trial)
        contact_indices = [
            i
            for i, record in enumerate(records)
            if record["contact_event_in_policy_step"]
        ]
        tail = records[-20:]
        outcomes.append(
            {
                "height_mm": height_mm,
                "contact": bool(contact_indices),
                "contact_time_s": (
                    contact_indices[0] * h.step_dt if contact_indices else None
                ),
                "peak_normal_force_N": max(
                    record["normal_force_history_peak_N"] for record in records
                ),
                "final_speed_mps": records[-1]["linear_speed_mps"],
                "tail_contact_ratio": _contact_ratio(tail),
                "minimum_z_m": min(record["position_m"][2] for record in records),
            }
        )
    checks = {
        "all_heights_contact": all(item["contact"] for item in outcomes),
        "all_heights_settle": all(item["final_speed_mps"] < 0.04 for item in outcomes),
        "contact_retained": all(item["tail_contact_ratio"] >= 0.70 for item in outcomes),
        "no_tunneling_below_wall": all(item["minimum_z_m"] > -0.010 for item in outcomes),
        "bounded_impact_force": all(
            math.isfinite(item["peak_normal_force_N"])
            and item["peak_normal_force_N"] < 50.0
            for item in outcomes
        ),
    }
    return h.save_summary(checks, {"drops": outcomes})


def run_incline_slide(h: ContactTestHarness) -> dict:
    base = BASE_CONTACT_POS
    starts = (
        ("region_left", (base[0] - 0.025, base[1], base[2] + 0.020)),
        ("region_forward", (base[0], base[1] + 0.025, base[2] + 0.020)),
        ("region_diagonal", (base[0] + 0.022, base[1] - 0.018, base[2] + 0.020)),
    )
    outcomes = []
    for name, start in starts:
        trial = h.reset_trial(name, start, magnetic=False)
        records = h.run_phase(trial, "drop_slide_settle", 7.0)
        h.finish_trial(trial)
        contact_records = [record for record in records if record["in_contact"]]
        displacement = _distance(records[0], records[-1], axes=(0, 1))
        outcomes.append(
            {
                "name": name,
                "contact": bool(contact_records),
                "contact_ratio": _contact_ratio(records),
                "planar_displacement_m": displacement,
                "final_speed_mps": records[-1]["linear_speed_mps"],
                "peak_friction_estimate_N": max(
                    record["friction_estimate_norm_N"] for record in records
                ),
            }
        )
    checks = {
        "all_regions_contact": all(item["contact"] for item in outcomes),
        "all_regions_remain_bounded": all(
            item["final_speed_mps"] < 0.08 for item in outcomes
        ),
        "sliding_observed": sum(
            item["planar_displacement_m"] >= 0.001 for item in outcomes
        )
        >= 2,
        "friction_response_finite": all(
            math.isfinite(item["peak_friction_estimate_N"]) for item in outcomes
        ),
    }
    return h.save_summary(checks, {"regions": outcomes})


def run_magnetic_attraction(h: ContactTestHarness) -> dict:
    base = BASE_CONTACT_POS
    # A 50 ms policy step already produces about 12 mm of free-fall travel.
    # Use two identical 50 mm starts so the reference does not reach the wall
    # before magnetic force is enabled.
    start = (base[0], base[1], base[2] + 0.050)
    reference_trial = h.reset_trial(
        "gravity_reference_from_50mm", start, magnetic=False
    )
    off = h.run_phase(reference_trial, "magnet_off_reference", h.step_dt)
    h.finish_trial(reference_trial)
    magnetic_trial = h.reset_trial(
        "magnetic_attraction_from_50mm", start, magnetic=True
    )
    on = h.run_phase(magnetic_trial, "magnet_on_attraction", 4.0)
    h.finish_trial(magnetic_trial)
    contact_indices = [i for i, record in enumerate(on) if record["in_contact"]]
    vertical_travel = start[2] - on[-1]["position_m"][2]
    tail = on[-20:]
    mean_tail_support = sum(r["normal_force_norm_N"] for r in tail) / len(tail)
    mean_tail_magnetic_z = sum(r["magnetic_force_N"][2] for r in tail) / len(tail)
    checks = {
        "off_wrench_zero": max(r["magnetic_force_norm_N"] for r in off) < 1.0e-8,
        "magnetic_wrench_nonzero": max(r["magnetic_force_norm_N"] for r in on) > 1.0e-4,
        "magnetic_force_points_to_lower_wall": mean_tail_magnetic_z < -1.0e-4,
        "moves_toward_lower_wall": vertical_travel > 0.003,
        "wall_contact_reached": bool(contact_indices),
        "contact_then_retained": _contact_ratio(on[-20:]) >= 0.60,
        "magnetic_load_added_to_support": mean_tail_support > h.mass_kg * 9.81 * 1.03,
    }
    return h.save_summary(
        checks,
        {
            "vertical_travel_m": vertical_travel,
            "contact_time_after_enable_s": (
                contact_indices[0] * h.step_dt if contact_indices else None
            ),
            "peak_magnetic_force_N": max(r["magnetic_force_norm_N"] for r in on),
            "peak_contact_force_N": max(r["normal_force_norm_N"] for r in on),
            "mean_tail_support_force_N": mean_tail_support,
            "capsule_weight_N": h.mass_kg * 9.81,
            "mean_tail_magnetic_z_N": mean_tail_magnetic_z,
        },
    )


def run_wall_roll_turn(h: ContactTestHarness) -> dict:
    trial = h.reset_trial("roll_and_turn", BASE_CONTACT_POS, magnetic=True)
    settle = h.run_phase(trial, "settle", 1.5)
    tilt = h.run_phase(
        trial,
        "tilt_for_traction",
        2.0,
        lambda u: (-0.55 * _smoothstep(u), 0.0, 0.0),
    )
    roll = h.run_phase(
        trial,
        "roll",
        4.0,
        lambda u: (-0.55, 0.12 * math.sin(2.0 * math.pi * u), 0.55 * u),
    )
    turn = h.run_phase(
        trial,
        "turn",
        4.0,
        lambda u: (
            -0.55,
            0.25 * math.sin(math.pi * u),
            0.55 - 1.10 * _smoothstep(u),
        ),
    )
    all_records = settle + tilt + roll + turn
    h.finish_trial(trial)
    planar = _distance(settle[-1], all_records[-1], axes=(0, 1))
    start_axis = settle[-1]["axis_world"]
    end_axis = all_records[-1]["axis_world"]
    dot = sum(a * b for a, b in zip(start_axis, end_axis, strict=True))
    axis_change_deg = math.degrees(math.acos(min(max(dot, -1.0), 1.0)))
    active = tilt + roll + turn
    checks = {
        "magnetic_wrench_present": max(r["magnetic_force_norm_N"] for r in active) > 1.0e-4,
        "wall_contact_mostly_retained": _contact_ratio(active) >= 0.65,
        "translation_observed": planar >= 0.001,
        "rolling_observed": max(r["angular_speed_radps"] for r in active) >= 0.05,
        "turning_observed": axis_change_deg >= 5.0,
        "no_capsule_launch": max(r["linear_speed_mps"] for r in active) < 0.30,
    }
    return h.save_summary(
        checks,
        {
            "contact_ratio": _contact_ratio(active),
            "planar_displacement_m": planar,
            "axis_change_deg": axis_change_deg,
            "peak_linear_speed_mps": max(r["linear_speed_mps"] for r in active),
            "peak_angular_speed_radps": max(r["angular_speed_radps"] for r in active),
        },
    )


def run_ball_pose_contact(h: ContactTestHarness) -> dict:
    trial = h.reset_trial("ball_pose_sequence", BASE_CONTACT_POS, magnetic=True)
    settle = h.run_phase(trial, "settle", 1.5)
    pose_a = h.run_phase(
        trial,
        "ball_pose_a",
        3.0,
        lambda u: (-0.50 * _smoothstep(u), 0.20 * _smoothstep(u), 0.0),
    )
    pose_b = h.run_phase(
        trial,
        "ball_pose_b",
        3.0,
        lambda u: (-0.50, 0.20 - 0.40 * _smoothstep(u), 0.50 * _smoothstep(u)),
    )
    pose_c = h.run_phase(
        trial,
        "ball_pose_c",
        3.0,
        lambda u: (
            -0.50 * (1.0 - _smoothstep(u)),
            -0.20 * (1.0 - _smoothstep(u)),
            0.50 * (1.0 - _smoothstep(u)),
        ),
    )
    active = pose_a + pose_b + pose_c
    h.finish_trial(trial)
    initial_ball = settle[-1]["ball_joint_pos_rad"]
    final_candidates = [pose_a[-1], pose_b[-1], pose_c[-1]]
    maximum_ball_delta = max(
        math.sqrt(
            sum(
                (record["ball_joint_pos_rad"][i] - initial_ball[i]) ** 2
                for i in range(3)
            )
        )
        for record in final_candidates
    )
    checks = {
        "ball_pose_changes": maximum_ball_delta >= 0.30,
        "wall_contact_retained": _contact_ratio(active) >= 0.65,
        "capsule_responds": max(r["angular_speed_radps"] for r in active) >= 0.03,
        "magnetic_force_active": max(r["magnetic_force_norm_N"] for r in active) > 1.0e-4,
        "motion_bounded": max(r["linear_speed_mps"] for r in active) < 0.30,
    }
    return h.save_summary(
        checks,
        {
            "maximum_ball_joint_delta_rad": maximum_ball_delta,
            "contact_ratio": _contact_ratio(active),
            "peak_capsule_angular_speed_radps": max(
                r["angular_speed_radps"] for r in active
            ),
        },
    )


def run_multi_start(h: ContactTestHarness) -> dict:
    base = BASE_CONTACT_POS
    starts = (
        ("center", (base[0], base[1], base[2] + 0.012)),
        ("x_minus", (base[0] - 0.018, base[1], base[2] + 0.018)),
        ("x_plus", (base[0] + 0.018, base[1], base[2] + 0.018)),
        ("y_minus", (base[0], base[1] - 0.018, base[2] + 0.018)),
        ("y_plus", (base[0], base[1] + 0.018, base[2] + 0.018)),
    )
    outcomes = []
    for name, start in starts:
        trial = h.reset_trial(name, start, magnetic=False)
        records = h.run_phase(trial, "drop_to_local_wall", 5.0)
        h.finish_trial(trial)
        outcomes.append(
            {
                "name": name,
                "contact": any(r["in_contact"] for r in records),
                "tail_contact_ratio": _contact_ratio(records[-20:]),
                "final_position_m": records[-1]["position_m"],
                "final_speed_mps": records[-1]["linear_speed_mps"],
                "finite": all(
                    math.isfinite(value)
                    for record in records
                    for value in (
                        record["linear_speed_mps"],
                        record["normal_force_norm_N"],
                    )
                ),
            }
        )
    checks = {
        "all_points_contact": all(item["contact"] for item in outcomes),
        "all_points_contact_retained": all(
            item["tail_contact_ratio"] >= 0.60 for item in outcomes
        ),
        "all_points_finite": all(item["finite"] for item in outcomes),
        "all_points_bounded": all(
            item["final_speed_mps"] < 0.08 for item in outcomes
        ),
    }
    return h.save_summary(checks, {"initial_points": outcomes})


SCENARIO_RUNNERS = {
    "resting": run_resting,
    "drop": run_drop,
    "incline_slide": run_incline_slide,
    "magnetic_attraction": run_magnetic_attraction,
    "wall_roll_turn": run_wall_roll_turn,
    "ball_pose_contact": run_ball_pose_contact,
    "multi_start": run_multi_start,
}


def run_cli(scenario: str) -> None:
    """Launch one visual scenario selected by a tiny entry-point script."""
    if scenario not in SCENARIO_RUNNERS:
        raise ValueError(f"Unknown scenario: {scenario}")

    parser = argparse.ArgumentParser(
        description=f"Visual stomach contact test: {SCENARIO_LABELS[scenario]}"
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
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
        help=(
            "Show a 2 mm contact marker in the main viewport. Disabled by "
            "default so the marker cannot hide the 13 mm capsule."
        ),
    )
    parser.add_argument(
        "--no_contact_debug",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace GUI runs close to policy time for visual inspection.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args = parser.parse_args()
    if args.num_envs != 1:
        parser.error("Contact acceptance scripts currently require --num_envs 1")
    args.enable_cameras = True
    if (args.capsule_camera_view or args.capsule_pose_view) and _is_headless(args):
        parser.error(
            "Capsule views cannot be combined with --visualizer none"
        )

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
        env_cfg.scene.capsule_contact.debug_vis = (
            args.contact_debug and not args.no_contact_debug
        )
        if env_cfg.scene.capsule_contact.debug_vis:
            # Isaac Lab's default contact marker is a red sphere with 20 mm
            # radius.  That is larger than this project's 13 mm diameter,
            # 25 mm long capsule and visually replaces it.  Keep contact
            # telemetry unchanged, but use a small optional viewport marker.
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
            harness = ContactTestHarness(env, args, scenario)
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
