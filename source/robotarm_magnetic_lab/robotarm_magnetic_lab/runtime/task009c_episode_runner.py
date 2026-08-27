"""Strict single-environment synchronous episode protocol for TASK-009C."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from robotarm_magnetic_lab.baselines.random_policies import PolicyAction
from robotarm_magnetic_lab.coverage.entry_pose_library import file_sha256
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.task009b_training_env import (
    _stable_rgb_digest,
)


CONTROL_DT_S = 0.1
PHYSICS_SUBSTEPS = 24
EPISODE_RECORD_SCHEMA = "robotarm_magnetic_lab.task009c_episode_boundaries"


class EpisodeProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    kind: str
    policy_id: str
    pose_id: str
    environment_seed: int
    policy_seed: int
    duration_s: float
    action_cycles: int

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "EpisodeSpec":
        spec = cls(
            episode_id=str(record["episode_id"]),
            kind=str(record["kind"]),
            policy_id=str(record["policy_id"]),
            pose_id=str(record["pose_id"]),
            environment_seed=int(record["environment_seed"]),
            policy_seed=int(record["policy_seed"]),
            duration_s=float(record["duration_s"]),
            action_cycles=int(record["action_cycles"]),
        )
        if spec.kind not in ("smoke", "formal"):
            raise ValueError("episode kind must be smoke or formal")
        if spec.action_cycles <= 0 or not math.isclose(
            spec.duration_s, spec.action_cycles * CONTROL_DT_S, abs_tol=1.0e-12
        ):
            raise ValueError("episode duration and 10 Hz action count are inconsistent")
        return spec


def _root_state(base) -> tuple[np.ndarray, np.ndarray]:
    capsule = base.scene["capsule"]
    pose = capsule.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    velocity = capsule.data.root_com_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
    return pose, velocity


def _boundary_row(
    *,
    config_sha256: str,
    run_id: str,
    spec: EpisodeSpec,
    boundary_index: int,
    action: PolicyAction | None,
    physics_substeps: int,
    camera_frame: int,
    rgb_digest: str,
    evaluator,
    base,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, Any]:
    pose, velocity = _root_state(base)
    telemetry = None
    if action is not None:
        trace = base.action_manager.get_term("parameterized_force").current_cycle_trace
        telemetry = trace[-1] if trace else None
    coverage = evaluator.latest_record
    if coverage is None:
        raise EpisodeProtocolError("coverage record is missing at a control boundary")
    finite = bool(
        np.isfinite(pose).all()
        and np.isfinite(velocity).all()
        and math.isfinite(float(coverage["reachable_coverage_fraction"]))
        and math.isfinite(float(coverage["raw_coverage_fraction"]))
    )
    return {
        "schema": EPISODE_RECORD_SCHEMA,
        "task_version": 1,
        "config_sha256": config_sha256,
        "run_id": run_id,
        "kind": spec.kind,
        "episode_id": spec.episode_id,
        "policy_id": spec.policy_id,
        "pose_id": spec.pose_id,
        "environment_seed": spec.environment_seed,
        "policy_seed": spec.policy_seed,
        "boundary_index": int(boundary_index),
        "sim_time_s": float(boundary_index) * CONTROL_DT_S,
        "mode_id": None if action is None else int(action.mode),
        "mode_name": "C0" if action is None else action.mode.name,
        "alpha": None if action is None else float(action.alpha),
        "force_ratio_mg": 0.0 if telemetry is None else float(telemetry.force_ratio),
        "physics_substeps": int(physics_substeps),
        "actor_rgb_frame": int(camera_frame),
        "coverage_rgb_frame": int(coverage["camera_frame"]),
        "rgb_content_sha256": str(rgb_digest),
        "reachable_current_visible_area_m2": float(coverage["reachable_visible_area_m2"]),
        "reachable_cumulative_coverage_area_m2": float(
            coverage["reachable_cumulative_area_m2"]
        ),
        "reachable_coverage_fraction": float(coverage["reachable_coverage_fraction"]),
        "raw_cumulative_coverage_area_m2": float(coverage["raw_cumulative_area_m2"]),
        "raw_coverage_fraction": float(coverage["raw_coverage_fraction"]),
        "capsule_position_world_m": pose[:3].tolist(),
        "capsule_quaternion_xyzw": pose[3:].tolist(),
        "capsule_linear_velocity_world_m_s": velocity[:3].tolist(),
        "capsule_angular_velocity_world_rad_s": velocity[3:].tolist(),
        "finite": finite,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def validate_episode_records(
    records: Iterable[dict[str, Any]],
    *,
    expected_cycles: int,
    expected_episode_id: str | None = None,
) -> list[dict[str, Any]]:
    """Reject incomplete or misaligned data; never interpolate or repair it."""
    rows = list(records)
    if len(rows) != int(expected_cycles) + 1:
        raise EpisodeProtocolError(
            f"episode requires {expected_cycles + 1} aligned points, found {len(rows)}"
        )
    previous_actor_frame = None
    previous_reachable = -math.inf
    previous_raw = -math.inf
    for index, row in enumerate(rows):
        if row.get("schema") != EPISODE_RECORD_SCHEMA:
            raise EpisodeProtocolError("episode boundary schema mismatch")
        if expected_episode_id is not None and row.get("episode_id") != expected_episode_id:
            raise EpisodeProtocolError("episode ID mismatch")
        if int(row.get("boundary_index", -1)) != index:
            raise EpisodeProtocolError("boundary indices are missing, duplicated, or out of order")
        if not math.isclose(float(row.get("sim_time_s", math.nan)), index * CONTROL_DT_S, abs_tol=1e-9):
            raise EpisodeProtocolError("episode timestamps are not exact 0.1 s boundaries")
        expected_substeps = 0 if index == 0 else PHYSICS_SUBSTEPS
        if int(row.get("physics_substeps", -1)) != expected_substeps:
            raise EpisodeProtocolError("episode boundary has the wrong physics substep count")
        actor_frame = int(row.get("actor_rgb_frame", -1))
        if int(row.get("coverage_rgb_frame", -2)) != actor_frame:
            raise EpisodeProtocolError("Actor and coverage RGB frames differ")
        if previous_actor_frame is not None and actor_frame != previous_actor_frame + 1:
            raise EpisodeProtocolError("Actor RGB frames are not unique consecutive boundaries")
        previous_actor_frame = actor_frame
        reachable = float(row.get("reachable_coverage_fraction", math.nan))
        raw = float(row.get("raw_coverage_fraction", math.nan))
        if not (0.0 <= reachable <= 1.0 and 0.0 <= raw <= 1.0):
            raise EpisodeProtocolError("coverage fraction is non-finite or outside [0,1]")
        if reachable + 1e-15 < previous_reachable or raw + 1e-15 < previous_raw:
            raise EpisodeProtocolError("cumulative coverage decreased")
        previous_reachable, previous_raw = reachable, raw
        if not bool(row.get("finite", False)):
            raise EpisodeProtocolError("episode contains non-finite state")
        if bool(row.get("terminated", False)) or bool(row.get("truncated", False)):
            raise EpisodeProtocolError("episode terminated before external boundary completion")
        if index == 0:
            if row.get("mode_name") != "C0" or row.get("mode_id") is not None:
                raise EpisodeProtocolError("boundary zero must be the post-HOLD C0 point")
        elif row.get("mode_id") is None or row.get("alpha") is None:
            raise EpisodeProtocolError("action boundary is missing mode or alpha")
    return rows


def read_episode_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize_episode(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    validate_episode_records(rows, expected_cycles=len(rows) - 1)
    actions = rows[1:]
    reachable = np.asarray([float(row["reachable_coverage_fraction"]) for row in rows])
    times = np.asarray([float(row["sim_time_s"]) for row in rows])
    raw = np.asarray([float(row["raw_coverage_fraction"]) for row in rows])
    positions = np.asarray([row["capsule_position_world_m"] for row in rows], dtype=np.float64)
    linear = np.asarray([row["capsule_linear_velocity_world_m_s"] for row in rows], dtype=np.float64)
    angular = np.asarray([row["capsule_angular_velocity_world_rad_s"] for row in rows], dtype=np.float64)
    counts = {name: 0 for name in ("HOLD", "MOVE_POS", "MOVE_NEG", "VIEW_POS", "VIEW_NEG", "UP")}
    for row in actions:
        counts[str(row["mode_name"])] += 1
    non_hold_alpha = np.asarray(
        [float(row["alpha"]) for row in actions if row["mode_name"] != "HOLD"], dtype=np.float64
    )
    duration = float(times[-1])
    auc = float(np.trapezoid(reachable, times) / duration)
    no_new = np.diff(reachable) <= 1.0e-15
    return {
        "episode_id": rows[0]["episode_id"],
        "kind": rows[0]["kind"],
        "policy_id": rows[0]["policy_id"],
        "pose_id": rows[0]["pose_id"],
        "environment_seed": rows[0]["environment_seed"],
        "policy_seed": rows[0]["policy_seed"],
        "boundary_count": len(rows),
        "action_cycles": len(actions),
        "physics_substeps": len(actions) * PHYSICS_SUBSTEPS,
        "duration_s": duration,
        "C0_reachable": float(reachable[0]),
        "C_final_reachable": float(reachable[-1]),
        "delta_reachable": float(reachable[-1] - reachable[0]),
        "C0_raw": float(raw[0]),
        "C_final_raw": float(raw[-1]),
        "normalized_reachable_auc": auc,
        "mode_counts": counts,
        "mode_fractions": {name: count / len(actions) for name, count in counts.items()},
        "non_hold_alpha_mean": None if len(non_hold_alpha) == 0 else float(non_hold_alpha.mean()),
        "non_hold_alpha_std": None if len(non_hold_alpha) == 0 else float(non_hold_alpha.std()),
        "no_new_coverage_boundary_fraction": float(no_new.mean()),
        "total_com_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
        "maximum_linear_speed_m_s": float(np.linalg.norm(linear, axis=1).max()),
        "maximum_angular_speed_rad_s": float(np.linalg.norm(angular, axis=1).max()),
        "rgb_frame_unique": len({row["actor_rgb_frame"] for row in rows}) == len(rows),
        "coverage_monotonic": bool(np.all(np.diff(reachable) >= -1.0e-15)),
        "status": "pass",
        "exception": None,
    }


class SynchronousEpisodeRunner:
    """Execute policies without exposing observations or evaluator truth to them."""

    def __init__(self, env, evaluator, *, config_sha256: str, run_id: str) -> None:
        self.env = env
        self.base = env.unwrapped
        self.evaluator = evaluator
        self.config_sha256 = str(config_sha256)
        self.run_id = str(run_id)

    def run(
        self,
        *,
        spec: EpisodeSpec,
        policy,
        initial_observation: dict,
        output_path: str | Path,
    ) -> tuple[Path, dict[str, Any]]:
        if self.base.num_envs != 1:
            raise EpisodeProtocolError("TASK-009C only supports one sequential environment")
        if float(self.base.cfg.episode_length_s) + 1e-12 < spec.duration_s:
            raise EpisodeProtocolError("environment time limit is shorter than external episode")
        if int(self.base.episode_length_buf[0].item()) != 0:
            raise EpisodeProtocolError("formal episode budget was not cleared after reset HOLD")
        policy.reset()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        rows: list[dict[str, Any]] = []
        sync = dict(self.base._task009b_policy_rgb_sync_latest)
        initial_rgb = initial_observation["policy"]["rgb"]
        initial_digest = _stable_rgb_digest(initial_rgb)
        update = self.evaluator.maybe_update(
            expected_camera_frame=int(sync["frame"]), rgb_content_sha256=initial_digest
        )
        if update is None or update.coverage_fraction <= 0.0:
            raise EpisodeProtocolError("post-HOLD frame did not initialize a nonzero C0")
        initial = _boundary_row(
            config_sha256=self.config_sha256,
            run_id=self.run_id,
            spec=spec,
            boundary_index=0,
            action=None,
            physics_substeps=0,
            camera_frame=int(sync["frame"]),
            rgb_digest=initial_digest,
            evaluator=self.evaluator,
            base=self.base,
        )
        rows.append(initial)
        with partial.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(initial, sort_keys=True, separators=(",", ":")) + "\n")
            for boundary in range(1, spec.action_cycles + 1):
                action = policy.act()
                tensor = torch.tensor(
                    [[float(action.mode), float(action.alpha)]],
                    device=self.base.device,
                    dtype=torch.float32,
                )
                observation, _, terminated, truncated, _ = self.env.step(tensor)
                term = bool(torch.any(terminated).item())
                trunc = bool(torch.any(truncated).item())
                trace = self.base.action_manager.get_term("parameterized_force").current_cycle_trace
                sync = dict(self.base._task009b_policy_rgb_sync_latest)
                rgb = observation["policy"]["rgb"]
                digest = _stable_rgb_digest(rgb)
                update = self.evaluator.maybe_update(
                    expected_camera_frame=int(sync["frame"]), rgb_content_sha256=digest
                )
                if update is None:
                    raise EpisodeProtocolError(f"coverage omitted boundary {boundary}")
                row = _boundary_row(
                    config_sha256=self.config_sha256,
                    run_id=self.run_id,
                    spec=spec,
                    boundary_index=boundary,
                    action=action,
                    physics_substeps=len(trace),
                    camera_frame=int(sync["frame"]),
                    rgb_digest=digest,
                    evaluator=self.evaluator,
                    base=self.base,
                    terminated=term,
                    truncated=trunc,
                )
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                rows.append(row)
                if term or trunc:
                    raise EpisodeProtocolError(f"episode ended early at boundary {boundary}")
        validate_episode_records(rows, expected_cycles=spec.action_cycles, expected_episode_id=spec.episode_id)
        os.replace(partial, destination)
        summary = summarize_episode(rows)
        summary["boundary_log_path"] = str(destination.resolve())
        summary["boundary_log_bytes"] = destination.stat().st_size
        summary["boundary_log_sha256"] = file_sha256(destination)
        return destination, summary
