# Local Dynamic Capsule Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build four sub-10-second closed-loop capsule posture primitives using only direct center-of-mass force and torque, validate them quantitatively on the flat table, then run the identical controller in the existing approximately horizontal stomach reset for visual review.

**Architecture:** Pure NumPy modules define command/state types, world-frame trajectories, and the feedback wrench law. One Isaac Lab action term reads live capsule state at 240 Hz and applies the resulting bounded world-frame wrench through the existing permanent wrench composer. Two isolated environment configurations share one action-config factory; the flat task is the quantitative gate and the stomach task is a scene-only rendered migration.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Isaac Lab ManagerBasedRLEnv, Isaac Sim 6.0/PhysX, Gymnasium, pytest, Kit rendering.

**Spec:** `docs/superpowers/specs/2026-08-18-local-dynamics-primitives-design.md`

## Global Constraints

- Base exactly on `06b15caf9a69bc9c20f85522ce4abbb32c8b9245` and work only on `feature/TASK-004-local-dynamics-primitives`.
- Preserve a non-kinematic capsule, gravity, CPU PhysX, scene/body CCD, 240 Hz physics, 60 Hz environment/render interval, and 30 Hz capsule-camera updates.
- Apply only center-of-mass world-frame force and torque; root-state writes are allowed only in the existing reset event before simulation advances.
- Instantiate no joint, robot, magnetic, ideal-surface, legacy-bridge, or TASK-003 open-loop force action in either new task.
- Use `u = R(q)[0,0,-1]`, world `+Z` as upright, world `+X` as the default azimuth, and local `+Z` as the non-camera end.
- Implement no clearance, free-space, sweep, boundary-margin, mesh-distance, surface-normal, tangent-frame, ray-cast, projection, avoidance, or recovery logic.
- Keep every primitive hard timeout strictly below `10.0 s`.
- Use the same `make_local_primitive_action_cfg()` result for flat and stomach tasks; freeze shared parameters after flat acceptance and do no stomach-only tuning.
- Do not modify USD/USDZ assets, physical materials, capsule mass/inertia/geometry, camera calibration, TASK-003 placement, solver/CCD settings, previous tasks, or previous reports.
- Write logs, JSONL, screenshots, and videos outside Git under `/mnt/isaac-linux/robotarm_magnetic_lab/logs/` and report byte sizes and SHA-256 hashes.

---

## File Structure

`controllers/local_primitives/types.py` owns primitive IDs, command/status enums, read-only capsule state, wrench output, and telemetry dataclasses. `config.py` owns all shared gains, limits, durations, tolerances, validation ranges, and `make_local_primitive_action_cfg()` input values. `trajectory.py` owns quintic timing, unit-vector interpolation, posture targets, cone phase, and desired angular velocity. `controller.py` owns the pure 240 Hz state machine, start-posture gates, non-camera XY anchor, wrench feedback, saturation, completion, timeout, and holding behavior.

`mdp/local_primitive_action.py` adapts the pure controller to Isaac Lab, decodes the four-float pulse command, reads live center-of-mass state, applies the bounded wrench, and exposes telemetry. The two new environment configuration files contain only scene-specific inheritance and register the one shared action term.

`scripts/local_primitives/inspect_local_primitives_prerequisites.py` proves the live endpoint, wrench, dynamic-body, timing, and isolation contracts. `validate_local_primitives_flat.py` performs quantitative flat sequences and contact-point endpoint classification. `teleop_local_primitives.py` provides the shared continuous rendered launcher for either scene.

`tests/local_primitives/` contains pure controller/trajectory tests, static source scans, environment isolation/shared-config tests, flat-summary evaluation tests, and launcher command tests. `docs/LOCAL_DYNAMICS_PRIMITIVES.md` is operator documentation and `handoffs/reports/TASK-004-local-dynamics-primitives-report.md` is the authoritative Linux evidence report.

### Task 1: Define Pure Primitive Contracts, Shared Configuration, and Trajectories

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/types.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/config.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/trajectory.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/__init__.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/__init__.py`
- Create: `tests/local_primitives/conftest.py`
- Create: `tests/local_primitives/test_types_and_config.py`
- Create: `tests/local_primitives/test_trajectory.py`

**Interfaces:**
- Produces: `PrimitiveId`, `PrimitiveStatus`, `CapsuleState`, `WrenchCommand`, `PrimitiveTelemetry`, `LocalPrimitiveControllerCfg`, `make_local_primitive_controller_cfg()`, `quintic_scale()`, `posture_axis()`, and `desired_axis_sample()`.
- Consumes: only Python standard library and NumPy; importing these modules must not launch Isaac Sim.

- [ ] **Step 1: Write failing enum and frozen-config tests**

```python
from dataclasses import asdict
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
    LocalPrimitiveControllerCfg,
    PrimitiveId,
    make_local_primitive_controller_cfg,
)


def test_primitive_codes_are_frozen():
    assert {item.name: int(item) for item in PrimitiveId} == {
        "SIDE_TO_UPRIGHT": 0,
        "UPRIGHT_TO_SIDE": 1,
        "UPRIGHT_TO_30_DEG": 2,
        "CONE_30_DEG_ONE_REVOLUTION": 3,
    }


def test_default_configuration_is_exact_and_sub_ten_seconds():
    cfg = make_local_primitive_controller_cfg()
    assert isinstance(cfg, LocalPrimitiveControllerCfg)
    assert cfg.axis_kp_nm_per_rad == pytest.approx(1.2e-5)
    assert cfg.axis_kd_nms_per_rad == pytest.approx(2.0e-6)
    assert cfg.roll_damping_nms_per_rad == pytest.approx(1.0e-6)
    assert cfg.torque_limit_nm == pytest.approx(2.0e-5)
    assert cfg.anchor_kp_n_per_m == pytest.approx(0.8)
    assert cfg.anchor_kd_ns_per_m == pytest.approx(0.03)
    assert cfg.horizontal_force_limit_weight_ratio == pytest.approx(0.5)
    assert cfg.downward_preload_weight_ratio == pytest.approx(0.15)
    assert cfg.motion_duration_s == pytest.approx((5.5, 4.5, 3.5, 8.0))
    assert cfg.hard_timeout_s == pytest.approx((8.0, 7.0, 6.0, 9.5))
    assert max(cfg.hard_timeout_s) < 10.0
    assert asdict(make_local_primitive_controller_cfg()) == asdict(cfg)
```

- [ ] **Step 2: Run the focused test and verify missing-module failure**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_types_and_config.py -q
```

Expected: FAIL during collection because `controllers.local_primitives` does not exist.

- [ ] **Step 3: Implement immutable types and the exact shared configuration**

```python
class PrimitiveId(IntEnum):
    SIDE_TO_UPRIGHT = 0
    UPRIGHT_TO_SIDE = 1
    UPRIGHT_TO_30_DEG = 2
    CONE_30_DEG_ONE_REVOLUTION = 3


class PrimitiveStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED_HOLDING = "succeeded_holding"
    INVALID_START = "invalid_start"
    TIMED_OUT = "timed_out"
    NONFINITE = "nonfinite"


@dataclass(frozen=True)
class CapsuleState:
    sim_time_s: float
    position_world_m: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_world_m_s: np.ndarray
    angular_velocity_world_rad_s: np.ndarray


@dataclass(frozen=True)
class WrenchCommand:
    force_world_n: np.ndarray
    torque_world_nm: np.ndarray


@dataclass(frozen=True)
class LocalPrimitiveControllerCfg:
    capsule_mass_kg: float = 0.0057349997
    capsule_half_total_length_m: float = 0.0125
    capsule_half_cylinder_length_m: float = 0.006
    gravity_m_s2: float = 9.81
    axis_kp_nm_per_rad: float = 1.2e-5
    axis_kd_nms_per_rad: float = 2.0e-6
    roll_damping_nms_per_rad: float = 1.0e-6
    torque_limit_nm: float = 2.0e-5
    anchor_kp_n_per_m: float = 0.8
    anchor_kd_ns_per_m: float = 0.03
    horizontal_force_limit_weight_ratio: float = 0.5
    downward_preload_weight_ratio: float = 0.15
    stable_duration_s: float = 0.4
    max_stable_linear_speed_m_s: float = 0.02
    max_stable_angular_speed_rad_s: float = 0.15
    motion_duration_s: tuple[float, float, float, float] = (5.5, 4.5, 3.5, 8.0)
    hard_timeout_s: tuple[float, float, float, float] = (8.0, 7.0, 6.0, 9.5)
```

Index the immutable duration tuples with `int(PrimitiveId)`, giving `5.5/8.0`, `4.5/7.0`, `3.5/6.0`, and `8.0/9.5` seconds respectively. Validate tuple length, ordering, and every finite range in `__post_init__`; reject a hard timeout greater than or equal to `10.0`.

- [ ] **Step 4: Write failing trajectory tests**

```python
import numpy as np

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
    PrimitiveId,
    desired_axis_sample,
    posture_axis,
    quintic_scale,
)


def test_quintic_scale_has_zero_endpoint_speed():
    assert quintic_scale(0.0) == (0.0, 0.0)
    assert quintic_scale(1.0) == (1.0, 0.0)


def test_world_posture_axis_uses_requested_azimuth():
    np.testing.assert_allclose(posture_axis(0.0, 1.7), [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(posture_axis(90.0, 0.0), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(posture_axis(30.0, np.pi / 2), [0.0, 0.5, np.sqrt(3) / 2], atol=1e-12)


def test_cone_target_accumulates_exactly_one_revolution():
    start = posture_axis(30.0, 0.25)
    first = desired_axis_sample(PrimitiveId.CONE_30_DEG_ONE_REVOLUTION, start, 0.25, 0.0, 8.0)
    last = desired_axis_sample(PrimitiveId.CONE_30_DEG_ONE_REVOLUTION, start, 0.25, 8.0, 8.0)
    np.testing.assert_allclose(first.axis_world, last.axis_world, atol=1e-12)
    assert last.unwrapped_target_phase_rad - first.unwrapped_target_phase_rad == pytest.approx(2 * np.pi)
```

- [ ] **Step 5: Implement quintic posture trajectories without scene geometry**

```python
def quintic_scale(ratio: float) -> tuple[float, float]:
    r = float(np.clip(ratio, 0.0, 1.0))
    value = 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5
    derivative = 30.0 * r**2 - 60.0 * r**3 + 30.0 * r**4
    return value, derivative


def posture_axis(tilt_deg: float, azimuth_rad: float) -> np.ndarray:
    tilt = np.deg2rad(float(tilt_deg))
    return np.array([
        np.sin(tilt) * np.cos(azimuth_rad),
        np.sin(tilt) * np.sin(azimuth_rad),
        np.cos(tilt),
    ], dtype=np.float64)
```

Use shortest-arc unit-vector spherical interpolation for side-to-upright. Use `posture_axis(90, azimuth)` and `posture_axis(30, azimuth)` for the two upright-origin transitions. For the cone use `phi = phi_start + 2*pi*s` and compute `axis_dot_world` analytically from `dphi/dt`; set `desired_omega_world = cross(axis, axis_dot)`.

- [ ] **Step 6: Run pure tests and commit**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_types_and_config.py tests/local_primitives/test_trajectory.py -q
```

Expected: PASS with no Isaac application launch.

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/__init__.py tests/local_primitives/conftest.py tests/local_primitives/test_types_and_config.py tests/local_primitives/test_trajectory.py
git commit -m "feat: define local capsule primitive trajectories"
```

### Task 2: Implement the Pure Closed-Loop Wrench State Machine

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/controller.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/__init__.py`
- Create: `tests/local_primitives/test_controller.py`

**Interfaces:**
- Consumes: `CapsuleState`, `PrimitiveId`, `LocalPrimitiveControllerCfg`, and `desired_axis_sample()`.
- Produces: `LocalPrimitiveController.start(primitive_id, azimuth_rad, state)`, `update(state, physics_dt_s) -> tuple[WrenchCommand, PrimitiveTelemetry]`, `reset()`, and `status`.

- [ ] **Step 1: Write failing coordinate, start-gate, and anchor tests**

```python
def test_axis_points_from_non_camera_to_camera():
    controller = LocalPrimitiveController(make_local_primitive_controller_cfg())
    state = capsule_state(quaternion_wxyz=[1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(controller.directed_axis_world(state), [0.0, 0.0, -1.0])


def test_invalid_upright_origin_command_does_not_apply_wrench():
    controller = LocalPrimitiveController(make_local_primitive_controller_cfg())
    side_state = state_with_axis([1.0, 0.0, 0.0])
    accepted = controller.start(PrimitiveId.UPRIGHT_TO_30_DEG, 0.0, side_state)
    wrench, telemetry = controller.update(side_state, 1.0 / 240.0)
    assert accepted is False
    assert telemetry.status is PrimitiveStatus.INVALID_START
    np.testing.assert_array_equal(wrench.force_world_n, np.zeros(3))
    np.testing.assert_array_equal(wrench.torque_world_nm, np.zeros(3))


def test_rise_records_non_camera_xy_anchor():
    controller = LocalPrimitiveController(make_local_primitive_controller_cfg())
    side_state = state_with_axis([1.0, 0.0, 0.0], position=[0.10, 0.20, 0.0125])
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, side_state)
    np.testing.assert_allclose(controller.anchor_xy, [0.0875, 0.20])
```

- [ ] **Step 2: Write failing feedback, saturation, timeout, and holding tests**

```python
from dataclasses import replace


def make_fast_test_cfg(*, hard_timeout_s: float = 0.8):
    cfg = make_local_primitive_controller_cfg()
    return replace(
        cfg,
        motion_duration_s=(0.05, 0.05, 0.05, 0.10),
        hard_timeout_s=(hard_timeout_s,) * 4,
        stable_duration_s=0.05,
    )


def advance_with_state(controller, state, *, seconds: float):
    wrench = telemetry = None
    for _ in range(int(np.ceil(seconds * 240.0))):
        wrench, telemetry = controller.update(state, 1.0 / 240.0)
    return wrench, telemetry


def test_wrench_is_bounded_and_world_down_preload_is_explicit():
    cfg = make_local_primitive_controller_cfg()
    controller = LocalPrimitiveController(cfg)
    state = state_with_axis([1.0, 0.0, 0.0], position=[0.0, 0.0, 0.02])
    controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, state)
    wrench, telemetry = controller.update(state, 1.0 / 240.0)
    assert np.linalg.norm(wrench.force_world_n[:2]) <= cfg.horizontal_force_limit_weight_ratio * cfg.capsule_mass_kg * cfg.gravity_m_s2 + 1e-12
    assert wrench.force_world_n[2] == pytest.approx(-cfg.downward_preload_weight_ratio * cfg.capsule_mass_kg * cfg.gravity_m_s2)
    assert np.linalg.norm(wrench.torque_world_nm) <= cfg.torque_limit_nm + 1e-12


def test_success_enters_closed_loop_holding_instead_of_restarting():
    controller = LocalPrimitiveController(make_fast_test_cfg())
    state = state_with_axis([0.0, 0.0, 1.0])
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, state_with_axis([1.0, 0.0, 0.0]))
    advance_with_state(controller, state, seconds=0.6)
    assert controller.status is PrimitiveStatus.SUCCEEDED_HOLDING
    wrench, telemetry = controller.update(state, 1.0 / 240.0)
    assert telemetry.elapsed_s == pytest.approx(telemetry.completion_time_s)
    assert controller.status is PrimitiveStatus.SUCCEEDED_HOLDING


def test_timeout_clears_wrench_and_never_changes_state():
    controller = LocalPrimitiveController(make_fast_test_cfg(hard_timeout_s=0.1))
    state = state_with_axis([1.0, 0.0, 0.0])
    controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, state)
    wrench, telemetry = advance_with_state(controller, state, seconds=0.2)
    assert telemetry.status is PrimitiveStatus.TIMED_OUT
    np.testing.assert_array_equal(wrench.force_world_n, np.zeros(3))
    np.testing.assert_array_equal(wrench.torque_world_nm, np.zeros(3))
```

- [ ] **Step 3: Implement start gates and read-only directed-axis conversion**

Normalize the WXYZ quaternion, construct its rotation matrix, and compute `u = R @ [0,0,-1]`. Reject nonfinite or zero-norm state. Implement exact gates `75 <= theta_deg <= 105`, `theta_deg <= 5`, and `abs(theta_deg - 30) <= 3`. Record the measured starting axis, requested/default azimuth, start time, and `anchor_xy = p_xy - 0.0125*u_xy` only after a command passes its start gate.

- [ ] **Step 4: Implement the 240 Hz feedback law and norm saturation**

```python
axis_error = np.cross(axis_world, target.axis_world)
omega_perp = omega_world - np.dot(omega_world, axis_world) * axis_world
torque = (
    cfg.axis_kp_nm_per_rad * axis_error
    + cfg.axis_kd_nms_per_rad * (target.angular_velocity_world_rad_s - omega_perp)
    - cfg.roll_damping_nms_per_rad * np.dot(omega_world, axis_world) * axis_world
)
torque = saturate_norm(torque, cfg.torque_limit_nm)

desired_position_xy = anchor_xy + cfg.capsule_half_total_length_m * target.axis_world[:2]
desired_velocity_xy = cfg.capsule_half_total_length_m * target.axis_dot_world_s[:2]
force_xy = (
    cfg.anchor_kp_n_per_m * (desired_position_xy - state.position_world_m[:2])
    + cfg.anchor_kd_ns_per_m * (desired_velocity_xy - state.linear_velocity_world_m_s[:2])
)
force_xy = saturate_norm(force_xy, cfg.horizontal_force_limit_weight_ratio * mass * gravity)
force = np.array([force_xy[0], force_xy[1], -cfg.downward_preload_weight_ratio * mass * gravity])
```

Do not read contact normals, mesh data, plane height, or task identity. The controller module must contain no Isaac imports.

- [ ] **Step 5: Implement completion, cone unwrapping, holding, and failure**

After the motion profile ends, accumulate stable time only while angle error and speed thresholds pass. For the cone, unwrap `atan2(u_y, u_x)` using successive signed angle differences, accumulate actual coverage, and accumulate tilt squared error. Require actual coverage at least `2*pi-radians(10)` and cone tilt RMSE at most `radians(5)` before the stable timer can complete.

On success freeze the completion time and continue updating against the final axis. Accept a new valid start pulse from holding. On timeout or nonfinite state set status and return an exactly zero wrench for every subsequent update until reset or a new valid command.

- [ ] **Step 6: Run pure controller tests, forbidden-term scan, and commit**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_controller.py tests/local_primitives/test_trajectory.py -q
python -c "from pathlib import Path; p=Path('source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives'); s='\n'.join(x.read_text() for x in p.glob('*.py')); assert all(k not in s for k in ('surface_mesh','clearance','raycast','write_root_pose','write_root_velocity','set_transforms','set_velocities'))"
```

Expected: PASS and no forbidden term.

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives tests/local_primitives/test_controller.py
git commit -m "feat: add closed-loop primitive wrench controller"
```

### Task 3: Adapt the Controller to the Isaac Lab Center-of-Mass Wrench API

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/local_primitive_action.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.pyi`
- Create: `scripts/local_primitives/inspect_local_primitives_prerequisites.py`
- Create: `tests/local_primitives/test_action_command.py`
- Create: `tests/local_primitives/test_preflight_contract.py`

**Interfaces:**
- Consumes: four-float actions `[start_pulse, primitive_code, direction_x, direction_y]`, live capsule mass/pose/velocity, `LocalPrimitiveController`, and the permanent wrench composer.
- Produces: `LocalPrimitiveAction`, `LocalPrimitiveActionTermCfg`, `make_local_primitive_action_cfg()`, applied wrench properties, command-result properties, and substep telemetry.

- [ ] **Step 1: Write failing pulse-decoder and busy-command tests**

```python
def test_start_is_rising_edge_and_zero_direction_defaults_to_world_x():
    decoder = PrimitiveCommandDecoder()
    assert decoder.decode([1.0, 2.0, 0.0, 0.0]) == PrimitiveRequest(
        primitive_id=PrimitiveId.UPRIGHT_TO_30_DEG,
        azimuth_rad=0.0,
    )
    assert decoder.decode([1.0, 2.0, 0.0, 0.0]) is None
    assert decoder.decode([0.0, 2.0, 0.0, 0.0]) is None
    assert decoder.decode([1.0, 2.0, 0.0, 1.0]).azimuth_rad == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("code", [-1.0, 4.0, np.nan, np.inf])
def test_invalid_primitive_code_is_rejected(code):
    decoder = PrimitiveCommandDecoder()
    with pytest.raises(ValueError):
        decoder.decode([1.0, code, 1.0, 0.0])
```

- [ ] **Step 2: Implement a pure command decoder and the ActionTerm skeleton**

Set `action_dim = 4`. `process_actions()` stores the latest finite command and calls the decoder only at 60 Hz. `apply_actions()` reads `root_com_pose_w` and `root_com_vel_w`, creates `CapsuleState`, advances the controller by exactly `1/240 s`, converts the returned NumPy arrays to the action tensor device, and applies both force and torque with `positions=None`, `body_ids=None`, `env_ids=None`, and `is_global=True`.

```python
self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
    forces=self._applied_force_world[:, None, :],
    torques=self._applied_torque_world[:, None, :],
    positions=None,
    body_ids=None,
    env_ids=None,
    is_global=True,
)
```

- [ ] **Step 3: Verify live mass and dynamic-body invariants before controller construction**

Read `body_mass` and replace the design's measured mass in the runtime config only if the live finite positive value matches within `1e-6 kg`; otherwise return `needs_decision`. Reuse the TASK-003 body-level CCD, non-kinematic, gravity, and scene-CCD checks. Reject more than one environment.

- [ ] **Step 4: Expose deterministic telemetry and reset semantics**

Expose `applied_force_world`, `applied_torque_world`, `telemetry`, `last_request_result`, and a bounded deque of 240 Hz samples. `reset()` must zero the action, reset the rising-edge decoder and pure controller, clear telemetry, and reset the permanent wrench composer. It must not call a root-state setter.

- [ ] **Step 5: Write the preflight source and live contract checks**

The preflight must report capsule prim, live dimensions, mass, inertia, camera local offset `(0,0,-0.0127)`, local `Z` capsule axis, non-kinematic state, gravity, body and scene CCD, timing, contact sensor, wrench API `positions=None` center-of-mass semantics, registered new tasks, action terms, and runtime forbidden-call scan.

If camera endpoint convention, direct torque support, or read-only flat contact-point access is unavailable, emit `LOCAL_PRIMITIVE_PREFLIGHT_NEEDS_DECISION` and exit before behavioral tuning.

- [ ] **Step 6: Run focused tests, compile, and commit**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_action_command.py tests/local_primitives/test_preflight_contract.py -q
./run_isaaclab.sh -p -m compileall -q source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/local_primitive_action.py scripts/local_primitives/inspect_local_primitives_prerequisites.py
```

Expected: PASS and no syntax errors.

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/local_primitive_action.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.pyi scripts/local_primitives/inspect_local_primitives_prerequisites.py tests/local_primitives/test_action_command.py tests/local_primitives/test_preflight_contract.py
git commit -m "feat: apply local primitive force and torque"
```

### Task 4: Add and Quantitatively Validate the Isolated Flat Task

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_local_primitives_flat_env_cfg.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Create: `scripts/local_primitives/validate_local_primitives_flat.py`
- Create: `tests/local_primitives/test_task_cfg.py`
- Create: `tests/local_primitives/test_flat_summary.py`

**Interfaces:**
- Produces task ID `Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0` and schema `local_primitives_flat_validation_v1`.
- Consumes the existing table scene, standard reset, shared action-config factory, capsule contact sensor, and read-only PhysX contact points.

- [ ] **Step 1: Write failing registration, isolation, and timing tests**

```python
FLAT_ID = "Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0"


def test_flat_task_is_isolated_and_uses_frozen_rates():
    spec = gym.spec(FLAT_ID)
    cfg = flat_cfg_type()()
    assert "LocalPrimitivesFlat" in spec.kwargs["env_cfg_entry_point"]
    assert cfg.scene.num_envs == 1
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.decimation == 4
    assert cfg.sim.render_interval == 4
    assert cfg.scene.capsule_camera.update_period == 1.0 / 30.0
    assert cfg.sim.device == "cpu"
    assert term_names(cfg.actions) == ["local_primitive"]
    assert term_names(cfg.events) == ["reset_scene"]
    assert "joint_position" not in term_names(cfg.actions)
    assert "magnetic_physics" not in term_names(cfg.actions)
    assert "dynamic_force" not in term_names(cfg.actions)
```

- [ ] **Step 2: Implement the flat environment configuration and registration**

Extend `RobotarmMagneticTableLabEnvCfg`, replace actions with a config containing only `local_primitive = make_local_primitive_action_cfg()`, replace events with the standard reset only, use RGB-only observations, no rewards, timeout-only termination, one environment, CPU PhysX with CCD, 240/60/60/30 simulated-Hz cadence, and the existing side-lying table reset. Do not edit the table USD or shared table task.

- [ ] **Step 3: Write failing flat-summary acceptance tests**

```python
def test_flat_summary_requires_four_sub_ten_second_successes(valid_flat_summary):
    summary = valid_flat_summary()
    assert evaluate_flat_summary(summary)["status"] == "pass"
    summary["primitives"]["cone_30"]["completion_time_s"] = 10.0
    assert evaluate_flat_summary(summary)["status"] == "fail"


def test_rise_rejects_camera_hemisphere_support(valid_flat_summary):
    summary = valid_flat_summary()
    summary["primitives"]["side_to_upright"]["camera_hemisphere_load_samples"] = 1
    assert evaluate_flat_summary(summary)["status"] == "fail"


def test_contact_does_not_fail_without_tracking_failure(valid_flat_summary):
    summary = valid_flat_summary()
    summary["contact"]["max_force_n"] = 100.0
    assert evaluate_flat_summary(summary)["status"] == "pass"
```

- [ ] **Step 4: Implement the four deterministic flat sequences**

Reset to the delivered side-lying state before each sequence. Pulse one command for one 60 Hz environment step and then send `start_pulse=0` while stepping continuously. Execute exactly:

```python
SEQUENCES = (
    (PrimitiveId.SIDE_TO_UPRIGHT,),
    (PrimitiveId.SIDE_TO_UPRIGHT, PrimitiveId.UPRIGHT_TO_SIDE),
    (PrimitiveId.SIDE_TO_UPRIGHT, PrimitiveId.UPRIGHT_TO_30_DEG),
    (PrimitiveId.SIDE_TO_UPRIGHT, PrimitiveId.UPRIGHT_TO_30_DEG, PrimitiveId.CONE_30_DEG_ONE_REVOLUTION),
)
```

Use world `+X` direction for all direction-bearing primitives. Do not reset directly to upright or 30 degrees.

- [ ] **Step 5: Implement read-only support-contact classification**

For each capsule-plane contact point compute

```python
sigma_m = float(np.dot(contact_point_world_m - capsule_center_world_m, directed_axis_world))
camera_hemisphere = sigma_m > 0.006 + 0.0005
non_camera_hemisphere = sigma_m < -0.006 + 0.0005
```

Classify a point as load-bearing when its normal impulse or normal force exceeds 10% of the largest simultaneous capsule-plane contact. During side-to-upright require zero load-bearing camera-hemisphere samples. After target tilt is below 45 degrees, require the dominant load-bearing point to be on the non-camera hemisphere. This is validator-only evidence; do not feed it to the controller.

- [ ] **Step 6: Implement summary metrics and acceptance**

Record per primitive start/end state, completion time, target/final angle, maximum angle error after profile end, stable duration, maximum linear/angular speed in the stable window, maximum force/torque, saturation fractions, actual unwrapped cone coverage, cone tilt RMSE, status/reason, contact-point classification, and nonfinite count.

Pass only if all four primitives succeed before their own hard timeouts; final errors and stable speeds satisfy the spec; cone coverage is at least `2*pi-radians(10)` and RMSE at most `radians(5)`; rise support passes; no state is nonfinite; and no forbidden state writer is observed. Do not compute or store surface clearance.

- [ ] **Step 7: Run the mandatory live preflight and flat validator**

Run:

```bash
./run_isaaclab.sh -p scripts/local_primitives/inspect_local_primitives_prerequisites.py --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 --headless
./run_isaaclab.sh -p scripts/local_primitives/validate_local_primitives_flat.py --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 --seed 42 --direction_azimuth_deg 0 --headless
```

Expected: `LOCAL_PRIMITIVE_PREFLIGHT_PASS` followed by `LOCAL_PRIMITIVES_FLAT_VALIDATION_PASS`. If preflight returns `needs_decision`, stop. If behavior fails, continue only with the bounded shared-controller calibration in Task 5; do not alter physics.

- [ ] **Step 8: Commit the flat task and validator**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_local_primitives_flat_env_cfg.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py scripts/local_primitives/validate_local_primitives_flat.py tests/local_primitives/test_task_cfg.py tests/local_primitives/test_flat_summary.py
git commit -m "test: validate local primitives on flat contact"
```

### Task 5: Calibrate Only the Shared Controller and Freeze the Flat-Passing Profile

**Files:**
- Modify only if measurements require it: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/config.py`
- Create outside Git: `logs/local_primitives_flat/<timestamp>/calibration_attempts.jsonl`
- Create outside Git: `logs/local_primitives_flat/<timestamp>/summary.json`

**Interfaces:**
- Consumes: flat validator failure metrics and the explicit parameter ranges in the design spec.
- Produces: one tracked shared final configuration and a complete external record of every attempted value set.

- [ ] **Step 1: Record the initial exact parameter set before any tuning**

Write one JSONL row containing commit, attempt index `0`, all controller config fields, validation command, summary path, status, completion times, final errors, saturation fractions, cone coverage/RMSE, and support-contact result.

- [ ] **Step 2: Tune one controller category at a time only when evidence requires it**

Use this fixed order: axis `Kp/Kd`, torque limit, XY anchor `Kp/Kd`, horizontal force limit, downward preload, then motion duration. Keep every value inside the design ranges. Never change physics, material, mass, inertia, damping, restitution, CCD, solver, reset, geometry, camera, or scene.

- [ ] **Step 3: Rerun all pure tests and the complete flat validator after every tracked config change**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives -q --disable-warnings
./run_isaaclab.sh -p scripts/local_primitives/validate_local_primitives_flat.py --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 --seed 42 --direction_azimuth_deg 0 --headless
```

Expected: the final attempt returns `LOCAL_PRIMITIVES_FLAT_VALIDATION_PASS`; intermediate failures remain in the external attempt log.

- [ ] **Step 4: Commit the final shared profile if it differs from the initial profile**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/config.py tests/local_primitives
git commit -m "tune: freeze flat-passing primitive controller"
```

If the initial profile passes, do not create an empty tuning commit.

### Task 6: Add the Stomach Scene Wrapper with an Enforced Identical Controller

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_local_primitives_stomach_env_cfg.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Modify: `tests/local_primitives/test_task_cfg.py`
- Create: `tests/local_primitives/test_no_stomach_adaptation.py`

**Interfaces:**
- Produces task ID `Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0`.
- Consumes TASK-003 stomach placement/timing/physics and the exact same `make_local_primitive_action_cfg()` factory used by the flat task.

- [ ] **Step 1: Write failing shared-config equality and reset-preservation tests**

```python
from dataclasses import asdict


def test_flat_and_stomach_use_identical_controller_config():
    flat = flat_cfg_type()().actions.local_primitive
    stomach = stomach_cfg_type()().actions.local_primitive
    assert flat.asset_name == stomach.asset_name == "capsule"
    assert asdict(stomach.controller_cfg) == asdict(flat.controller_cfg)


def test_stomach_preserves_task003_scene_and_reset():
    task003 = RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg()
    stomach = stomach_cfg_type()()
    assert stomach.scene.stomach.init_state.pos == task003.scene.stomach.init_state.pos
    assert stomach.scene.stomach.init_state.rot == task003.scene.stomach.init_state.rot
    assert stomach.scene.capsule.init_state.pos == task003.scene.capsule.init_state.pos
    assert stomach.scene.capsule.init_state.rot == task003.scene.capsule.init_state.rot
    assert stomach.sim.dt == task003.sim.dt
    assert stomach.decimation == task003.decimation
    assert stomach.sim.render_interval == task003.sim.render_interval
    assert stomach.scene.capsule_camera.update_period == task003.scene.capsule_camera.update_period
```

- [ ] **Step 2: Implement the stomach environment as a scene-only wrapper**

Extend `RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg`, replace the inherited `dynamic_force` action with only `local_primitive = make_local_primitive_action_cfg()`, and preserve TASK-003's stomach flip, capsule reset, camera, contact sensor, rates, CPU device, CCD, observations, reset event, and viewer. Do not add a `__post_init__` override for control parameters.

- [ ] **Step 3: Add a static no-adaptation source scan**

```python
def test_task004_runtime_has_no_geometry_adaptation_or_state_writer():
    runtime = read_task004_runtime_sources()
    forbidden = (
        "surface_mesh", "clearance", "nearest_triangle", "raycast",
        "swept", "local_normal", "tangent_frame", "project_to_surface",
        "write_root_pose", "write_root_velocity", "set_transforms", "set_velocities",
    )
    for token in forbidden:
        assert token not in runtime


def test_controller_has_no_task_or_stomach_branch():
    source = controller_source()
    assert "task_id" not in source
    assert "stomach" not in source.lower()
```

- [ ] **Step 4: Run configuration/isolation tests and live stomach preflight**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_task_cfg.py tests/local_primitives/test_no_stomach_adaptation.py -q
./run_isaaclab.sh -p scripts/local_primitives/inspect_local_primitives_prerequisites.py --task Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0 --headless
```

Expected: PASS. The stomach preflight confirms the same controller-profile digest as the flat task and the unchanged TASK-003 reset.

- [ ] **Step 5: Commit the unchanged-controller stomach wrapper**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_local_primitives_stomach_env_cfg.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py tests/local_primitives/test_task_cfg.py tests/local_primitives/test_no_stomach_adaptation.py
git commit -m "feat: migrate primitive controller unchanged to stomach"
```

### Task 7: Add Continuous Rendering, Run Visual Migration, and Regress Existing Tasks

**Files:**
- Create: `scripts/local_primitives/teleop_local_primitives.py`
- Create: `tests/local_primitives/test_launcher_protocol.py`
- Output outside Git: `logs/local_primitives_visual/<timestamp>/samples.jsonl`
- Output outside Git: `logs/local_primitives_visual/<timestamp>/session.json`
- Output outside Git: `logs/local_primitives_visual/<timestamp>/snapshots/`

**Interfaces:**
- Consumes either TASK-004 task ID, keys `1` through `4`, reset/snapshot/exit commands, and the shared action term.
- Produces one-step start pulses, continuous zero-pulse stepping, rendered external/capsule views, scripted sequences, and evidence.

- [ ] **Step 1: Write failing keyboard and pulse tests**

```python
@pytest.mark.parametrize(
    ("key", "primitive"),
    [("1", PrimitiveId.SIDE_TO_UPRIGHT),
     ("2", PrimitiveId.UPRIGHT_TO_SIDE),
     ("3", PrimitiveId.UPRIGHT_TO_30_DEG),
     ("4", PrimitiveId.CONE_30_DEG_ONE_REVOLUTION)],
)
def test_number_keys_queue_one_start_pulse(key, primitive):
    protocol = LocalPrimitiveKeyboardProtocol()
    protocol.key_event(key, True)
    assert protocol.next_action() == pytest.approx([1.0, float(primitive), 1.0, 0.0])
    assert protocol.next_action() == pytest.approx([0.0, float(primitive), 1.0, 0.0])


def test_direction_azimuth_is_configurable_without_scene_frame():
    protocol = LocalPrimitiveKeyboardProtocol(direction_azimuth_deg=90.0)
    protocol.key_event("3", True)
    np.testing.assert_allclose(protocol.next_action()[2:], [0.0, 1.0], atol=1e-12)
```

- [ ] **Step 2: Implement one shared continuous launcher**

Support `--task`, `--direction_azimuth_deg`, `--scripted_sequence`, `--max_steps`, `--capsule_camera_view`, and `--output_directory`. Allow only the two TASK-004 IDs. The main loop must call `env.step(action)` continuously, including while the action pulse is zero and while the controller is holding.

```python
while simulation_app.is_running() and not exit_requested:
    action = protocol.next_action()
    tensor = torch.as_tensor(action, device=env.unwrapped.device, dtype=torch.float32).reshape(1, 4)
    observation, reward, terminated, truncated, info = env.step(tensor)
    recorder.append(sample_from_env(env, action_term))
    process_reset_snapshot_exit()
```

- [ ] **Step 3: Run a rendered flat sequence before stomach migration**

Run:

```bash
./run_isaaclab.sh -p scripts/local_primitives/teleop_local_primitives.py --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 --scripted_sequence "0,1;reset;0,2;reset;0,2,3" --direction_azimuth_deg 0 --capsule_camera_view --viz kit
```

Expected: continuous external motion and 30 Hz capsule-camera updates; every scripted primitive reaches its recorded terminal status; no snap, teleport, or idle render freeze is observed. If no human reviews smoothness, mark only that subjective claim unverified.

- [ ] **Step 4: Run the identical rendered sequence in the stomach scene**

Run:

```bash
./run_isaaclab.sh -p scripts/local_primitives/teleop_local_primitives.py --task Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0 --scripted_sequence "0,1;reset;0,2;reset;0,2,3" --direction_azimuth_deg 0 --capsule_camera_view --viz kit
```

Expected: the task uses the flat-frozen controller-profile digest and unchanged TASK-003 reset, renders continuously, records every contact/saturation/timeout, and performs no geometry query or pose correction. Contact with folds or wall is not an independent failure. Do not tune any parameter after this command.

- [ ] **Step 5: Run all TASK-004 tests and TASK-003 regressions**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives -q --disable-warnings
./run_isaaclab.sh -p -m pytest tests/dynamic_force -q --disable-warnings
./run_isaaclab.sh -p scripts/dynamic_force/inspect_dynamic_force_prerequisites.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 --num_envs 1 --headless
```

Expected: TASK-004 and TASK-003 focused tests pass and the TASK-003 preflight remains passing. Do not require the historical TASK-003 penetration-sensitive validator to change from its recorded `partial` disposition.

- [ ] **Step 6: Run delivered broader regressions and hygiene checks**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface tests/coverage tests/action_layer/test_atomic_protocol.py tests/action_layer/test_executor.py tests/action_layer/test_safety.py tests/action_layer/test_atomic_stomach_teleop_cfg.py tests/action_layer/test_atomic_keyboard_protocol.py -q --disable-warnings
./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py --num_envs 1 --coverage_samples 5
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_table.py --num_envs 1 --max_steps_per_action 60 --viz kit
./run_isaaclab.sh -p scripts/zero_agent.py --task Template-Robotarm-Magnetic-Table-Lab-v0 --num_envs 1 --max_steps 5 --viz kit
./run_isaaclab.sh -p -m compileall -q scripts/local_primitives source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab
git diff --check
```

Expected: all delivered regression commands preserve their previous passing results; compile and diff checks pass.

- [ ] **Step 7: Commit the shared launcher**

```bash
git add scripts/local_primitives/teleop_local_primitives.py tests/local_primitives/test_launcher_protocol.py
git commit -m "feat: add rendered local primitive launcher"
```

### Task 8: Document, Report Evidence, and Push Without Merging

**Files:**
- Create: `docs/LOCAL_DYNAMICS_PRIMITIVES.md`
- Create: `handoffs/reports/TASK-004-local-dynamics-primitives-report.md`

**Interfaces:**
- Consumes: exact commands/results, final config, calibration JSONL, flat summary, rendered session metadata, regressions, Git state, and artifact hashes.
- Produces: operator documentation and the authoritative TASK-004 report.

- [ ] **Step 1: Write operator documentation**

Document both task IDs, four primitive codes and keys, the camera/non-camera axis convention, world-coordinate posture meaning, command pulse interface, default direction, feedback and holding semantics, exact final shared parameters, flat validation command, rendered flat/stomach commands, log schemas, reset/snapshot/exit controls, and the absence of magnet control, surface adaptation, clearance checks, collision avoidance, and pose correction.

- [ ] **Step 2: Write the report from observed evidence only**

The report must state `complete`, `partial`, `needs_decision`, or `blocked`; record planning base/head and implementation branch/head; list every commit and command; record live mass/inertia/endpoint/wrench/timing gates; include every calibration attempt and final parameter; summarize each flat primitive's completion time, final error, speeds, saturation, contact classification, and cone coverage; state the stomach controller-profile digest and reset equality; list rendered observations, collisions, timeouts, deviations, and unverified subjective claims; and list external artifacts with absolute path, byte size, and SHA-256.

- [ ] **Step 3: Apply the disposition rules exactly**

Use `needs_decision` if endpoint convention, torque API, contact points, or task isolation cannot be established. Use `partial` if any flat primitive fails, reaches `10.0 s`, uses camera-hemisphere load support during rise, becomes nonfinite, or violates a forbidden-state rule. Use implementation `complete` only when the flat gate and regressions pass and unchanged-controller stomach rendered evidence exists. Do not equate an unreviewed stomach video with user-confirmed usability.

- [ ] **Step 4: Commit documentation and report**

```bash
git add docs/LOCAL_DYNAMICS_PRIMITIVES.md handoffs/reports/TASK-004-local-dynamics-primitives-report.md
git commit -m "docs: report local capsule dynamics primitives"
```

- [ ] **Step 5: Verify scope and push without merging**

Run:

```bash
git diff --check
git status --short
git diff --name-only 06b15caf9a69bc9c20f85522ce4abbb32c8b9245...HEAD
git push -u origin feature/TASK-004-local-dynamics-primitives
```

Expected: only TASK-004 source, tests, scripts, docs, and the report differ from the base; no asset file is changed; the feature branch is available to Windows; Linux does not merge it.

## Final Review Gate

Before claiming completion, confirm that every new runtime file is free of capsule state setters and surface-geometry adaptation; every motion comes from the applied wrench plus PhysX; the camera-end convention is correct; the rise support evidence excludes the camera hemisphere; all four completion times are strictly below 10 seconds; the cone uses actual unwrapped coverage; the flat and stomach controller-profile digests are identical; the stomach reset equals TASK-003; all collisions remain visible rather than repaired; and no parameter was changed after the stomach visual run began.
