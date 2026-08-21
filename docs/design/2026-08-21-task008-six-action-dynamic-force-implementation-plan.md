# TASK-008 Six-Action Dynamic-Force Controller Implementation Plan

## Execution rule

Linux shall execute this checklist manually on `feature/TASK-008-six-action-dynamic-force-controller`. No Codex skill or plugin is required or authorized as a prerequisite. The authoritative design is `docs/design/2026-08-21-task008-six-action-dynamic-force-design.md`; when this plan and the design differ, stop and return `needs_decision` instead of choosing silently.

## Goal

Build a six-ID, one-second, force-driven capsule macro controller on the TASK-003 dynamic rigid-body path; calibrate and validate it on a flat table; then expose the unchanged controller in a synchronized three-view stomach keyboard launcher.

## Fixed baseline and file map

The source baseline is `06b15caf9a69bc9c20f85522ce4abbb32c8b9245`. Preserve TASK-003's CPU PhysX, body CCD, gravity, contact, camera, and absence of runtime pose correction.

Create or modify only the following implementation areas unless the report identifies a directly required export:

```text
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/dynamic_force_macro.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/dynamic_force_macro_action.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_dynamic_force_macro_table_env_cfg.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_dynamic_force_macro_stomach_env_cfg.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/dynamic_force_macro_keyboard.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/dynamic_force_macro_runner.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/reference_mesh.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/visibility.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/simulator_runtime.py
scripts/dynamic_force_macro/common.py
scripts/dynamic_force_macro/inspect_prerequisites.py
scripts/dynamic_force_macro/calibrate_validate_table.py
scripts/dynamic_force_macro/teleop_stomach.py
tests/dynamic_force_macro/conftest.py
tests/dynamic_force_macro/test_contract.py
tests/dynamic_force_macro/test_action_term.py
tests/dynamic_force_macro/test_keyboard.py
tests/dynamic_force_macro/test_metrics.py
tests/dynamic_force_macro/test_calibration_protocol.py
tests/dynamic_force_macro/test_task_cfg.py
tests/dynamic_force_macro/test_sync_protocol.py
tests/dynamic_force_macro/test_stomach_views.py
tests/coverage/test_visibility_geometry.py
docs/DYNAMIC_FORCE_MACRO_CONTROLLER.md
handoffs/reports/TASK-008-six-action-dynamic-force-controller-report.md
```

Do not modify USD assets, the TASK-003 controller, magnetic code, ideal-surface code, VLM/RL code, rewards, or previously recorded reports.

## Task A: Freeze the pure six-action contract

**Deliverable:** A simulator-independent action, timing, force, geometry, and acceptance-metric module.

- [ ] Add `DynamicForceMacroActionId(IntEnum)` with exactly `HOLD=0`, `MOVE_POS=1`, `MOVE_NEG=2`, `VIEW_POS=3`, `VIEW_NEG=4`, and `UP=5`.
- [ ] Add immutable `DynamicForceMacroConfig` with `physics_hz=240`, `environment_hz=60`, `camera_hz=30`, `actor_hz=1`, `move_force_ratio=0.9`, `view_force_ratio=0.9`, `up_force_ratio=0.9`, `max_force_ratio=3.0`, and camera-side local-axis sign `-1`.
- [ ] Add pure phase lookup that returns zero for MOVE/VIEW substeps `0..47`, active for `48..191`, and zero for `192..239`; UP is active for `0..239`; HOLD is always zero.
- [ ] Add pure lateral-direction calculation from world Z and the camera-directed long axis. It must return unit finite `+d/-d` and raise a descriptive numerical-contract error if the cross product cannot be normalized.
- [ ] Add pure endpoint-force composition that returns named point forces before any equivalent-wrench conversion. MOVE must return two forces of `0.5*r_move*mg`; VIEW and UP must return one force of `r*mg`.
- [ ] Add pure equivalent-COM-wrench composition using `sum(F_i)` and `sum((p_i-p_com) cross F_i)`.
- [ ] Add pure MOVE, VIEW, and UP metric functions implementing the exact design equations and units.

The first failing tests must include the following assertions:

```python
def test_action_ids_are_frozen():
    assert [int(value) for value in DynamicForceMacroActionId] == list(range(6))


def test_move_total_force_is_nine_tenths_weight_split_equally():
    points = point_forces_for_action(
        DynamicForceMacroActionId.MOVE_POS,
        mass_kg=0.005735,
        lateral_direction_world=np.array([0.0, 1.0, 0.0]),
        camera_center_world=np.array([0.0, 0.0, -0.006]),
        other_center_world=np.array([0.0, 0.0, 0.006]),
        config=DynamicForceMacroConfig(),
    )
    assert len(points) == 2
    np.testing.assert_allclose(np.linalg.norm(points[0].force_world), 0.45 * 0.005735 * 9.81)
    np.testing.assert_allclose(np.linalg.norm(points[1].force_world), 0.45 * 0.005735 * 9.81)


def test_move_and_view_phase_boundaries_are_exact():
    assert not phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 47).force_active
    assert phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 48).force_active
    assert phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 191).force_active
    assert not phase_for_substep(DynamicForceMacroActionId.MOVE_POS, 192).force_active


def test_up_is_active_on_final_substep():
    assert phase_for_substep(DynamicForceMacroActionId.UP, 239).force_active
```

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force_macro/test_contract.py tests/dynamic_force_macro/test_metrics.py -q
```

Commit only after the focused tests pass:

```bash
git add -f source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/dynamic_force_macro.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/__init__.py tests/dynamic_force_macro/conftest.py tests/dynamic_force_macro/test_contract.py tests/dynamic_force_macro/test_metrics.py
git commit -m "feat: define six-action dynamic force macros"
```

## Task B: Verify live geometry and endpoint-wrench API semantics

**Deliverable:** A live preflight that proves which capsule frames and PhysX API path the controller uses.

- [ ] Extend TASK-003 preflight in a new `scripts/dynamic_force_macro/inspect_prerequisites.py`; do not weaken the existing preflight.
- [ ] Record capsule radius, cylinder height, total length, link origin, local COM, world COM, local long axis, camera offset, camera-side sphere-center sign, body CCD, scene CCD, gravity, mass, inertia, table collider, stomach collider, and all clock values.
- [ ] Verify that the camera is on local negative Z and that the hemisphere centers are derived from half the verified cylindrical height rather than from the optical-center offset.
- [ ] Probe the installed point-force API. If it can apply multiple point forces to one body, record the method and position frame. Otherwise select the equivalent-COM-wrench path and verify it against a pure two-force fixture.
- [ ] Add a source scan proving that no new runtime module calls `write_root_pose`, `write_root_velocity`, `set_transforms`, or `set_velocities`; reset and randomized trial setup are the only allowed state writers.
- [ ] Fail preflight before behavioral tests if the force point, coordinate frame, camera-side sign, or dynamic-body state is ambiguous.

Required comparison test:

```python
def test_two_point_force_and_equivalent_com_wrench_match():
    force, torque = equivalent_com_wrench(point_forces, com_world)
    np.testing.assert_allclose(force, sum(item.force_world for item in point_forces))
    np.testing.assert_allclose(
        torque,
        sum(np.cross(item.position_world - com_world, item.force_world) for item in point_forces),
    )
```

Run:

```bash
./run_isaaclab.sh -p scripts/dynamic_force_macro/inspect_prerequisites.py --task Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0 --headless
```

Expected terminal marker: `TASK008_PREFLIGHT_PASS`. Commit the preflight and its tests separately.

## Task C: Implement the 240 Hz action term and 60 Hz macro runner

**Deliverable:** Direct endpoint-force dynamics with one-ID, one-second synchronous invocation.

- [ ] Create `DynamicForceMacroAction(ActionTerm)` with action dimension one and explicit lifecycle `idle -> running -> boundary_ready -> idle`.
- [ ] Interpret a valid integral scalar as one of the six IDs. While running, repeated copies of the same ID from the 60 Hz runner must not restart the substep counter.
- [ ] On every `apply_actions()` call, read the live link pose and COM pose, transform both hemisphere centers to world coordinates, recompute `u_cam` and `d_t`, select the current phase, and apply the point forces or their exactly equivalent COM wrench.
- [ ] Increment exactly one macro substep per PhysX substep and expose read-only telemetry for action ID, substep index, phase, point forces, applied force, applied torque, endpoint positions, long axis, and lateral direction.
- [ ] Keep UP force active through substep 239. Do not clear it inside the last `apply_actions()` call.
- [ ] Add `release_after_boundary_capture()` that clears the permanent wrench without stepping physics and changes `boundary_ready` to `idle`.
- [ ] Add `SynchronousMacroRunner.step(action_id)` in the installed package module `robotarm_magnetic_lab/runtime/dynamic_force_macro_runner.py`. It must perform exactly 60 environment steps at four physics substeps each, call the 30 Hz camera/coverage update on unique frames, capture the returned one-second boundary RGB, then call `release_after_boundary_capture()`. Scripts must import this runner; they may not carry a private duplicate.
- [ ] Return a structured transition containing the action ID, start RGB frame ID, boundary RGB frame ID, simulated start/end time, 240-substep trace digest, boundary RGB, and catastrophic-fault status. Do not include privileged pose or coverage in the future Actor observation object.

Required timing tests:

```python
def test_one_macro_has_exactly_240_physics_substeps(fake_env):
    transition = SynchronousMacroRunner(fake_env).step(DynamicForceMacroActionId.HOLD)
    assert fake_env.physics_substeps == 240
    assert transition.simulated_duration_s == pytest.approx(1.0)


def test_up_captures_before_force_release(fake_env):
    SynchronousMacroRunner(fake_env).step(DynamicForceMacroActionId.UP)
    assert fake_env.events[-2:] == ["capture_boundary_rgb", "release_active_wrench"]
```

Run focused pure and simulator tests, then commit the action term, MDP export, package runtime, common script utilities, and tests.

## Task D: Add isolated table and stomach task configurations

**Deliverable:** Two new tasks using the same controller and parameters.

- [ ] Register `Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0` and `Template-Robotarm-Magnetic-Dynamic-Force-Macro-Stomach-Lab-v0` without changing any existing task ID.
- [ ] Build the table task from the delivered table scene and the stomach task from the TASK-003 stomach scene placement.
- [ ] Give both tasks only the one-dimensional `dynamic_force_macro` action, RGB policy observation, reset event, and timeout termination. Remove magnetic, joint, ideal-surface, latch, and virtual-magnet action terms.
- [ ] Use one environment, `sim.dt=1/240`, `decimation=4`, `render_interval=4`, capsule-camera `update_period=1/30`, CPU PhysX, scene CCD, and body CCD.
- [ ] Keep the Actor-facing observation RGB-only. Privileged telemetry remains accessible through the evaluator and action-term properties, not the observation group.
- [ ] Add static tests for registrations, actions, clocks, CPU PhysX, camera rate, no rewards, no action mask, no magnetic bridge, and no forbidden runtime state writer.

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force_macro/test_task_cfg.py tests/dynamic_force_macro/test_action_term.py tests/dynamic_force_macro/test_sync_protocol.py -q
```

Commit the task configurations only after these tests pass.

## Task E: Implement deterministic flat calibration and held-out acceptance

**Deliverable:** Reproducible force search and independent final results for all five non-HOLD actions.

- [ ] Generate and persist separate calibration and held-out manifests. Each manifest contains 20 valid reset seeds per action with safe XY, uniform yaw, uniform long-axis roll, zero initial velocities, and no near-world-vertical long axis.
- [ ] Apply reset pose and velocity writes only before the trial begins. Run zero active force until a finite stable contact state is observed or the setup timeout expires; an invalid setup is resampled and does not consume one of the 20 trials.
- [ ] Evaluate `0.9`, then successive `*1.25` ratios, and finally `3.0` if needed. After the first pass, perform exactly three deterministic midpoint refinements between the nearest lower failure and first passing value.
- [ ] Tune MOVE as one shared positive/negative group, VIEW as one shared positive/negative group, and UP independently. A group passes only when every member action reaches at least `16/20` and no candidate trial has FAULT.
- [ ] Implement MOVE evaluation from `t=0.2` to `t=1.0` using the frozen force-onset direction and the `5 mm` projection threshold.
- [ ] Implement VIEW evaluation from `t=0.2` to `t=1.0` using the signed `15 deg` formula in the design.
- [ ] Implement UP evaluation at `t=1.0`, require at least `45 deg`, continuously detect crossing beyond world vertical, and verify that the boundary frame preceded force release.
- [ ] Implement HOLD diagnostics proving 240 zero-active-force substeps; do not include HOLD in force search.
- [ ] Classify only non-finite state, solver interruption, complete table traversal, or escape from the derived table region as FAULT. Store normal failures separately by reason.
- [ ] Select the profile on calibration data, freeze its JSON digest, and run one fresh 20-trial held-out set per non-HOLD action. Never return to calibration after observing held-out outcomes.

Run:

```bash
./run_isaaclab.sh -p scripts/dynamic_force_macro/calibrate_validate_table.py --task Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0 --calibration_samples 20 --held_out_samples 20 --initial_ratio 0.9 --growth 1.25 --max_ratio 3.0 --refinement_rounds 3 --headless
```

The summary must print one line per action in the form `TASK008_HELD_OUT action=<name> success=<n>/20 faults=<n> status=<PASS|FAIL>` and a final `TASK008_TABLE_ACCEPTANCE_PASS` only when all five actions pass.

Commit calibration code and pure protocol tests before running the expensive live campaign. Keep generated manifests, traces, images, and summaries outside Git.

## Task F: Add 30 Hz normal-aware coverage without changing old defaults

**Deliverable:** TASK-008 coverage uses FOV, 50 mm range, first-hit occlusion, and a camera-facing hit-face normal gate on every unique 30 Hz frame.

- [ ] Extend `MeshInput` and `ReferenceMesh` in `coverage/reference_mesh.py` with the authored USD orientation needed to interpret triangle winding, and include that orientation in the deterministic geometry hash.
- [ ] Add pure triangle-normal calculation and `camera_facing_first_hits(...)` to `coverage/visibility.py`.
- [ ] Correct triangle winding according to the authored USD mesh orientation before calculating the face normal.
- [ ] Define camera-facing as a first-hit face normal whose dot product with the camera-to-hit ray direction is strictly negative outside a small numerical tolerance.
- [ ] Keep every existing public function's default behavior unchanged. Add a TASK-008 runtime option `require_camera_facing_normal=True`; old P0 callers default to `False`.
- [ ] Decouple coverage raycast device from physics device. TASK-008 may use CPU PhysX and `cuda:0` Warp raycasts simultaneously.
- [ ] Set runtime metadata to physics `240`, environment/render `60`, camera/coverage `30`, and Actor `1` Hz.
- [ ] Call `maybe_update()` throughout each macro so all unique 30 Hz frames contribute to the cumulative union. Never send coverage mask, visible indices, or pose truth to the Actor observation.

Required pure tests in `tests/coverage/test_reference_mesh.py` and `tests/coverage/test_visibility_geometry.py` must cover orientation preservation, geometry-hash sensitivity to orientation, front-facing acceptance, back-facing rejection, grazing tolerance, winding correction, first-hit occlusion, 50 mm inclusivity, duplicate frame deduplication, and monotonic accumulation.

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/coverage tests/dynamic_force_macro/test_stomach_views.py -q
./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all
```

Commit the backward-compatible coverage extension after both existing and new coverage tests pass.

## Task G: Build the one-key/one-action stomach inspection UI

**Deliverable:** The unchanged flat-selected profile runs in the stomach with three simultaneous synchronized views.

- [ ] Implement rising-edge keyboard input with `Space=HOLD`, `D/A=MOVE_POS/MOVE_NEG`, `E/Q=VIEW_POS/VIEW_NEG`, `W=UP`, `Backspace=reset`, `F12=snapshot`, and `Escape=exit`.
- [ ] Ignore key repeat and ignore additional action requests while a macro is running.
- [ ] Run one full macro after a valid key press. After boundary capture and wrench release, stop physics stepping and continue Kit/UI updates until the next key press.
- [ ] Show the default external viewport, the live capsule RGB panel, and the isolated coverage panel simultaneously in the same Kit application. Display current action, phase, simulated time, and cumulative coverage percentage.
- [ ] Reset capsule state and coverage together on Backspace. Save synchronized external, RGB, coverage, telemetry, and mask artifacts on F12.
- [ ] Load only the frozen flat-selected profile. Do not search, tune, or silently substitute stomach-specific force ratios.

Run the scripted UI smoke first:

```bash
./run_isaaclab.sh -p scripts/dynamic_force_macro/teleop_stomach.py --task Template-Robotarm-Magnetic-Dynamic-Force-Macro-Stomach-Lab-v0 --profile <absolute-flat-profile-json> --scripted_actions 0,1,2,3,4,5 --max_actions 6 --viz kit
```

Then launch the human inspection command without `--scripted_actions`. Linux verifies that all panels update and reports subjective usefulness as unverified pending Windows review.

## Task H: Regression, documentation, and return report

**Deliverable:** Reviewable code, reproducible evidence, and an unambiguous Linux disposition.

- [ ] Run compile checks, all TASK-003 tests, all new TASK-008 tests, all coverage tests, coverage geometry validation, and existing ideal-surface/action-layer smoke tests that are present on the TASK-003 baseline.
- [ ] Scan all new runtime files for pose/velocity setters and all new policy-observation configuration for privileged truth.
- [ ] Run `git diff --check` and confirm that no asset, prior report, VLM, RL, magnetic, latch, virtual-magnet, or ideal-surface file changed unexpectedly.
- [ ] Write `docs/DYNAMIC_FORCE_MACRO_CONTROLLER.md` with action IDs, equations, timings, keyboard mapping, calibration command, held-out command, stomach command, output schema, and known limitations.
- [ ] Write `handoffs/reports/TASK-008-six-action-dynamic-force-controller-report.md` with exact commits, commands, observed results, selected profile digest, all candidate and held-out counts, FAULT taxonomy, boundary ordering evidence, coverage rate evidence, view evidence, deviations, and unverified claims.
- [ ] Record every external artifact's absolute path, byte size, and SHA-256. Do not commit generated logs, videos, screenshots, profiles, or datasets.

The minimum final verification commands are:

```bash
./run_isaaclab.sh -p -m compileall -q scripts/dynamic_force_macro source/robotarm_magnetic_lab/robotarm_magnetic_lab
./run_isaaclab.sh -p -m pytest tests/dynamic_force tests/dynamic_force_macro tests/coverage -q --disable-warnings
./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all
git diff --check
git status --short
```

Push `feature/TASK-008-six-action-dynamic-force-controller` without merging. Return `complete` only under the design's completion rule; return `partial` with full evidence after an exhausted authorized search if a flat or launcher gate remains unmet.
