# TASK-004 Simulation-First Continuation Plan

> **Execution note:** Execute this plan directly in the Linux VS Code Codex session. No external Codex skill, plugin, subagent, or orchestration command is required.

**Goal:** Continue the existing TASK-004 implementation, replace the physically constrained failed rise controller with a simulation-authority endpoint-wrench controller, pass all four flat primitives, and migrate the identical frozen profile to the stomach scene.

**Starting implementation:** `origin/feature/TASK-004-local-dynamics-primitives` at `2bce0d2`

**Revised specification:** `docs/superpowers/specs/2026-08-18-local-dynamics-primitives-simulation-first-revision.md`

**Required branch:** Continue `feature/TASK-004-local-dynamics-primitives`; do not recreate or merge it.

## Global Constraints

- Read the revised specification, this plan, active TASK-004 contract, and existing report completely before editing.
- Preserve existing TASK-004 commits, 28 passing tests, calibration evidence, and first `partial` report history.
- Do not limit force or torque by weight, inertia, magnetic capability, hardware capability, or physical realism.
- Keep the capsule non-kinematic and use only wrench commands plus PhysX outside reset.
- Keep the four-float command protocol, world-axis semantics, four primitives, and strict action time below 10 seconds.
- Keep gravity, contact, CCD, 240 Hz physics, 60 Hz environment/render cadence, and 30 Hz capsule camera.
- Do not write capsule state, command robot/magnet, alter assets/physics, add clearance/avoidance, or add stomach-specific control.
- Use the same tracked simulation profile and SHA-256 in flat and stomach tasks.
- Enforce only the numerical guards `5.0 N`, `0.02 N m`, `50.0 N/s`, and `0.2 N m/s`.

## Existing Code to Preserve

Preserve the pure trajectory, pulse decoder, action term, flat environment, preflight, validator, coordinate correction using `root_link_pose_w`, and all prior evidence. Modify only controller/config/types and necessary tests first; then add calibration/profile support, stomach wrapper, launcher, and report continuation.

### Revision Task 1: Add the Shared Simulation Authority Profile

**Files:**
- Create: `configs/local_primitives/simulation_profile.json`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/config.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/types.py`
- Modify: `tests/local_primitives/test_types_and_config.py`

**Interfaces:**
- Produce `load_simulation_profile(path: Path | None = None) -> SimulationAuthorityProfile`.
- Produce `simulation_profile_sha256(path: Path | None = None) -> str`.
- Preserve `make_local_primitive_controller_cfg()` while sourcing authority fields from the tracked JSON.

- [ ] **Step 1: Write failing schema, digest, and numerical-envelope tests**

```python
def test_simulation_profile_is_tracked_and_not_weight_limited():
    profile = load_simulation_profile()
    assert profile.pose_torque_limit_nm >= 1.0e-4
    assert profile.total_force_limit_n <= 5.0
    assert profile.total_torque_limit_nm <= 0.02
    assert profile.force_slew_limit_n_per_s <= 50.0
    assert profile.torque_slew_limit_nm_per_s <= 0.2
    assert len(simulation_profile_sha256()) == 64


def test_profile_accepts_unrealistic_but_numerically_allowed_values(tmp_path):
    path = write_profile(tmp_path, total_force_limit_n=2.0, total_torque_limit_nm=0.01)
    profile = load_simulation_profile(path)
    assert profile.total_force_limit_n == 2.0
    assert profile.total_torque_limit_nm == 0.01
```

- [ ] **Step 2: Run the focused test and confirm missing-profile failure**

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_types_and_config.py -q
```

Expected: FAIL on missing profile loader or JSON.

- [ ] **Step 3: Add the initial profile**

```json
{
  "schema_version": "task004_simulation_authority_v1",
  "axis_kp_nm_per_rad": 0.001,
  "axis_kd_nms_per_rad": 0.00008,
  "roll_damping_nms_per_rad": 0.00002,
  "pose_torque_limit_nm": 0.001,
  "anchor_kp_n_per_m": 10.0,
  "anchor_kd_ns_per_m": 0.4,
  "endpoint_pin_force_n": 0.1,
  "total_force_limit_n": 1.0,
  "total_torque_limit_nm": 0.005,
  "force_slew_limit_n_per_s": 20.0,
  "torque_slew_limit_nm_per_s": 0.05,
  "motion_duration_s": [5.5, 4.5, 3.5, 8.0],
  "hard_timeout_s": [8.0, 7.0, 6.0, 9.5]
}
```

- [ ] **Step 4: Implement strict loading and canonical SHA-256**

Reject missing/extra keys, nonfinite values, values outside the numerical envelope, and invalid durations. Hash canonical sorted compact JSON. Do not compare values with `mg`, inertia, magnetic moment, or old limits.

- [ ] **Step 5: Extend telemetry types**

Add immutable fields for pose torque, endpoint force, endpoint equivalent torque, total force/torque, saturation flags, slew-limit flags, and profile digest.

- [ ] **Step 6: Run tests and commit**

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_types_and_config.py tests/local_primitives/test_controller.py -q
git add configs/local_primitives/simulation_profile.json source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/config.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/types.py tests/local_primitives/test_types_and_config.py tests/local_primitives/test_controller.py
git commit -m "feat: add simulation authority profile"
```

### Revision Task 2: Implement the Equivalent Non-Camera Endpoint Wrench

**Files:**
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/controller.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/local_primitive_action.py`
- Modify: `tests/local_primitives/test_controller.py`
- Modify: `tests/local_primitives/test_action_command.py`

**Interfaces:**
- Preserve `LocalPrimitiveController.start()` and `update()`.
- Add `non_camera_endpoint_state()` and `compose_endpoint_wrench()` pure helpers.
- Continue applying one total world COM wrench through the existing action term.

- [ ] **Step 1: Write failing endpoint kinematics and wrench-equivalence tests**

```python
def test_non_camera_endpoint_uses_rigid_body_kinematics():
    state = state_with_axis([1, 0, 0], position=[0.1, 0.2, 0.03], angular_velocity=[0, 2, 0])
    endpoint = non_camera_endpoint_state(state, 0.0125)
    np.testing.assert_allclose(endpoint.offset_world_m, [-0.0125, 0, 0])
    np.testing.assert_allclose(endpoint.position_world_m, [0.0875, 0.2, 0.03])
    np.testing.assert_allclose(endpoint.velocity_world_m_s, state.linear_velocity_world_m_s + np.cross(state.angular_velocity_world_rad_s, endpoint.offset_world_m))


def test_endpoint_force_converts_to_equivalent_com_wrench():
    offset = np.array([-0.0125, 0, 0])
    endpoint_force = np.array([0, 0, -0.1])
    force, torque = compose_endpoint_wrench(offset, endpoint_force, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(force, endpoint_force)
    np.testing.assert_allclose(torque, np.cross(offset, endpoint_force))
```

- [ ] **Step 2: Write failing authority and slew tests**

```python
def test_controller_can_exceed_old_torque_limit():
    controller = started_rise_controller(pose_torque_limit_nm=0.001, total_torque_limit_nm=0.005)
    _, telemetry = advance_to_elapsed(controller, side_state(), 2.75)
    assert np.linalg.norm(telemetry.total_torque_world_nm) > 3.0e-5
    assert np.linalg.norm(telemetry.total_torque_world_nm) <= 0.005


def test_total_wrench_obeys_slew_limits():
    controller = started_rise_controller()
    first, _ = controller.update(side_state(), 1 / 240)
    second, _ = controller.update(side_state(), 1 / 240)
    assert np.linalg.norm(second.force_world_n - first.force_world_n) <= 20 / 240 + 1e-12
    assert np.linalg.norm(second.torque_world_nm - first.torque_world_nm) <= 0.05 / 240 + 1e-12
```

- [ ] **Step 3: Implement endpoint force and equivalent torque**

```python
endpoint = non_camera_endpoint_state(state, self.cfg.capsule_half_total_length_m)
force_xy = self.cfg.anchor_kp_n_per_m * (self._anchor_xy - endpoint.position_world_m[:2]) - self.cfg.anchor_kd_ns_per_m * endpoint.velocity_world_m_s[:2]
endpoint_force = np.array([force_xy[0], force_xy[1], -self.cfg.endpoint_pin_force_n])
endpoint_torque = np.cross(endpoint.offset_world_m, endpoint_force)
total_force = endpoint_force + com_damping_force
total_torque = pose_torque + endpoint_torque
```

Clip total force/torque to profile limits, then vector-slew-limit against the previous command. Do not add height targets, normals, geometry queries, task-ID branches, or state correction.

- [ ] **Step 4: Keep the existing COM API and add profile diagnostics**

Apply only the already composed total wrench with `positions=None` and `is_global=True`. Expose profile SHA-256 and component telemetry in the action term and preflight.

- [ ] **Step 5: Run tests, forbidden scan, and commit**

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_controller.py tests/local_primitives/test_action_command.py -q
python -c "from pathlib import Path; p=Path('source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab'); s='\n'.join(x.read_text() for x in list((p/'controllers/local_primitives').glob('*.py'))+[p/'mdp/local_primitive_action.py']); assert all(k not in s for k in ('write_root_pose','write_root_velocity','set_transforms','set_velocities','surface_mesh','clearance','raycast'))"
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/controller.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/local_primitive_action.py tests/local_primitives/test_controller.py tests/local_primitives/test_action_command.py
git commit -m "feat: add non-camera endpoint wrench control"
```

### Revision Task 3: Calibrate Flat Simulation Authority

**Files:**
- Create: `scripts/local_primitives/calibrate_simulation_authority.py`
- Create: `tests/local_primitives/test_authority_calibration.py`
- Modify: `scripts/local_primitives/validate_local_primitives_flat.py`
- Output outside Git: `logs/local_primitives_sim_authority/<timestamp>/attempts.jsonl`

**Interfaces:**
- Execute the specification’s candidate grids from identical resets.
- Deterministically update the tracked JSON only after a passing candidate.

- [ ] **Step 1: Write failing grid-order and selector tests**

```python
def test_selector_chooses_lowest_authority_pass():
    records = [record(1e-4, 0.1, "fail", 8.0), record(3e-4, 0.1, "pass", 5.0), record(3e-4, 0.05, "pass", 6.0)]
    selected = select_candidate(records)
    assert selected.pose_torque_limit_nm == pytest.approx(3e-4)
    assert selected.endpoint_pin_force_n == pytest.approx(0.05)
```

- [ ] **Step 2: Implement reset-isolated candidate execution**

For every candidate, create/reset the flat environment, pulse only `SIDE_TO_UPRIGHT`, and step until success, 8-second timeout, nonfinite state, or center displacement over 5 mm per physics step. Record tilt, endpoint contacts, velocity, component wrenches, saturation, slew, and reason.

- [ ] **Step 3: Implement automatic second-grid expansion and profile write**

Try the first grid exactly. If none passes, try torque `[0.01, 0.02]` and pin force `[1.0, 2.0]`. Canonically write the selected profile and SHA-256. Never edit assets, physics, or reset pose.

- [ ] **Step 4: Run calibration and focused rise validation**

```bash
./run_isaaclab.sh -p scripts/local_primitives/calibrate_simulation_authority.py --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 --seed 42 --write_selected_profile configs/local_primitives/simulation_profile.json --headless
./run_isaaclab.sh -p scripts/local_primitives/validate_local_primitives_flat.py --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 --seed 42 --only SIDE_TO_UPRIGHT --direction_azimuth_deg 0 --headless
```

Expected: `SIMULATION_AUTHORITY_SELECTED`; rise succeeds before 8 seconds with upright error at most 3 degrees, stable hold, finite state, continuity, and no camera-hemisphere load support.

- [ ] **Step 5: Test and commit**

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_authority_calibration.py tests/local_primitives/test_flat_summary.py -q
git add configs/local_primitives/simulation_profile.json scripts/local_primitives/calibrate_simulation_authority.py scripts/local_primitives/validate_local_primitives_flat.py tests/local_primitives/test_authority_calibration.py tests/local_primitives/test_flat_summary.py
git commit -m "tune: select simulation primitive authority"
```

### Revision Task 4: Pass and Freeze All Four Flat Primitives

**Files:**
- Modify when measurements require: `configs/local_primitives/simulation_profile.json`
- Modify only for a demonstrated defect: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/controller.py`
- Modify: `scripts/local_primitives/validate_local_primitives_flat.py`
- Modify: `tests/local_primitives/test_flat_summary.py`

- [ ] **Step 1: Run all four sequences with one unchanged profile**

```bash
./run_isaaclab.sh -p scripts/local_primitives/validate_local_primitives_flat.py --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 --seed 42 --direction_azimuth_deg 0 --headless
```

Expected: four successes, every time below its timeout and 10 seconds, correct support, and passing cone coverage/RMSE.

- [ ] **Step 2: If a later primitive fails, change one category and rerun all four**

Use this order: pose gains, endpoint anchor gains, endpoint pin force, total limits, slew limits, then motion duration. Stay inside the numerical envelope, log every attempt, and rerun every sequence after each change. Do not tune physics or bypass start gates.

- [ ] **Step 3: Test that physical implausibility is not a failure**

```python
def test_unrealistic_but_allowed_wrench_does_not_fail(valid_flat_summary):
    summary = valid_flat_summary()
    summary["wrench"]["max_force_n"] = 4.9
    summary["wrench"]["max_torque_nm"] = 0.019
    assert evaluate_flat_summary(summary)["status"] == "pass"


def test_numerical_discontinuity_still_fails(valid_flat_summary):
    summary = valid_flat_summary()
    summary["continuity"]["max_physics_step_displacement_m"] = 0.0051
    assert evaluate_flat_summary(summary)["status"] == "fail"
```

- [ ] **Step 4: Freeze and commit the passing profile**

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives -q --disable-warnings
git add configs/local_primitives/simulation_profile.json scripts/local_primitives/validate_local_primitives_flat.py tests/local_primitives
git commit -m "test: pass four flat simulation primitives"
```

Record the frozen SHA-256 and do not modify the profile after this commit.

### Revision Task 5: Create the Identical-Profile Stomach Task and Launcher

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_local_primitives_stomach_env_cfg.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Create: `scripts/local_primitives/teleop_local_primitives.py`
- Modify: `tests/local_primitives/test_task_cfg.py`
- Create: `tests/local_primitives/test_no_stomach_adaptation.py`
- Create: `tests/local_primitives/test_launcher_protocol.py`

- [ ] **Step 1: Write failing digest and reset tests**

```python
def test_flat_and_stomach_share_exact_profile():
    flat = flat_cfg_type()().actions.local_primitive
    stomach = stomach_cfg_type()().actions.local_primitive
    assert flat.controller_cfg_values == stomach.controller_cfg_values
    assert flat.profile_sha256 == stomach.profile_sha256 == simulation_profile_sha256()


def test_stomach_preserves_task003_reset():
    task003 = RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg()
    stomach = stomach_cfg_type()()
    assert stomach.scene.stomach.init_state.pos == task003.scene.stomach.init_state.pos
    assert stomach.scene.stomach.init_state.rot == task003.scene.stomach.init_state.rot
    assert stomach.scene.capsule.init_state.pos == task003.scene.capsule.init_state.pos
    assert stomach.scene.capsule.init_state.rot == task003.scene.capsule.init_state.rot
```

- [ ] **Step 2: Implement the stomach scene-only wrapper**

Extend the TASK-003 dynamic stomach configuration, replace only its action group with `local_primitive = make_local_primitive_action_cfg()`, and preserve scene/reset/device/CCD/contact/camera/timing. Do not override profile fields or query geometry.

- [ ] **Step 3: Implement one continuous launcher for both tasks**

Support keys `1/2/3/4`, reset, snapshot, exit, scripted sequences, direction azimuth, capsule-camera view, and both TASK-004 IDs. Emit a one-step start pulse and continue `env.step()` with zero pulse while running/holding. Log digest and component wrench telemetry.

- [ ] **Step 4: Run isolation tests and rendered flat smoke**

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives/test_task_cfg.py tests/local_primitives/test_no_stomach_adaptation.py tests/local_primitives/test_launcher_protocol.py -q
./run_isaaclab.sh -p scripts/local_primitives/teleop_local_primitives.py --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 --scripted_sequence "0,1;reset;0,2;reset;0,2,3" --direction_azimuth_deg 0 --capsule_camera_view --viz kit
```

- [ ] **Step 5: Run unchanged-profile stomach rendering**

```bash
./run_isaaclab.sh -p scripts/local_primitives/teleop_local_primitives.py --task Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0 --scripted_sequence "0,1;reset;0,2;reset;0,2,3" --direction_azimuth_deg 0 --capsule_camera_view --viz kit
```

Expected: identical digest, continuous external/capsule views, and recorded outcomes. Collision alone is allowed. Do not change the profile after this run.

- [ ] **Step 6: Commit**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_local_primitives_stomach_env_cfg.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py scripts/local_primitives/teleop_local_primitives.py tests/local_primitives/test_task_cfg.py tests/local_primitives/test_no_stomach_adaptation.py tests/local_primitives/test_launcher_protocol.py
git commit -m "feat: migrate simulation primitives unchanged to stomach"
```

### Revision Task 6: Regress, Update the Existing Report, and Push

**Files:**
- Modify: `docs/LOCAL_DYNAMICS_PRIMITIVES.md`
- Modify: `docs/PROJECT_RUN_LOG.md`
- Modify: `handoffs/reports/TASK-004-local-dynamics-primitives-report.md`

- [ ] **Step 1: Run all required regressions**

```bash
./run_isaaclab.sh -p -m pytest tests/local_primitives tests/dynamic_force -q --disable-warnings
./run_isaaclab.sh -p -m pytest tests/ideal_surface tests/coverage tests/action_layer/test_atomic_protocol.py tests/action_layer/test_executor.py tests/action_layer/test_safety.py tests/action_layer/test_atomic_stomach_teleop_cfg.py tests/action_layer/test_atomic_keyboard_protocol.py -q --disable-warnings
./run_isaaclab.sh -p scripts/dynamic_force/inspect_dynamic_force_prerequisites.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 --num_envs 1 --headless
./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py --num_envs 1 --coverage_samples 5
./run_isaaclab.sh -p -m compileall -q scripts/local_primitives source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab
git diff --check
```

- [ ] **Step 2: Append to the existing report**

Preserve the first failed attempt. Append revision authority, every expanded calibration attempt, selected profile/digest, four flat metrics, stomach outcomes, rendering evidence, regressions, deviations, and external artifact bytes/hashes. State that wrench values may be physically unrealistic.

- [ ] **Step 3: Apply disposition correctly**

Use `complete` only when flat passes, stomach uses the same digest, regressions pass, and rendered stomach evidence exists. Use `partial` if flat passes but a stomach target times out. Use `needs_decision` only if interface/numerical failure persists at `5 N / 0.02 N m`, never for physical implausibility.

- [ ] **Step 4: Commit and push without merging**

```bash
git add docs/LOCAL_DYNAMICS_PRIMITIVES.md docs/PROJECT_RUN_LOG.md handoffs/reports/TASK-004-local-dynamics-primitives-report.md
git commit -m "docs: report simulation-first local primitives"
git diff --check
git status --short
git push -u origin feature/TASK-004-local-dynamics-primitives
```

## Final Manual Check

Confirm that the final profile is unconstrained by physical realism but inside numerical guards; every motion comes from a wrench and PhysX; no state setter or kinematic switch exists; rise biases the non-camera endpoint; every action is below 10 seconds; no clearance/avoidance exists; flat and stomach digests match; TASK-003 stomach reset is unchanged; and the first failed report evidence remains.
