# TASK-007-R1 Generalized-Wrench and Surface-Frame Controller Implementation Plan

> **Linux executor:** This plan is self-contained. Execute it manually, task by task. The Linux session is not required to install or invoke `superpowers:subagent-driven-development`, `superpowers:executing-plans`, or any other skill. The `docs/superpowers` path is historical repository organization only.

**Goal:** Correct TASK-007's directional VIEW asymmetry without relaxing its one-second action or stability contracts, then complete the previously unrun flat, disturbance, stomach, sequence, and visualization gates.

**Architecture:** The finite-magnet inverse is reformulated in generalized-wrench coordinates about the capsule center of mass while PhysX still receives force and magnetic couple torque at the capsule magnet center. A frozen surface-action frame replaces the full capsule-rotation nominal pose, and one prescribed-time terminal governor handles all VIEW directions. A deterministic multi-candidate search selects one shared profile before new held-out validation.

**Tech stack:** Python 3.11, NumPy, SciPy, PyTorch, Magpylib 5.2.3, Isaac Lab 2.3.2, Isaac Sim 5.0.0, GPU PhysX, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-task007-r1-generalized-wrench-surface-frame-design.md`

## Global constraints

- Fetch `origin/workflow/TASK-007-R1-generalized-wrench-surface-frame`, create `feature/TASK-007-R1-generalized-wrench-surface-frame`, and verify that the planning commit's parent is exactly `29c459637b0d36c8a289ed2f8553e8e896120aa3`.
- Do not merge `main`, TASK-005, TASK-006, or rewrite TASK-007 history. Preserve `configs/virtual_magnet/closed_loop_v1.json` as the old-profile baseline.
- Keep the capsule dynamic and gravity enabled. Never write capsule pose/velocity during an action, lock DOFs, teleport, project, create a temporary joint, or directly apply the desired generalized wrench.
- Apply only finite-model magnetic force and couple torque at the capsule magnet center. Use the equivalent COM wrench for inverse/control calculations only; never double-count its lever-arm torque in PhysX.
- Keep eleven IDs, three results, 240/60/1 Hz, 240 substeps, 15-degree VIEW, 5 mm MOVE, 45-degree MOVE eligibility, and all existing acceptance thresholds unchanged.
- Keep one shared R1 profile with no scene or direction-specific branches, gains, offsets, thresholds, timings, or limits.
- Keep large logs/videos outside Git. Commit compact JSON summaries, seed manifests, candidate tables, digests, code, tests, and the final report.
- Run dependency-light tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Run live tests through `./run_isaaclab.sh -p` on `cuda:0`.
- A failed development trial is evidence for diagnosis and search. Do not return after one profile or one direction fails.

---

### Task 1: Preserve and reproduce the TASK-007 directional baseline

**Files:**

- Create: `configs/virtual_magnet/closed_loop_r1.json`
- Create: `scripts/virtual_magnet/diagnose_directional_covariance.py`
- Create: `handoffs/reports/TASK-007-R1-baseline-summary.json`
- Create: `tests/virtual_magnet/test_r1_profile_contract.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/config.py`

**Interfaces:**

- Produces `load_profile(path) -> ClosedLoopProfile` support for both immutable V1 and new schema `task007_r1_generalized_wrench_surface_frame_v1`.
- Produces an eight-direction diagnostic JSON containing one row per action/seed and complete wrench/terminal telemetry.

- [ ] **Step 1: Create and verify the branch**

```bash
git fetch origin workflow/TASK-007-R1-generalized-wrench-surface-frame
git switch --create feature/TASK-007-R1-generalized-wrench-surface-frame \
  origin/workflow/TASK-007-R1-generalized-wrench-surface-frame
git rev-parse HEAD
git rev-parse HEAD^
git status --short --branch
```

Expected: `HEAD^` equals `29c459637b0d36c8a289ed2f8553e8e896120aa3` and the worktree is clean.

- [ ] **Step 2: Preserve external baseline evidence without inventing missing data**

Verify every path and hash listed in the TASK-007 report. Copy existing compact `/tmp/task007-*.json` files into a durable external directory under `/mnt/isaac-linux/robotarm_magnetic_lab/logs/task007_baseline/`. Record source path, destination path, bytes, and SHA-256 in `TASK-007-R1-baseline-summary.json`. For a missing path, record `present=false` and its expected hash; do not reconstruct it.

- [ ] **Step 3: Write the failing R1 profile test**

```python
def test_v1_remains_immutable_and_r1_has_no_direction_or_scene_keys():
    v1 = load_profile("configs/virtual_magnet/closed_loop_v1.json")
    r1 = load_profile("configs/virtual_magnet/closed_loop_r1.json")
    assert v1.schema_version == "task007_virtual_magnet_closed_loop_v1"
    assert r1.schema_version == "task007_r1_generalized_wrench_surface_frame_v1"
    assert r1.physics_hz == 240
    assert r1.feedback_hz == 60
    assert r1.action_substeps == 240
    direction_key = re.compile(r"(^|_)(up|down|left|right)($|_)")
    assert not any(direction_key.search(field.name.lower()) for field in fields(r1))
    assert not any("flat" in field.name.lower() or "stomach" in field.name.lower() for field in fields(r1))
```

- [ ] **Step 4: Confirm failure, copy V1 values into R1, and add only R1 shared fields**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_r1_profile_contract.py -q
```

Add `nominal_position_surface_m`, `nominal_quaternion_surface_xyzw`, `terminal_max_angular_accel_rad_s2`, `terminal_max_torque_nm`, `contact_moment_compensation_gain`, `contact_moment_filter_time_constant_s`, and deterministic search bounds. Set compensation gain to zero for the first diagnostic. Preserve every frozen timing and acceptance value.

- [ ] **Step 5: Implement and run the old-profile eight-direction reproduction**

```bash
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/diagnose_directional_covariance.py \
  --profile configs/virtual_magnet/closed_loop_v1.json \
  --samples-per-view 1 --seed 70071 --device cuda:0 --headless \
  --output logs/task007_r1/baseline_directional.json
```

Expected: reproduce the old directional pattern closely enough to identify the same passing/failing action classes. Record numerical differences; do not require bit identity from GPU PhysX.

- [ ] **Step 6: Commit baseline preservation**

```bash
git add configs/virtual_magnet/closed_loop_r1.json \
  scripts/virtual_magnet/diagnose_directional_covariance.py \
  tests/virtual_magnet/test_r1_profile_contract.py \
  handoffs/reports/TASK-007-R1-baseline-summary.json \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/config.py
git commit -m "test: preserve TASK-007 directional baseline"
```

---

### Task 2: Express the finite-magnet inverse about the capsule COM

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/generalized_wrench.py`
- Create: `tests/virtual_magnet/test_generalized_wrench.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/types.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/pose_inverse.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/controller.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/virtual_magnet_bridge.py`
- Create: `scripts/virtual_magnet/diagnose_generalized_wrench.py`

**Interfaces:**

- Produces `MagneticWrenchAtPoint`, `GeneralizedWrenchAtCom`, `generalized_wrench_about_com()`, `contact_moment_about_com()`, and `equivalent_application_tuple()`.
- Changes the pose-inverse `WrenchModel` return value to the COM generalized six-vector while the bridge retains the at-point application tuple.

- [ ] **Step 1: Write failing lever-arm tests**

```python
def test_force_at_magnet_center_is_transformed_to_com_torque():
    result = generalized_wrench_about_com(
        force_world_n=np.array([0.0, 2.0, 0.0]),
        couple_torque_world_nm=np.array([0.0, 0.0, 0.5]),
        application_point_world_m=np.array([1.0, 0.0, 0.0]),
        com_position_world_m=np.zeros(3),
    )
    np.testing.assert_allclose(result.force_world_n, [0.0, 2.0, 0.0])
    np.testing.assert_allclose(result.lever_torque_world_nm, [0.0, 0.0, 2.0])
    np.testing.assert_allclose(result.torque_com_world_nm, [0.0, 0.0, 2.5])


def test_physx_tuple_keeps_couple_torque_not_com_torque():
    at_point = MagneticWrenchAtPoint(force_world_n=F, couple_torque_world_nm=tau, point_world_m=p)
    force, couple, point = equivalent_application_tuple(at_point)
    np.testing.assert_allclose(force, F)
    np.testing.assert_allclose(couple, tau)
    np.testing.assert_allclose(point, p)
```

- [ ] **Step 2: Run the focused test and confirm failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_generalized_wrench.py -q
```

- [ ] **Step 3: Implement pure transforms and rotational covariance tests**

For yaw rotations `Rz(k*45 deg)`, require

```python
np.testing.assert_allclose(
    generalized_wrench_about_com(R @ F, R @ tau, R @ p, R @ com).vector,
    np.concatenate((R @ expected.force_world_n, R @ expected.torque_com_world_nm)),
    rtol=1e-10,
    atol=1e-12,
)
```

Use the capsule COM state exposed by Isaac Lab. If `root_com_pos_w` is not exposed in the installed API, derive it from the live root pose and PhysX local COM returned by `root_physx_view.get_coms()`; record the selected source in runtime telemetry. Do not assume root origin equals COM without readback.

- [ ] **Step 4: Change the inverse model and preserve physical application semantics**

The controller's finite-model closure must calculate Magpylib force/couple at the capsule magnet center and transform them to the current COM before returning the six-vector to `numerical_pose_jacobian()` and `solve_pose_increment()`. The bridge must still pass original force, original couple torque, and capsule-magnet application point to `set_forces_and_torques_index()`.

Add the five distinct wrench telemetry fields from the spec. Add a live assertion that recomputing the COM generalized wrench from the applied tuple equals the inverse model's pre-filter representation before controller/passive filters.

- [ ] **Step 5: Run pure and live generalized-wrench diagnostics**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_generalized_wrench.py \
  tests/virtual_magnet/test_pose_inverse.py \
  tests/virtual_magnet/test_controller.py -q
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/diagnose_generalized_wrench.py \
  --profile configs/virtual_magnet/closed_loop_r1.json --device cuda:0 --headless
```

Expected: exact finite-vector readback, no double counting, finite Jacobians, and all eight directions emit correctly transformed torque telemetry. Keep contact-moment compensation disabled.

- [ ] **Step 6: Commit the generalized-wrench correction**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/virtual_magnet_bridge.py \
  scripts/virtual_magnet/diagnose_generalized_wrench.py \
  tests/virtual_magnet/test_generalized_wrench.py \
  tests/virtual_magnet/test_pose_inverse.py \
  tests/virtual_magnet/test_controller.py
git commit -m "fix: solve magnetic wrench about capsule COM"
```

---

### Task 3: Replace capsule-frame regularization with a frozen surface-action frame

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/surface_frame.py`
- Create: `tests/virtual_magnet/test_surface_action_frame.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/types.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/controller.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/virtual_magnet_bridge.py`

**Interfaces:**

- Produces `SurfaceActionFrame`, `build_surface_action_frame()`, `view_preferred_tangent()`, `nominal_magnet_pose_surface()`, and `rotate_frame_covariantly()`.
- Extends `FrozenActionTarget` with immutable `surface_action_rotation_world`.

- [ ] **Step 1: Write failing frame and degeneracy tests**

Test orthonormality, right-handedness, VIEW preferred-tangent selection, MOVE sign reversal, HOLD continuity, camera-right fallback, least-aligned-world-axis fallback, quaternion sign continuity, and common-yaw covariance.

```python
@pytest.mark.parametrize("yaw_deg", range(0, 360, 45))
def test_surface_action_frame_rotates_covariantly(yaw_deg):
    rotation = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    base = build_surface_action_frame(normal=N, preferred_tangent=Q, continuity_tangent=C)
    turned = build_surface_action_frame(
        normal=rotation @ N,
        preferred_tangent=rotation @ Q,
        continuity_tangent=rotation @ C,
    )
    np.testing.assert_allclose(turned.rotation_world, rotation @ base.rotation_world, atol=1e-10)
```

- [ ] **Step 2: Confirm failure and implement the pure frame helpers**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_surface_action_frame.py -q
```

Implement the exact fallback order from the spec. Degeneracy that all fallbacks cannot resolve is a true numerical `FAULT`; ordinary near-alignment uses the next deterministic fallback.

- [ ] **Step 3: Freeze one frame per action and follow only its moving origin**

At submit, compute and store the frame. At each feedback update, compute

```python
nominal_position = state.capsule_com_position + frame.rotation_world @ profile.nominal_position_surface_m
nominal_rotation = frame.rotation_world @ Rotation.from_quat(
    profile.nominal_quaternion_surface_xyzw
).as_matrix()
```

Do not multiply the nominal offset by the current full capsule rotation. Do not recompute frame axes inside the one-second action. Continue smooth pose interpolation and separation/trust limits.

- [ ] **Step 4: Prove pure inverse covariance and live no-teleport behavior**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_surface_action_frame.py \
  tests/virtual_magnet/test_pose_inverse.py \
  tests/virtual_magnet/test_controller.py -q
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/diagnose_directional_covariance.py \
  --profile configs/virtual_magnet/closed_loop_r1.json \
  --samples-per-view 1 --seed 70072 --device cuda:0 --headless \
  --output logs/task007_r1/surface_frame_directional.json
```

Record maximum substep magnet translation/rotation, relative separation, per-direction error, and transformed generalized-wrench covariance residual. No action boundary may jump the debug magnet.

- [ ] **Step 5: Commit the surface-frame correction**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/virtual_magnet_bridge.py \
  tests/virtual_magnet/test_surface_action_frame.py
git commit -m "fix: anchor virtual magnet in surface action frame"
```

---

### Task 4: Add one prescribed-time terminal governor for all directions

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/terminal_governor.py`
- Create: `tests/virtual_magnet/test_terminal_governor.py`
- Modify: `scripts/virtual_magnet/diagnose_directional_covariance.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/types.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/outer_loop.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/controller.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/virtual_magnet_bridge.py`

**Interfaces:**

- Produces `TerminalGovernorState`, `prescribed_time_angular_acceleration()`, `rigid_torque_from_acceleration()`, `estimate_contact_moment()`, and `desired_magnetic_generalized_torque()`.
- Extends `ControllerState` with `capsule_com_position`, `inertia_world`, `angular_drag_torque`, and `estimated_contact_torque_com`.

- [ ] **Step 1: Write failing terminal-boundary tests**

Verify the cubic-boundary formula, one-feedback-interval time floor, twist removal, shared caps, world-inertia transform, gyroscopic term, known-drag subtraction, bounded contact compensation, and direction covariance. Assert that the function accepts no action ID or direction-specific parameter.

```python
def test_prescribed_time_governor_has_no_direction_branch():
    positive = prescribed_time_angular_acceleration(E, W, remaining_s=0.2, max_accel=40.0)
    negative = prescribed_time_angular_acceleration(-E, -W, remaining_s=0.2, max_accel=40.0)
    np.testing.assert_allclose(negative, -positive)


def test_desired_magnetic_torque_subtracts_bounded_passive_estimate():
    result = desired_magnetic_generalized_torque(
        rigid_torque=RIGID,
        angular_drag_torque=DRAG,
        contact_torque=CONTACT,
        contact_gain=0.25,
        contact_limit_nm=LIMIT,
    )
    np.testing.assert_allclose(result, RIGID - DRAG - 0.25 * clipped_contact)
```

- [ ] **Step 2: Implement live COM inertia and contact-moment state**

Read local COM and body inertia from the PhysX rigid view during initialization, verify they are finite and positive definite, and transform inertia to world each substep. Estimate the flat support point for the known spherocapsule using its 13 mm diameter and 25 mm total length; combine it with the measured net contact force to produce `r_contact x F_contact`. Record `source=flat_spherocapsule_support_estimate`. Keep compensation gain zero for the first live comparison.

- [ ] **Step 3: Replace only the fixed final VIEW/HOLD torque cap**

During the first 0.8 seconds, retain quintic tracking. During the final 0.2 seconds, compute the prescribed-time torque from current error, twist-removed angular velocity, remaining time, and live inertia. Convert it to a desired magnetic generalized torque, then use the corrected pose inverse. Do not create per-direction gains and do not control long-axis twist.

For MOVE, retain zero requested magnetic couple torque and passive roll. The final 0.2 seconds continue endpoint force stabilization and configured passive angular drag; do not introduce a tilt or roll target.

- [ ] **Step 4: Run unit tests and paired live ablations**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_terminal_governor.py \
  tests/virtual_magnet/test_outer_loop.py \
  tests/virtual_magnet/test_controller.py -q
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/diagnose_directional_covariance.py \
  --profile configs/virtual_magnet/closed_loop_r1.json \
  --samples-per-view 1 --seed 70073 --ablate-terminal-governor \
  --device cuda:0 --headless \
  --output logs/task007_r1/terminal_governor_ablation.json
```

Each paired row starts from the same state and seed. Report target error and last-0.1-second angular speed with governor disabled and enabled. Enable bounded contact compensation only in an additional paired run after the zero-gain generalized-wrench and surface-frame effects are recorded.

- [ ] **Step 5: Commit the terminal governor**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/virtual_magnet_bridge.py \
  tests/virtual_magnet/test_terminal_governor.py
git commit -m "feat: add shared prescribed-time VIEW stabilization"
```

---

### Task 5: Replace the one-profile calibration stub with a real search

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/calibration_search.py`
- Create: `tests/virtual_magnet/test_calibration_search.py`
- Modify: `scripts/virtual_magnet/calibrate_closed_loop.py`
- Modify: `scripts/virtual_magnet/common.py`
- Modify: `scripts/virtual_magnet/validate_flat.py`
- Modify: `tests/virtual_magnet/test_trial_protocol.py`
- Create: `handoffs/reports/TASK-007-R1-calibration-summary.json`

**Interfaces:**

- Produces `SearchBounds`, `CandidateProfile`, `CandidateResult`, `space_filling_candidates()`, `local_refinement_candidates()`, `rank_candidates()`, and `search_exhausted()`.
- The CLI must evaluate the actual requested candidate count and load each candidate profile without editing the checked-in profile in place.

- [ ] **Step 1: Write failing search tests**

Test deterministic candidate generation, unique profile digests, bound enforcement, no frozen-physics fields, at least 32 first-stage candidates, lexicographic worst-direction ranking, local refinement, early success only after every development class reaches 4/5, and exhaustion only after the required candidate/refinement evidence.

```python
def test_ranking_cannot_hide_one_failed_direction_with_good_mean():
    balanced = candidate_result(class_passes=[4, 4, 4, 4, 4, 4, 4, 4])
    averaged = candidate_result(class_passes=[5, 5, 5, 5, 5, 5, 5, 0])
    assert rank_candidates([averaged, balanced])[0] is balanced


def test_search_budget_is_not_a_noop():
    candidates = space_filling_candidates(BASE, BOUNDS, count=32, seed=7071)
    assert len(candidates) == 32
    assert len({item.sha256 for item in candidates}) == 32
```

- [ ] **Step 2: Implement two-stage candidate generation and immutable evaluation**

Use a deterministic Latin-hypercube first stage with 32 candidates over the authorized R1 fields. Evaluate one matched sample for HOLD and each VIEW as a screening set. Select the best eight finite candidates by the specified lexicographic order, then evaluate five development samples for all eleven valid action classes plus invalid MOVE classes. Generate bounded coordinate-refinement candidates around the best finite profiles until one passes or the total reaches 200.

Add explicit `--profile`, `--split`, `--required-passes`, and `--expected-samples-per-class` arguments to `validate_flat.py`. Change `summarize_trials()` to receive the required count instead of hard-coding 16 of 20, while retaining 20 and 16 as the formal-validation defaults.

- [ ] **Step 3: Run tests and then the live search**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_calibration_search.py \
  tests/virtual_magnet/test_trial_protocol.py \
  tests/virtual_magnet/test_acceptance_summary.py -q
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/calibrate_closed_loop.py \
  --scene flat --profile configs/virtual_magnet/closed_loop_r1.json \
  --first-stage-candidates 32 --finalists 8 --trials-per-action 5 \
  --search-budget 200 --seed 170071 --device cuda:0 --headless \
  --output logs/task007_r1/calibration/search.json
```

Expected: `evaluated_candidates` reflects real distinct digests. Continue within the budget after individual failures. Copy only the selected finite profile to `closed_loop_r1.json`, record its digest and candidate lineage, and rerun its five-sample gate in a fresh process.

- [ ] **Step 4: Run the blocking five-sample development gate**

```bash
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/validate_flat.py \
  --profile configs/virtual_magnet/closed_loop_r1.json \
  --split development-r1-final --trials-per-action 5 --seed 270071 \
  --required-passes 4 --expected-samples-per-class 5 \
  --device cuda:0 --headless \
  --output logs/task007_r1/development/final.json
```

Blocking gate: every valid action class passes at least 4/5 and both invalid-MOVE classes pass at least 4/5. Replace constrained VIEW samples. If it fails, resume search with unused development seeds; do not enter held-out validation.

- [ ] **Step 5: Commit the search and selected shared profile**

```bash
git add configs/virtual_magnet/closed_loop_r1.json \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/calibration_search.py \
  scripts/virtual_magnet/calibrate_closed_loop.py \
  scripts/virtual_magnet/common.py \
  scripts/virtual_magnet/validate_flat.py \
  tests/virtual_magnet/test_calibration_search.py \
  tests/virtual_magnet/test_trial_protocol.py \
  handoffs/reports/TASK-007-R1-calibration-summary.json
git commit -m "tune: select shared R1 closed-loop profile"
```

---

### Task 6: Run new formal flat no-disturbance acceptance

**Files:**

- Modify: `scripts/virtual_magnet/validate_flat.py`
- Modify: `tests/virtual_magnet/test_trial_protocol.py`
- Create: `handoffs/reports/TASK-007-R1-flat-no-disturbance-summary.json`

- [ ] **Step 1: Add explicit split/profile arguments and contamination guards**

The validator must reject a held-out manifest whose seeds appear in calibration/development summaries. It must store selected profile SHA-256, code HEAD, physics snapshot digest, exact class counts, constrained replacements, and source manifest digest.

- [ ] **Step 2: Generate new held-out manifests only after the profile is frozen**

```bash
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/validate_flat.py \
  --mode no-disturbance --profile configs/virtual_magnet/closed_loop_r1.json \
  --split held-out-r1-flat-no-disturbance --trials-per-action 20 \
  --required-passes 16 --expected-samples-per-class 20 \
  --seed 370071 --held-out --device cuda:0 --headless \
  --output logs/task007_r1/held_out/flat_no_disturbance.json
```

Blocking gate: each of eleven valid classes has 20 actual-valid samples and at least 16 passes. Each invalid MOVE sign has 20 samples and at least 16 correct full-second `REJECTED` results. No true `FAULT` is accepted as a pass.

- [ ] **Step 3: Audit and commit only the compact summary**

```bash
git add scripts/virtual_magnet/validate_flat.py \
  tests/virtual_magnet/test_trial_protocol.py \
  handoffs/reports/TASK-007-R1-flat-no-disturbance-summary.json
git commit -m "test: pass R1 flat no-disturbance gate"
```

If this gate fails and its samples inform tuning, invalidate the entire held-out set, return to Task 5, and generate a new held-out split after freezing a new digest.

---

### Task 7: Implement and pass paired flat disturbance validation

**Files:**

- Create: `scripts/virtual_magnet/validate_disturbance.py`
- Create: `tests/virtual_magnet/test_disturbance_protocol.py`
- Create: `handoffs/reports/TASK-007-R1-flat-disturbance-summary.json`

- [ ] **Step 1: Write failing paired-protocol tests**

Test identical initial pose/contact/seed/disturbance between feedback-enabled and feedback-disabled members, finite mid-action force/torque pulse metadata, no disturbance information in controller inputs, command-divergence measurement after the pulse, endpoint recovery comparison, 20 samples per class, and 16/20 gate enforcement.

- [ ] **Step 2: Implement external disturbance injection outside the controller**

Inject a bounded force and couple torque through a separate test-only PhysX term for a recorded finite substep interval. It must not alter the finite-model bridge, controller state, profile, or public action. Remove the disturbance term entirely from non-disturbance and stomach task configurations.

- [ ] **Step 3: Run new paired held-out trials**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_disturbance_protocol.py -q
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/validate_disturbance.py \
  --profile configs/virtual_magnet/closed_loop_r1.json \
  --trials-per-action 20 --seed 470071 --held-out --paired-open-loop \
  --device cuda:0 --headless \
  --output logs/task007_r1/held_out/flat_disturbance.json
```

Blocking gate: each feedback-enabled action class passes at least 16/20. The summary must also show post-disturbance magnet-command divergence and better endpoint recovery than the paired feedback-disabled baseline. Success without command divergence is not proof of closed-loop recovery.

- [ ] **Step 4: Commit compact disturbance evidence**

```bash
git add scripts/virtual_magnet/validate_disturbance.py \
  tests/virtual_magnet/test_disturbance_protocol.py \
  handoffs/reports/TASK-007-R1-flat-disturbance-summary.json
git commit -m "test: prove R1 closed-loop disturbance recovery"
```

---

### Task 8: Integrate the same R1 controller with stomach surface queries

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_virtual_magnet_stomach_env_cfg.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Create: `scripts/virtual_magnet/validate_stomach.py`
- Create: `tests/virtual_magnet/test_stomach_task_cfg.py`
- Create: `handoffs/reports/TASK-007-R1-stomach-summary.json`

- [ ] **Step 1: Write the same-profile and local-surface tests**

Assert that flat and stomach import the same controller class and profile path, have identical action/rate/threshold/controller parameters, and contain no scene-name branch. Test local inward normal orientation, spherocapsule support-point estimation, surface-action-frame construction, and finite mesh queries at named development/held-out regions.

- [ ] **Step 2: Implement the stomach wrapper and development split**

The stomach wrapper changes only scene asset, reset sampler, local surface query, lighting/camera, and evidence labels. It may not add controller parameters. Run named development regions without injected disturbance:

```bash
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/validate_stomach.py \
  --profile configs/virtual_magnet/closed_loop_r1.json \
  --split development --trials-per-action 5 --seed 570071 \
  --device cuda:0 --headless \
  --output logs/task007_r1/stomach/development.json
```

If development changes the shared profile, return to Tasks 5, 6, and 7 with new development and held-out manifests before continuing.

- [ ] **Step 3: Run new held-out stomach regions**

```bash
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/validate_stomach.py \
  --profile configs/virtual_magnet/closed_loop_r1.json \
  --split held-out --trials-per-action 20 --seed 670071 \
  --device cuda:0 --headless \
  --output logs/task007_r1/stomach/held_out.json
```

Blocking gate: every valid class passes at least 16/20 and both invalid-MOVE classes pass at least 16/20. Record constrained VIEWs, solver saturation, contact-moment estimates, passive roll, local normal, and region IDs. No artificial disturbance is allowed.

- [ ] **Step 4: Commit stomach integration and compact evidence**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_virtual_magnet_stomach_env_cfg.py \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py \
  scripts/virtual_magnet/validate_stomach.py \
  tests/virtual_magnet/test_stomach_task_cfg.py \
  handoffs/reports/TASK-007-R1-stomach-summary.json
git commit -m "feat: validate shared R1 controller in stomach"
```

---

### Task 9: Complete continuous sequences, visualization, regressions, and report

**Files:**

- Create: `configs/virtual_magnet/r1_sequence_100.json`
- Create: `scripts/virtual_magnet/validate_sequence.py`
- Modify: `scripts/virtual_magnet/teleop_virtual_magnet.py`
- Create: `tests/virtual_magnet/test_r1_sequence_protocol.py`
- Create: `handoffs/reports/TASK-007-R1-sequence-summary.json`
- Create: `handoffs/reports/TASK-007-R1-generalized-wrench-surface-frame-report.md`
- Modify: `handoffs/reports/README.md`

- [ ] **Step 1: Freeze and test the 100-action sequence**

Use one JSON ID sequence for both scenes. It must contain every ID, opposite VIEW pairs, consecutive MOVE, invalid/rejected MOVE, MOVE-to-VIEW, and MOVE-to-HOLD. Unit tests must verify exact length, coverage, transition coverage, and that only a true `FAULT` truncates execution.

- [ ] **Step 2: Run no-reset flat and stomach sequences**

```bash
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/validate_sequence.py \
  --scene flat --profile configs/virtual_magnet/closed_loop_r1.json \
  --sequence configs/virtual_magnet/r1_sequence_100.json \
  --small-disturbances --device cuda:0 --headless
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/validate_sequence.py \
  --scene stomach --profile configs/virtual_magnet/closed_loop_r1.json \
  --sequence configs/virtual_magnet/r1_sequence_100.json \
  --device cuda:0 --headless
```

Both sequences must process all 100 requests without true `FAULT`, preserve 240-substep boundaries, remain finite, and keep magnet separation bounded. The stomach CLI must reject a disturbance flag.

- [ ] **Step 3: Run full regressions**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/virtual_magnet -q
./run_isaaclab.sh -p scripts/table_motion/test_02_axial_field_scan.py --headless
./run_isaaclab.sh -p scripts/table_motion/test_05_long_axis_roll.py --headless
./run_isaaclab.sh -p scripts/virtual_magnet/diagnose_generalized_wrench.py \
  --profile configs/virtual_magnet/closed_loop_r1.json --device cuda:0 --headless
```

- [ ] **Step 4: Run one-key/one-action visualization in both scenes**

```bash
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/teleop_virtual_magnet.py \
  --scene flat --profile configs/virtual_magnet/closed_loop_r1.json \
  --device cuda:0 --render_fps 120
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/teleop_virtual_magnet.py \
  --scene stomach --profile configs/virtual_magnet/closed_loop_r1.json \
  --device cuda:0 --render_fps 120
```

Preserve one example of every action, one rejected MOVE, measured FPS, terminal telemetry, and video/log hashes. Keep one key press per completed one-second action with no queue or repeat.

- [ ] **Step 5: Write the evidence-grounded final report**

The report must include base/head/branch, exact commands and exit codes, dependency/profile/physics digests, baseline preservation, generalized-wrench equivalence, frame covariance, terminal ablation, evaluated calibration candidates, selected lineage, every 20-sample table, disturbance pairs, stomach split, sequence results, regression counts, warnings, and external paths/bytes/SHA-256. Distinguish unit-test success, development success, formal held-out success, `COMPLETED`, quantitative pass, and passive rolling.

- [ ] **Step 6: Verify, commit, and push without merging**

```bash
git diff --check origin/workflow/TASK-007-R1-generalized-wrench-surface-frame...HEAD
git status --short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/virtual_magnet -q
git add configs/virtual_magnet/r1_sequence_100.json \
  scripts/virtual_magnet/validate_sequence.py \
  scripts/virtual_magnet/teleop_virtual_magnet.py \
  tests/virtual_magnet/test_r1_sequence_protocol.py \
  handoffs/reports/TASK-007-R1-sequence-summary.json \
  handoffs/reports/TASK-007-R1-generalized-wrench-surface-frame-report.md \
  handoffs/reports/README.md
git commit -m "docs: report TASK-007-R1 validation"
git push -u origin feature/TASK-007-R1-generalized-wrench-surface-frame
git status --short --branch
```

Return `complete` only after every R1 gate passes. Return `partial` only when both flat gates pass but stomach cannot reach the unchanged threshold after authorized shared-profile tuning. Return `needs_decision` only after the specified multi-candidate search and structural diagnostics are exhausted. Never merge the implementation branch.
