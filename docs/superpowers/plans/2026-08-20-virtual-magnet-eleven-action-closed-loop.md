# Virtual-Magnet Eleven-Action Closed-Loop Controller Implementation Plan

> **Linux executor:** This plan is self-contained. The Linux session must execute it directly, task by task. It is not required to install or invoke `superpowers:subagent-driven-development`, `superpowers:executing-plans`, or any other skill. The historical `docs/superpowers` directory name is only a repository path and is not an execution dependency.

**Goal:** Build a closed-loop eleven-action controller in which a smooth, non-colliding virtual external magnet drives the existing dynamic capsule through the repository-local finite-magnet model, then quantitatively validate it on flat and stomach scenes.

**Architecture:** A dependency-light action controller freezes one-second HOLD, VIEW, or MOVE targets and produces a desired magnetic wrench from truth-state feedback. A numerical finite-magnet pose solver converts that desired wrench into bounded 6-DOF virtual-magnet pose increments around a capsule-relative nominal pose. An Isaac Lab bridge evaluates the analytical field at 240 Hz and applies only that model-produced capsule wrench while PhysX handles gravity and contact.

**Tech stack:** Python 3.11, NumPy, PyTorch, SciPy where already available, Magpylib 5.2.3, Isaac Lab 2.3.2, Isaac Sim 5.0.0, GPU PhysX, pytest.

**Authoritative design:** `docs/superpowers/specs/2026-08-20-virtual-magnet-eleven-action-closed-loop-design.md`

## Global execution rules

- Fetch `origin/workflow/TASK-007-virtual-magnet-closed-loop`, create `feature/TASK-007-virtual-magnet-closed-loop` from the fetched planning head, and do not merge `main`, TASK-005, or TASK-006.
- Verify that the planning commit's parent is exactly `bff26174ebd1aff2800883a0afdd5295f4f222d1` before editing.
- Preserve the old open-loop controller, scripts, report, and evidence as a regression baseline. Port only reusable action names or geometry deliberately; do not cherry-pick the failed TASK-005/TASK-006 controller implementations.
- Keep the capsule a gravity-enabled, non-kinematic dynamic rigid body. Never write capsule pose/velocity during action execution, lock DOFs, create a temporary joint, teleport, project, or inject an arbitrary desired wrench.
- Remove the arm and Ball from the TASK-007 actuation path. The virtual external magnet is a non-colliding analytical pose source and optional debug Xform, not a PhysX rigid body.
- Every applied controller wrench must be returned by the repository-local finite-magnet forward model at the current virtual magnet and capsule-magnet poses.
- Keep physics/magnetic evaluation at 240 Hz, feedback updates at 60 Hz, smooth pose interpolation at 240 Hz, and every action at exactly 240 substeps/1.000 seconds.
- Keep only eleven public IDs and only `COMPLETED`, `REJECTED`, and `FAULT`. Discard active-state requests without queueing.
- Keep large JSONL, videos, images, caches, and datasets outside Git. Commit code, profiles, compact JSON summaries, manifests, contracts, and the Markdown report.
- Run dependency-light tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Use `./run_isaaclab.sh -p` for live runtime tests. Do not install unrelated packages.
- Calibration failures are evidence, not immediate task termination. Tune within the authorized parameter set, replace contaminated held-out trials, and stop only at the explicit gates below.

---

### Task 1: Establish the branch and make the finite magnetic model repository-local

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/magnetics/__init__.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/magnetics/config.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/magnetics/field_models.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/magnetics/resources/default.json`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/magnetics/THIRD_PARTY_NOTICES.md`
- Create: `handoffs/reports/TASK-007-magnetic-dependency-manifest.sha256`
- Create: `tests/virtual_magnet/test_magnetic_dependency_regression.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/legacy_bridge.py`

**Required source hashes:** `config.py` must originate from SHA-256 `5d32740c62a75e06b7b876ed16f0043378ad45b72317b1f99637466b7f71ee07`, `field_models.py` from `be2f4d4af8db2e3a04552add61cbbc84d89e2348c08864a0c9cc3e6283265965`, and `default.json` from `e38563d558f6945f3041458060965ce6cd4b7044eacce573318c0f0fdcd319a6` before deliberate package-path-only edits.

- [ ] **Step 1: Create and verify the implementation branch**

```bash
git fetch origin workflow/TASK-007-virtual-magnet-closed-loop
git switch --create feature/TASK-007-virtual-magnet-closed-loop origin/workflow/TASK-007-virtual-magnet-closed-loop
git rev-parse HEAD
git rev-parse HEAD^
git status --short --branch
```

Expected: the worktree is clean and `HEAD^` is exactly `bff26174ebd1aff2800883a0afdd5295f4f222d1`.

- [ ] **Step 2: Verify the external source before copying**

```bash
sha256sum \
  /mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/robotarm/magnetic_sim/config.py \
  /mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/robotarm/magnetic_sim/magnetics/field_models.py \
  /mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/data/config/default.json
```

Expected: all three hashes match the authority above. If a hash differs, preserve the output and return `needs_decision` before importing unknown model code.

- [ ] **Step 3: Write failing regression tests**

The test must load both the legacy external implementation and the new package implementation, evaluate at least 32 deterministic relative magnet poses covering the working separation/orientation range, and compare field, force, and torque. Require finite outputs and relative error at most `1e-9` with absolute floors suitable for near-zero components. Also assert that package loading does not reference `/mnt/isaac-linux/isaacsim/extsUser`.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_magnetic_dependency_regression.py -q
```

Expected before implementation: FAIL because the repository-local package does not exist.

- [ ] **Step 4: Copy the minimal pure model, adapt only resource lookup, and preserve provenance**

Copy the verified source into the paths above. Change configuration/resource lookup to `importlib.resources`; do not change formulas, magnet dimensions, remanence, units, force method, or torque method. Record upstream paths, original hashes, post-adaptation hashes, Magpylib version, and licenses in the notice and manifest. Do not copy XRDF, streamlines, or visualization code unless a failing TASK-007 import proves it is required.

Change `legacy_bridge.py` to prefer the repository-local model, with an explicit environment-variable opt-in fallback to the old extension during this regression task. The final TASK-007 runtime must use the local package and print its resource digest.

- [ ] **Step 5: Run numerical and old-controller regressions**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_magnetic_dependency_regression.py -q
./run_isaaclab.sh -p scripts/table_motion/test_02_axial_field_scan.py --headless
./run_isaaclab.sh -p scripts/table_motion/test_05_long_axis_roll.py --headless
```

Expected: numerical comparison passes; the two old scripts retain their existing result criteria. Record new log paths and hashes without overwriting old evidence.

- [ ] **Step 6: Commit the self-contained magnetic dependency**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/magnetics \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/legacy_bridge.py \
  tests/virtual_magnet/test_magnetic_dependency_regression.py \
  handoffs/reports/TASK-007-magnetic-dependency-manifest.sha256
git commit -m "refactor: vendor validated finite magnet model"
```

---

### Task 2: Define the eleven-action, profile, and geometry contracts

**Files:**

- Create: `configs/virtual_magnet/closed_loop_v1.json`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/__init__.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/types.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/config.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/geometry.py`
- Create: `tests/virtual_magnet/conftest.py`
- Create: `tests/virtual_magnet/test_contract_and_geometry.py`

**Required interfaces:** `ActionId`, `ActionResult`, `Lifecycle`, `ControllerState`, `FrozenActionTarget`, `ControllerCommand`, `ControllerTelemetry`, `ClosedLoopProfile`, `load_profile()`, `profile_sha256()`, `view_target_axis()`, `unsigned_axis_tilt()`, `move_direction()`, and `quintic_progress()`.

- [ ] **Step 1: Write failing contract tests**

Test the exact ID/name mapping, exact three-result set, 240 substeps, 60 Hz update cadence, 15-degree relative VIEW geometry, image-coordinate signs, unsigned 45-degree MOVE tilt, 12-substep recent-sidewall window, opposite MOVE directions, 5 mm target, 4-to-6 mm acceptance interval, 0.8/0.2 timing, and 2 mm/s plus 0.1 rad/s boundary limits.

```python
def test_public_action_contract_is_exact():
    assert [(a.value, a.name) for a in ActionId] == [
        (0, "HOLD_VIEW"),
        (1, "VIEW_UP"),
        (2, "VIEW_UP_RIGHT"),
        (3, "VIEW_RIGHT"),
        (4, "VIEW_DOWN_RIGHT"),
        (5, "VIEW_DOWN"),
        (6, "VIEW_DOWN_LEFT"),
        (7, "VIEW_LEFT"),
        (8, "VIEW_UP_LEFT"),
        (9, "MOVE_SIDE_POS"),
        (10, "MOVE_SIDE_NEG"),
    ]
    assert {r.value for r in ActionResult} == {"completed", "rejected", "fault"}


def test_move_uses_unsigned_tilt_and_frozen_opposite_tangents():
    assert unsigned_axis_tilt((0, 0, -1), (0, 0, 1)) == pytest.approx(0.0)
    pos = move_direction(axis=(1, 0, 0), normal=(0, 0, 1), sign=1)
    neg = move_direction(axis=(1, 0, 0), normal=(0, 0, 1), sign=-1)
    np.testing.assert_allclose(pos, -neg)
```

- [ ] **Step 2: Confirm failure, implement dependency-light contracts, then pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_contract_and_geometry.py -q
```

The JSON profile must contain every tunable numeric parameter and schema version `task007_virtual_magnet_closed_loop_v1`. It must not contain scene-specific sections. Reject unknown keys, wrong units, missing fields, NaN/Inf, and inconsistent frequency/timing values.

- [ ] **Step 3: Commit the contracts**

```bash
git add configs/virtual_magnet \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet \
  tests/virtual_magnet
git commit -m "feat: define virtual magnet eleven-action contract"
```

---

### Task 3: Implement and verify the finite-magnet pose inverse

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/pose_inverse.py`
- Create: `tests/virtual_magnet/test_pose_inverse.py`

**Required interfaces:** `PoseInverseState`, `PoseInverseResult`, `numerical_pose_jacobian()`, `solve_pose_increment()`, and `integrate_pose_increment()`.

- [ ] **Step 1: Write failing analytical-wrapper tests**

Use a deterministic differentiable fake wrench model to verify central differences, SI units, weighted damped least squares, regularization toward relative pose, translation/rotation trust regions, quaternion normalization, sign continuity, condition-number telemetry, and finite hold-last behavior on singular Jacobians.

- [ ] **Step 2: Write failing finite-model convergence tests**

At deterministic safe working poses, create reachable wrench targets by evaluating the finite model at known nearby poses. Starting from perturbed poses, require iterative residual reduction without exceeding separation, orientation, or step limits. Include near-singular and unreachable targets and require a finite saturated result rather than an exception or `FAULT`.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_pose_inverse.py -q
```

- [ ] **Step 3: Implement the solver and pass both test classes**

Use central finite differences for three translations and three local-axis rotations. Apply independent wrench weights, Levenberg damping, relative-pose regularization, and a trust region before integrating the pose. Never return a pose containing NaN/Inf. On saturation, return the previous finite pose and complete diagnostics.

- [ ] **Step 4: Commit the inverse solver**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/pose_inverse.py \
  tests/virtual_magnet/test_pose_inverse.py
git commit -m "feat: solve bounded finite magnet pose updates"
```

---

### Task 4: Implement the pure one-second closed-loop action controller

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/outer_loop.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet/controller.py`
- Create: `tests/virtual_magnet/test_outer_loop.py`
- Create: `tests/virtual_magnet/test_controller.py`

**Required interfaces:** `desired_hold_wrench()`, `desired_view_wrench()`, `desired_move_wrench()`, and `VirtualMagnetElevenActionController.reset/submit/step`.

- [ ] **Step 1: Write failing lifecycle tests**

Test exact 240-step occupancy, target freezing, 60 Hz solve events, 240 Hz interpolation, no pose discontinuity, active-request discard with no queue, terminal results, full-second rejected HOLD substitution, and true-FAULT-only behavior. Test that truth inputs remain internal telemetry and are not part of the public action result.

- [ ] **Step 2: Write failing action-law tests**

HOLD must prioritize frozen optical axis then tangent anchor, with no twist or normal-position target. VIEW must use the frozen 15-degree camera-frame target, quintic 0.8-second motion, 0.2-second hold, minimal swing, and inward-swing cancellation on camera-end wall contact. MOVE must use the frozen normal/direction, 5 mm quintic displacement, passive roll, no tilt/roll controller, and the exact unsigned-tilt plus recent-sidewall eligibility rule.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_outer_loop.py \
  tests/virtual_magnet/test_controller.py -q
```

- [ ] **Step 3: Implement the controller and pass focused tests**

The outer loop may output only a bounded desired magnetic wrench. The controller must pass that request through `pose_inverse.py`, retain the capsule-relative nominal magnet pose, and output a finite virtual-magnet target plus telemetry. No test helper or runtime path may bypass the finite model by applying the desired wrench directly.

- [ ] **Step 4: Add a closed-loop causality test**

From the same frozen target, feed two different capsule states at a 60 Hz update and assert that the virtual-magnet target changes. Repeat with feedback disabled in a test-only baseline and assert that it does not change. This unit test is necessary but does not replace the live paired disturbance test.

- [ ] **Step 5: Commit the pure controller**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet \
  tests/virtual_magnet
git commit -m "feat: implement closed-loop magnetic action controller"
```

---

### Task 5: Integrate the controller with Isaac Lab without arm or Ball actuation

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/virtual_magnet_action.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/virtual_magnet_bridge.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_virtual_magnet_flat_env_cfg.py`
- Create: `tests/virtual_magnet/test_action_term.py`
- Create: `tests/virtual_magnet/test_flat_task_cfg.py`
- Create: `scripts/virtual_magnet/inspect_runtime_contract.py`

**Required behavior:** one scalar discrete action term, one zero-dimensional internal magnetic physics term, a dynamic capsule, no arm/Ball command manager term, and an optional non-colliding debug Xform for the magnet.

- [ ] **Step 1: Write failing configuration and adapter tests**

Assert that the public action dimension is one, accepted values are integer IDs 0 through 10, `-1` is internal only, requests during execution are discarded, the task uses 240 Hz physics and one-second actions, and no joint-position action drives `j1..j6` or `ballxj..ballzj`. Assert from the composed USD/runtime schema that the capsule remains dynamic and that the debug magnet has no collider or rigid-body API.

- [ ] **Step 2: Implement the runtime adapter**

Read capsule state, contact history, and local surface query every physics substep. Evaluate the local finite magnet model and inject its filtered/limited capsule wrench at 240 Hz. The 60 Hz controller update changes the magnet target; each 240 Hz substep interpolates its pose before wrench evaluation. Retain validated coupling ramp, filter, damping, and hard limits as configurable starting values.

Expose a runtime audit structure containing desired wrench, model-produced raw/filtered wrench, magnet pose/relative pose, solver diagnostics, controller error, contact class, lifecycle, and result. Add a development assertion that the applied wrench equals the bridge's filtered model wrench within floating-point tolerance.

- [ ] **Step 3: Run static and live runtime inspection**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_action_term.py \
  tests/virtual_magnet/test_flat_task_cfg.py -q
./run_isaaclab.sh -p scripts/virtual_magnet/inspect_runtime_contract.py --headless
```

The live inspection must show 240 substeps, 60 feedback events, finite smooth magnet-pose increments, a changing capsule state caused only by model-produced wrench plus physics, no robot/Ball command changes, dynamic capsule readback, and no virtual-magnet collision API.

- [ ] **Step 4: Commit Isaac integration**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_virtual_magnet_flat_env_cfg.py \
  scripts/virtual_magnet/inspect_runtime_contract.py \
  tests/virtual_magnet
git commit -m "feat: integrate virtual magnet with dynamic capsule"
```

---

### Task 6: Add deterministic trials, telemetry, calibration, and keyboard visualization

**Files:**

- Create: `scripts/virtual_magnet/common.py`
- Create: `scripts/virtual_magnet/calibrate_closed_loop.py`
- Create: `scripts/virtual_magnet/validate_flat.py`
- Create: `scripts/virtual_magnet/teleop_virtual_magnet.py`
- Create: `tests/virtual_magnet/test_trial_protocol.py`
- Create: `tests/virtual_magnet/test_keyboard.py`
- Create: `tests/virtual_magnet/test_acceptance_summary.py`

- [ ] **Step 1: Write failing protocol tests**

Test deterministic seed manifests, train/development/held-out separation, replacement of any held-out sample used for tuning, 20-sample counts, per-action 16-of-20 threshold, separate invalid-MOVE classes, exact one-second substeps, boundary stability calculation over the last 24 substeps, external evidence hashing, and aggregate disposition.

- [ ] **Step 2: Implement the calibration search**

Search only controller-authorized parameters: feedback gains, inverse weights/damping/trust regions, nominal relative magnet pose, filters, and magnetic wrench limits. Log every candidate, seed set, score, failure classification, profile digest, and selected profile. Keep mass, inertia, friction, geometry, gravity, magnetic material constants, and PhysX fixed.

The search should optimize a robust aggregate rather than one nominal trajectory. Diagnose sign mistakes, saturation, contact classification, pose separation, and controller oscillation before enlarging limits. Do not return an error after the first failed trial.

- [ ] **Step 3: Implement flat validation and terminal summaries**

HOLD pass: optical-axis error at most 3 degrees, tangent drift at most 2 mm, and boundary stability. Unblocked VIEW pass: target error at most 3 degrees, tangent drift at most 2 mm, and boundary stability. Constrained VIEW is reported separately. Eligible MOVE pass: signed tangent displacement between 4 and 6 mm and boundary stability; passive roll sign/angle and slip are recorded. Invalid MOVE pass: `REJECTED`, 240 substeps, and HOLD metrics. All samples require finite state and no true `FAULT`.

- [ ] **Step 4: Implement one-key/one-action visualization**

Use keys `0` through `8` for HOLD and eight VIEWs, `9` for positive MOVE, and `-` for negative MOVE. Ignore key-repeat and all input while executing. After completion, print the full terminal telemetry and wait indefinitely for the next fresh key press. Provide `--render_fps {60,120,240}`, default 120, while physics remains 240 Hz.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/virtual_magnet/test_trial_protocol.py \
  tests/virtual_magnet/test_keyboard.py \
  tests/virtual_magnet/test_acceptance_summary.py -q
./run_isaaclab.sh -p scripts/virtual_magnet/teleop_virtual_magnet.py --render_fps 120
```

- [ ] **Step 5: Commit the experiment infrastructure**

```bash
git add scripts/virtual_magnet tests/virtual_magnet
git commit -m "test: add virtual magnet calibration and visualization"
```

---

### Task 7: Calibrate and pass flat no-disturbance acceptance

**Files:**

- Modify only as justified: `configs/virtual_magnet/closed_loop_v1.json`
- Modify only as structurally necessary: virtual-magnet controller/runtime files
- Create: `handoffs/reports/TASK-007-flat-no-disturbance-summary.json`

- [ ] **Step 1: Generate immutable development and held-out manifests**

Name every initial-state seed and class. For each public action, reserve at least 20 valid held-out starts. Reserve a separate 20 invalid starts for each MOVE sign. VIEW needs enough unblocked starts to fill 20 samples after constrained samples are separated.

- [ ] **Step 2: Calibrate on development samples until stable or structurally blocked**

```bash
./run_isaaclab.sh -p scripts/virtual_magnet/calibrate_closed_loop.py \
  --scene flat --trials-per-action 20 --search-budget 200 --headless
```

`--search-budget` is a maximum candidate count, not an instruction to stop early after one failure. Preserve the best candidate table and reason for every rejected candidate.

- [ ] **Step 3: Run untouched held-out no-disturbance validation**

```bash
./run_isaaclab.sh -p scripts/virtual_magnet/validate_flat.py \
  --mode no-disturbance --trials-per-action 20 --held-out --headless
```

Blocking gate: every valid action class passes at least 16 of 20. Each invalid MOVE class returns correct `REJECTED` behavior in at least 16 of 20, with predicate mismatches reported and replaced when caused by initial-state generation rather than the controller. If any parameter changes after viewing these results, invalidate the affected held-out set, generate a new set, and rerun all classes.

- [ ] **Step 4: Preserve the compact summary and commit the selected profile**

```bash
git add configs/virtual_magnet/closed_loop_v1.json \
  handoffs/reports/TASK-007-flat-no-disturbance-summary.json \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/virtual_magnet \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/virtual_magnet_bridge.py
git commit -m "tune: pass flat virtual magnet actions"
```

Do not begin Task 8 until this gate passes. If authorized tuning is exhausted with a reproducible structural failure, write the report and return `needs_decision` rather than weakening thresholds.

---

### Task 8: Prove closed-loop disturbance recovery on the flat scene

**Files:**

- Create: `scripts/virtual_magnet/validate_disturbance.py`
- Create: `tests/virtual_magnet/test_disturbance_protocol.py`
- Create: `handoffs/reports/TASK-007-flat-disturbance-summary.json`

- [ ] **Step 1: Write and pass disturbance-protocol tests**

The protocol must generate randomized initial state error and one bounded mid-action force/torque disturbance with seed, start substep, duration, vector, and magnitude recorded. The controller receives no disturbance metadata. Each sample has a paired feedback-disabled run with identical start and disturbance.

- [ ] **Step 2: Run 20 new held-out disturbed samples for every action**

```bash
./run_isaaclab.sh -p scripts/virtual_magnet/validate_disturbance.py \
  --trials-per-action 20 --paired-open-loop --held-out --headless
```

Blocking gate: every action passes at least 16 of 20 under the same action metrics. Evidence must show that after disturbance the feedback-enabled virtual magnet command changes relative to the no-disturbance prediction and produces better endpoint recovery than its paired feedback-disabled baseline. Do not claim closed-loop behavior from final success alone.

- [ ] **Step 3: Commit protocol and compact evidence**

```bash
git add scripts/virtual_magnet/validate_disturbance.py \
  tests/virtual_magnet/test_disturbance_protocol.py \
  handoffs/reports/TASK-007-flat-disturbance-summary.json
git commit -m "test: validate flat closed-loop disturbance recovery"
```

Do not begin stomach integration if this gate remains below threshold after authorized diagnosis and tuning. Any tuned profile requires both Task 7 and Task 8 to be repeated with fresh held-out sets.

---

### Task 9: Integrate the unchanged controller with the stomach scene and pass held-out regions

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_virtual_magnet_stomach_env_cfg.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Create: `scripts/virtual_magnet/validate_stomach.py`
- Create: `tests/virtual_magnet/test_stomach_task_cfg.py`
- Create: `handoffs/reports/TASK-007-stomach-summary.json`

- [ ] **Step 1: Write a failing same-controller configuration test**

Assert that flat and stomach tasks import the same controller class and same profile path/digest, use the same rates/actions/contact rules/thresholds, and contain no scene-name gain branches. The stomach wrapper may change only scene asset, initial-state sampler, surface-query source, lighting/camera setup, and evidence labels.

- [ ] **Step 2: Integrate and run stomach development regions**

```bash
./run_isaaclab.sh -p scripts/virtual_magnet/validate_stomach.py \
  --split development --trials-per-action 20 --headless
```

The development split may guide changes to the single shared controller profile. Do not add stomach-only parameters. If the shared profile changes, create fresh flat held-out manifests and rerun Tasks 7 and 8 before final stomach acceptance.

- [ ] **Step 3: Run new held-out stomach regions with no injected disturbance**

```bash
./run_isaaclab.sh -p scripts/virtual_magnet/validate_stomach.py \
  --split held-out --trials-per-action 20 --headless
```

Blocking gate: every valid action class passes at least 16 of 20; invalid MOVE classes are separately validated. Record region IDs, initial-state seeds, constrained VIEWs, low-effect actions, solver saturation, contact behavior, and passive rolling. No artificial disturbance is permitted in this run.

- [ ] **Step 4: Commit stomach integration and compact evidence**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_virtual_magnet_stomach_env_cfg.py \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py \
  scripts/virtual_magnet/validate_stomach.py \
  tests/virtual_magnet/test_stomach_task_cfg.py \
  handoffs/reports/TASK-007-stomach-summary.json
git commit -m "feat: validate virtual magnet controller in stomach"
```

---

### Task 10: Run continuous sequences, regressions, and user visualization

**Files:**

- Create: `scripts/virtual_magnet/validate_sequence.py`
- Create: `configs/virtual_magnet/sequence_100.json`
- Create: `tests/virtual_magnet/test_sequence_protocol.py`
- Create: `handoffs/reports/TASK-007-sequence-summary.json`

- [ ] **Step 1: Freeze and test one 100-ID sequence**

The sequence must contain all IDs, opposite VIEW pairs, consecutive eligible and ineligible MOVE requests, rejected MOVE, MOVE-to-VIEW, and MOVE-to-HOLD. Use the same ID file in flat and stomach. Test that finite low-effect results continue and only a true `FAULT` truncates.

- [ ] **Step 2: Run flat and stomach without reset**

```bash
./run_isaaclab.sh -p scripts/virtual_magnet/validate_sequence.py \
  --scene flat --sequence configs/virtual_magnet/sequence_100.json --small-disturbances --headless
./run_isaaclab.sh -p scripts/virtual_magnet/validate_sequence.py \
  --scene stomach --sequence configs/virtual_magnet/sequence_100.json --headless
```

Both runs must finish all 100 requests without a true `FAULT`. Verify exact 240-substep boundaries, no queued input, finite magnet/capsule states, bounded magnet-relative pose, result counts, and boundary stability statistics. The stomach command must reject any disturbance flag.

- [ ] **Step 3: Run the full dependency-light and live regression suites**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/virtual_magnet -q
./run_isaaclab.sh -p scripts/table_motion/test_02_axial_field_scan.py --headless
./run_isaaclab.sh -p scripts/table_motion/test_05_long_axis_roll.py --headless
./run_isaaclab.sh -p scripts/virtual_magnet/inspect_runtime_contract.py --headless
```

- [ ] **Step 4: Launch both keyboard scenes for visual review**

```bash
./run_isaaclab.sh -p scripts/virtual_magnet/teleop_virtual_magnet.py \
  --scene flat --render_fps 120
./run_isaaclab.sh -p scripts/virtual_magnet/teleop_virtual_magnet.py \
  --scene stomach --render_fps 120
```

Record launch commands, measured FPS, one example of every action, at least one rejected MOVE, and external video/log hashes. The user remains the visual acceptance authority; do not convert visual preference into an unapproved numeric threshold.

- [ ] **Step 5: Commit sequence and regression artifacts**

```bash
git add scripts/virtual_magnet/validate_sequence.py \
  configs/virtual_magnet/sequence_100.json \
  tests/virtual_magnet/test_sequence_protocol.py \
  handoffs/reports/TASK-007-sequence-summary.json
git commit -m "test: validate continuous virtual magnet action sequences"
```

---

### Task 11: Write the final report, audit the branch, and return it without merging

**Files:**

- Create: `handoffs/reports/TASK-007-virtual-magnet-eleven-action-closed-loop-report.md`
- Modify: `handoffs/reports/README.md`

- [ ] **Step 1: Write the evidence-grounded report**

Include exact base/head/branch, commits, commands, runtime versions, GPU/PhysX warnings, dependency provenance, model/profile/physics digests, all per-action 20-sample counts and rates, invalid-MOVE tables, disturbance paired comparisons, stomach split definitions, both 100-action sequences, keyboard evidence, test counts, external paths/bytes/SHA-256, known limitations, and final disposition.

Do not describe any failed or unrun test as passed. Do not call the controller hardware-realizable, arm-safe, obstacle-crossing, or free of CCD limitations. Distinguish `COMPLETED` from quantitative pass and distinguish passive rolling telemetry from the 4-to-6 mm MOVE gate.

- [ ] **Step 2: Run final repository checks**

```bash
git diff --check origin/workflow/TASK-007-virtual-magnet-closed-loop...HEAD
git status --short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/virtual_magnet -q
git log --oneline --decorate -12
```

Expected: no diff errors, no untracked implementation files, all dependency-light tests pass, and evidence accurately matches the report.

- [ ] **Step 3: Commit and push the implementation branch**

```bash
git add handoffs/reports/TASK-007-virtual-magnet-eleven-action-closed-loop-report.md \
  handoffs/reports/README.md
git commit -m "docs: report TASK-007 virtual magnet validation"
git push -u origin feature/TASK-007-virtual-magnet-closed-loop
git status --short --branch
```

Return `complete` only after all gates in the design pass. Return `partial` if both flat gates pass but stomach acceptance remains below threshold after authorized tuning or a true stomach runtime failure prevents completion. Return `needs_decision` only after authorized diagnosis/tuning is exhausted and a reproducible structural blocker remains. Never merge the branch.
