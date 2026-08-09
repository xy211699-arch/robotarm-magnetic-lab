# P0 Manual Teleoperation and Coverage Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated Isaac Lab stomach task in which a human submits the frozen eleven atomic actions from the keyboard and evaluates cumulative, ray-occlusion-aware inner-surface coverage at the recorded 1 Hz camera boundary.

**Architecture:** Keep the existing atomic executor unchanged. Add an isolated privileged evaluation path consisting of reference-surface preprocessing, GPU-batched visibility queries, a monotonic coverage accumulator, records, and debug visualization. Connect it to a keyboard session controller at the same request boundary that a future Actor will use.

**Tech Stack:** Isaac Lab and Isaac Sim APIs already pinned by the repository, Python, PyTorch, USD/PhysX scene queries, Kit input and debug drawing, pytest, and the existing action-layer validation scripts.

## Global Constraints

- Work only on `feature/TASK-001-p0-coverage-teleop`, based on the exact commit recorded in the active handoff contract.
- Do not modify the eleven action IDs, templates, controller semantics, existing device result vocabulary, camera calibration, stomach asset, magnetic model, or existing tasks.
- Do not feed capsule truth, coverage state, ray results, or visualization data into deployable observations or any executor decision.
- Stop with `needs_decision` before implementation if the stomach inner surface is ambiguous or a GPU-batched first-hit ray API is unavailable.
- Use test-driven development for every task and commit after each passing task.
- Keep generated logs, masks, timing arrays, screenshots, and simulator caches out of Git.
- The repository currently ignores new files below `/tests/`; use `git add -f` only for the explicitly named new test files, and do not broaden that exception.

## Expected File Map

Create or modify the closest equivalent package paths after Linux preflight confirms the actual repository layout. Any path deviation must be documented in the final report.

```text
source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/reference_mesh.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/visibility.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/accumulator.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/records.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/atomic_keyboard.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/session_controller.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/ui/coverage_view.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_atomic_stomach_teleop_env_cfg.py
scripts/action_layer/inspect_p0_coverage_prerequisites.py
scripts/action_layer/validate_coverage_geometry.py
scripts/action_layer/validate_atomic_stomach_teleop.py
scripts/action_layer/teleop_atomic_stomach_coverage.py
tests/coverage/test_prerequisite_report.py
tests/coverage/test_reference_mesh.py
tests/coverage/test_visibility_geometry.py
tests/coverage/test_coverage_accumulator.py
tests/coverage/test_coverage_records.py
tests/coverage/test_coverage_colors.py
tests/action_layer/test_atomic_stomach_teleop_cfg.py
tests/action_layer/test_atomic_keyboard_protocol.py
docs/P0_COVERAGE_TELEOP.md
```

## Task 1: Run the Mandatory Linux Preflight

**Files:**

- Create: `scripts/action_layer/inspect_p0_coverage_prerequisites.py`
- Create: `tests/coverage/test_prerequisite_report.py`
- Output outside Git: `logs/p0_coverage_preflight/<timestamp>/prerequisites.json`

- [ ] Write a failing test that requires a structured report containing repository commit, dependency versions, registered existing atomic task, action table, camera prim/configuration, all `UsdGeom.Mesh` prims under the stomach root, transforms, bounds, vertex/face counts, and discovered GPU batch-ray APIs.
- [ ] Run the test and record the expected missing-script failure.
- [ ] Implement a read-only inspector. It must not choose an inner mesh silently and must not modify the stage.
- [ ] Run the inspector in the target stomach scene and save its JSON output outside Git.
- [ ] Compare candidate stomach meshes and ray APIs against the design gate. If either is ambiguous or unavailable, write a `needs_decision` report and stop all later tasks.
- [ ] Run the test again and commit the inspector and test.

```bash
python -m pytest tests/coverage/test_prerequisite_report.py -q
python scripts/action_layer/inspect_p0_coverage_prerequisites.py --headless --num_envs 1
git add scripts/action_layer/inspect_p0_coverage_prerequisites.py
git add -f tests/coverage/test_prerequisite_report.py
git commit -m "test: inspect P0 coverage prerequisites"
```

## Task 2: Build the Reference Inner-Surface Model

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/__init__.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/reference_mesh.py`
- Create: `tests/coverage/test_reference_mesh.py`

- [ ] Write synthetic tests for world-transform application, explicit prim selection, exclusion of nonselected meshes, `1e-6 m` duplicate welding, incident-triangle preservation, deterministic ordering, and stable geometry hashing.
- [ ] Run the tests and confirm they fail because the module is absent.
- [ ] Implement immutable reference-mesh data structures and preprocessing without simulator-global mutable state.
- [ ] Add strict validation for empty selections, non-triangular faces unless explicitly triangulated, invalid indices, nonfinite vertices, and unsupported topology.
- [ ] Run the focused tests, then validate the selected stomach surface using the preflight evidence.
- [ ] Commit the passing implementation and tests.

```bash
python -m pytest tests/coverage/test_reference_mesh.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/__init__.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/reference_mesh.py
git add -f tests/coverage/test_reference_mesh.py
git commit -m "feat: preprocess stomach coverage reference mesh"
```

## Task 3: Implement Visibility and the Monotonic Accumulator

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/visibility.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/accumulator.py`
- Create: `scripts/action_layer/validate_coverage_geometry.py`
- Create: `tests/coverage/test_visibility_geometry.py`
- Create: `tests/coverage/test_coverage_accumulator.py`

- [ ] Write deterministic fixtures for points inside, on, and outside the 50 mm sphere and 60-degree circular cone.
- [ ] Write occlusion fixtures for an incident first hit, a non-incident nearer hit, and the `1e-4 m` hit-distance tolerance.
- [ ] Write accumulator tests proving frame-ID de-duplication, one update per new frame, cumulative monotonicity, accurate gain, and full reset.
- [ ] Run the focused tests and record the expected failures.
- [ ] Implement the pure candidate filter and scalar ray oracle first.
- [ ] Implement the approved GPU-batched first-hit adapter using the API verified in Task 1; do not add a depth-buffer fallback.
- [ ] Compare GPU and scalar results on deterministic fixtures and fail loudly on disagreement.
- [ ] Run the focused tests and geometry validator, then commit.

```bash
python -m pytest tests/coverage/test_visibility_geometry.py tests/coverage/test_coverage_accumulator.py -q
python scripts/action_layer/validate_coverage_geometry.py --check all
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/visibility.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/accumulator.py scripts/action_layer/validate_coverage_geometry.py
git add -f tests/coverage/test_visibility_geometry.py tests/coverage/test_coverage_accumulator.py
git commit -m "feat: compute occlusion-aware cumulative coverage"
```

## Task 4: Register the Dedicated Stomach Teleoperation Task

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_atomic_stomach_teleop_env_cfg.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Create: `tests/action_layer/test_atomic_stomach_teleop_cfg.py`

- [ ] Write a failing configuration test for environment ID `Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0`, one environment, scalar action shape `(1,)`, 240 Hz physics/magnetic update, 20 Hz atomic control, and reuse of the frozen action table.
- [ ] Add assertions that deployable policy observations remain limited to the established robot-joint and external-magnet state contract and contain no capsule truth or coverage fields.
- [ ] Implement the smallest derived environment configuration that composes the existing stomach scene and atomic action term.
- [ ] Do not copy or fork action templates when inheritance or composition is sufficient.
- [ ] Run configuration and task-registration smoke tests, then commit.

```bash
python -m pytest tests/action_layer/test_atomic_stomach_teleop_cfg.py -q
python -c "import gymnasium as gym; import source; print(gym.spec('Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0'))"
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_atomic_stomach_teleop_env_cfg.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py
git add -f tests/action_layer/test_atomic_stomach_teleop_cfg.py
git commit -m "feat: register atomic stomach teleoperation task"
```

## Task 5: Implement the Keyboard Boundary Protocol

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/atomic_keyboard.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/session_controller.py`
- Create: `tests/action_layer/test_atomic_keyboard_protocol.py`

- [ ] Write pure tests for every fixed key mapping and for key-down-only behavior.
- [ ] Write state-machine tests for OS-repeat suppression, `IGNORED_WHILE_BUSY`, `MASKED_ACTION`, `RESET_WHILE_BUSY`, `EPISODE_TERMINATED`, one acknowledgement after `DONE`, and reset/exit-only behavior after `HARD_FAILURE`.
- [ ] Assert that busy inputs are discarded and are never executed later.
- [ ] Run the tests and confirm the missing implementation fails.
- [ ] Implement an input adapter separated from the session state machine so the protocol tests need no simulator.
- [ ] Connect requests to the existing scalar action boundary without modifying the executor.
- [ ] Run the focused tests and commit.

```bash
python -m pytest tests/action_layer/test_atomic_keyboard_protocol.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/atomic_keyboard.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/session_controller.py
git add -f tests/action_layer/test_atomic_keyboard_protocol.py
git commit -m "feat: add boundary-safe atomic keyboard control"
```

## Task 6: Add Reproducible Records

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/records.py`
- Create: `tests/coverage/test_coverage_records.py`

- [ ] Write tests for versioned `metadata.json`, append-only `actions.jsonl`, append-only `frames.jsonl`, unique request IDs, unique recorded frame IDs, timestamps, coverage count consistency, and atomic finalization.
- [ ] Write a test proving the policy/executor whitelist cannot receive privileged evaluator fields.
- [ ] Run the tests and record the failures.
- [ ] Implement bounded record schemas and SHA-256 artifact inventory helpers.
- [ ] Keep masks, images, trajectories, and timing arrays outside Git while recording their relative paths and hashes.
- [ ] Run the focused tests and commit.

```bash
python -m pytest tests/coverage/test_coverage_records.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/records.py
git add -f tests/coverage/test_coverage_records.py
git commit -m "feat: record P0 actions and coverage frames"
```

## Task 7: Add the Isolated 3D View and Deterministic 2D Export

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/ui/coverage_view.py`
- Create: `tests/coverage/test_coverage_colors.py`

- [ ] Write pure color-mapping tests requiring uncovered red, covered green, dark capsule marker, black trajectory, and exact agreement between color counts and the boolean mask.
- [ ] Write projection tests for deterministic output dimensions, orientation metadata, reset behavior, and exact percentage text input.
- [ ] Run the tests and confirm the expected failures.
- [ ] Implement a 30 Hz view that reads only the last completed 1 Hz mask and never invokes visibility updates.
- [ ] Isolate debug drawables from collision, magnetic queries, coverage rays, and capsule RGB visibility.
- [ ] Implement exports on `F12`, reset, hard failure, and exit.
- [ ] Run the focused tests and a rendered smoke test, capture evidence outside Git, then commit.

```bash
python -m pytest tests/coverage/test_coverage_colors.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/ui/coverage_view.py
git add -f tests/coverage/test_coverage_colors.py
git commit -m "feat: visualize and export stomach coverage"
```

## Task 8: Integrate the Launcher and End-to-End Validator

**Files:**

- Create: `scripts/action_layer/teleop_atomic_stomach_coverage.py`
- Create: `scripts/action_layer/validate_atomic_stomach_teleop.py`
- Create: `docs/P0_COVERAGE_TELEOP.md`

- [ ] Write a scripted input source for headless integration tests; the interactive keyboard adapter and scripted source must drive the same session controller.
- [ ] Add an integration check that submits all eleven action IDs only at valid boundaries and confirms no request duplication or queueing.
- [ ] Add a synthetic frame-clock check proving exactly one coverage update per unique recorded 1 Hz frame ID.
- [ ] Add consistency checks across stored masks, overlay values, records, 3D colors, and 2D exports.
- [ ] Implement the interactive launcher with `--task`, `--num_envs 1`, seed, output directory, and optional headless scripted validation arguments.
- [ ] Document startup, keys, status meanings, output layout, reset semantics, and known non-goals.
- [ ] Run headless validation and an interactive rendered session, then commit.

```bash
python scripts/action_layer/validate_atomic_stomach_teleop.py --headless --num_envs 1
python scripts/action_layer/teleop_atomic_stomach_coverage.py --task Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0 --num_envs 1
git add scripts/action_layer docs/P0_COVERAGE_TELEOP.md
git commit -m "feat: integrate P0 coverage teleoperation"
```

## Task 9: Run Acceptance, Regression, and Performance Evidence

**Files:**

- Create: `handoffs/reports/TASK-001-p0-coverage-teleop-report.md`
- Modify only if required by established workflow: `.gitignore`

- [ ] Run all new pure protocol, geometry, accumulator, records, color, and configuration tests.
- [ ] Run the stage-one pure protocol suite and record the exact 10/10 result or every deviation.
- [ ] Run the existing table atomic-action acceptance and record every one of the eleven action results.
- [ ] Run the legacy 9D table smoke test.
- [ ] Run the stomach teleoperation validator and a manual rendered session.
- [ ] Collect at least 100 coverage updates and report median, p95, maximum, candidate counts, ray counts, and synchronization assumptions; every update must stay within the 1 second recorded-frame deadline.
- [ ] Confirm `git diff --check`, repository status, and absence of generated evidence in Git.
- [ ] Write the handoff report with base commit, head commit, branch, commands, results, deviations, unverified items, and artifact path/size/SHA-256 inventory.
- [ ] Commit the report, push the feature branch, and stop. Do not merge to `main`.

```bash
python -m pytest tests/coverage tests/action_layer/test_atomic_stomach_teleop_cfg.py tests/action_layer/test_atomic_keyboard_protocol.py -q
git diff --check
git status --short
git add handoffs/reports/TASK-001-p0-coverage-teleop-report.md
git commit -m "docs: report TASK-001 P0 coverage acceptance"
git push -u origin feature/TASK-001-p0-coverage-teleop
```

## Completion Rule

Linux may report `complete` only when all acceptance gates pass with reproducible evidence. Any unresolved inner-surface selection, unavailable GPU-batched first-hit API, unauthorized interface change, regression, missed 1 Hz deadline, or incomplete evidence must be reported as `needs_decision`, `partial`, or `blocked` as appropriate. Passing P0 does not authorize reinforcement-learning training or claim that full stomach coverage is achievable.
