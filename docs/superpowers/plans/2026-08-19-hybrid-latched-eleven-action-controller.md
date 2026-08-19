# Hybrid Latched Eleven-Action Controller Implementation Plan

> **Linux executor:** This plan is self-contained. The Linux Codex session is not required to have or invoke `superpowers:subagent-driven-development`, `superpowers:executing-plans`, or any other superpowers skill. Execute the checked steps manually in order and preserve evidence after every gate.

**Goal:** Build and validate `hybrid_latched_v1`, which keeps TASK-005's force/torque-driven one-second eleven actions but hard-locks the capsule at every 1 Hz RGB observation boundary.

**Architecture:** A dependency-light controller emits latch intents; an Isaac-specific runtime adapter applies either six dynamic lock flags or the explicitly approved kinematic fallback. A capture barrier exposes RGB only after latch, zero-wrench, and zero-velocity readback. Flat randomized and paired-release gates block unchanged stomach migration.

**Tech Stack:** Python 3.11, NumPy, PyTorch, Isaac Lab 2.3.2, Isaac Sim 5.0.0, PhysX GPU dynamics, USD `UsdPhysics`/`PhysxSchema`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-hybrid-latched-eleven-action-controller-design.md`

## Global Constraints

- Base exactly `67b7bf44747f08422add0cee7e6b94280bbeff6d`; work on `feature/TASK-006-hybrid-latched-v1`; do not merge `main`.
- Keep action IDs 0 through 10, keyboard mapping, 15 degree VIEW target, 240 Hz, 240 substeps, MOVE 60/120/60 timing, and `0.9mg` unchanged.
- Keep GPU dynamics. Do not change CCD, assets, geometry, mass, inertia, gravity, materials, solver, VLM, Actor-Critic, rewards, coverage, or robot/magnet code.
- Public results remain only `COMPLETED`, `REJECTED`, and `FAULT`; do not add `target_reached`.
- Lock the actual current pose only. Never project, snap, repair penetration, or repeatedly overwrite pose.
- All policy RGB frames require confirmed latch, zero wrench, zero linear velocity, and zero angular velocity.
- Preserve large logs, videos, screenshots, and JSONL outside Git; commit only code, compact summaries, contracts, and the final Markdown report.
- Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; do not install unrelated dependencies.
- If an API or runtime behavior differs from this plan, record evidence and stop at the named gate instead of silently substituting another mechanism.

---

### Task 1: Establish the TASK-006 branch and latch data contracts

**Files:**

- Create: `configs/eleven_action/hybrid_latched_profile.json`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/latch.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/types.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/__init__.py`
- Create: `tests/eleven_action/test_hybrid_latch_contract.py`

**Interfaces:**

- Produces `LatchIntent`, `LatchReason`, `LatchBackendName`, `LatchedContactSnapshot`, `LatchProfile`, `load_latch_profile()`, and `latch_profile_sha256()`.
- Extends `Lifecycle` with `LATCHED_READY` while preserving `READY_HOLD` only for reading old TASK-005 evidence if needed.
- Extends internal `ActionTelemetry` with latch/capture metadata but does not extend `ActionResult`.

- [ ] **Step 1: Verify and create the branch from the exact base**

```bash
git fetch origin workflow/TASK-006-hybrid-latched-v1
git switch --create feature/TASK-006-hybrid-latched-v1 origin/workflow/TASK-006-hybrid-latched-v1
git rev-parse HEAD
git rev-parse HEAD^
git status --short --branch
```

Expected: HEAD contains the Windows planning commit, `HEAD^` is exactly the TASK-005 report commit `67b7bf44747f08422add0cee7e6b94280bbeff6d`, and the worktree is clean before edits.

- [ ] **Step 2: Write failing contract tests**

```python
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
    ActionResult,
    LatchBackendName,
    LatchIntent,
    LatchProfile,
    LatchReason,
    LatchedContactSnapshot,
    load_latch_profile,
)


def test_public_results_remain_exactly_three():
    assert {item.value for item in ActionResult} == {"completed", "rejected", "fault"}


def test_latch_profile_is_frozen():
    profile = load_latch_profile()
    assert profile.schema_version == "task006_hybrid_latched_v1"
    assert profile.physics_hz == 240
    assert profile.policy_rgb_hz == 1
    assert profile.view_error_limit_deg == 3.0
    assert profile.support_drift_limit_m == 0.002
    assert profile.release_window_s == 0.05
    assert profile.release_position_delta_limit_m == 0.0005
    assert profile.release_axis_delta_limit_deg == 1.0
    assert profile.preferred_backend is LatchBackendName.DYNAMIC_LOCK_FLAGS
    assert profile.fallback_backend is LatchBackendName.KINEMATIC
    assert profile.selected_backend is LatchBackendName.DYNAMIC_LOCK_FLAGS


def test_contact_snapshot_is_immutable_and_independent_of_live_force():
    snapshot = LatchedContactSnapshot(
        any_contact=True,
        camera_contact=False,
        sidewall_contact=True,
        source_physics_substep=120,
    )
    assert snapshot.sidewall_contact
    assert snapshot.source_physics_substep == 120


def test_latch_intent_and_reason_are_internal_not_action_results():
    assert LatchIntent.LOCK.value == "lock"
    assert LatchIntent.UNLOCK.value == "unlock"
    assert LatchReason.VIEW_TARGET.value == "view_target"
    assert not hasattr(ActionResult, "TARGET_REACHED")
```

- [ ] **Step 3: Run the focused test and confirm the expected import failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/eleven_action/test_hybrid_latch_contract.py -q
```

Expected: FAIL because the TASK-006 contracts do not yet exist.

- [ ] **Step 4: Implement the strict profile and dependency-light types**

Use this exact JSON authority:

```json
{
  "schema_version": "task006_hybrid_latched_v1",
  "physics_hz": 240,
  "policy_rgb_hz": 1,
  "view_error_limit_deg": 3.0,
  "support_drift_limit_m": 0.002,
  "release_window_s": 0.05,
  "release_position_delta_limit_m": 0.0005,
  "release_axis_delta_limit_deg": 1.0,
  "preferred_backend": "dynamic_lock_flags",
  "fallback_backend": "kinematic",
  "selected_backend": "dynamic_lock_flags"
}
```

Implement enums with these exact values:

```python
class LatchIntent(str, Enum):
    NONE = "none"
    LOCK = "lock"
    UNLOCK = "unlock"


class LatchReason(str, Enum):
    INITIAL = "initial"
    HOLD = "hold"
    VIEW_TARGET = "view_target"
    CAMERA_CONTACT = "camera_contact"
    ACTION_BOUNDARY = "action_boundary"
    REJECTED_MOVE = "rejected_move"


class LatchBackendName(str, Enum):
    DYNAMIC_LOCK_FLAGS = "dynamic_lock_flags"
    KINEMATIC = "kinematic"
```

Add `latched`, `latch_intent`, `latch_reason`, `latch_substep`, `policy_frame_ready`, and `latch_backend` to internal telemetry. Do not add a success/result label beyond the existing three.

- [ ] **Step 5: Run the contract tests and existing type/profile tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/eleven_action/test_hybrid_latch_contract.py \
  tests/eleven_action/test_types_and_profile.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the contract**

```bash
git add configs/eleven_action/hybrid_latched_profile.json \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action \
  tests/eleven_action/test_hybrid_latch_contract.py
git commit -m "feat: define hybrid latch contract"
```

---

### Task 2: Implement the pure 240-substep latch lifecycle

**Files:**

- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/controller.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/types.py`
- Modify: `tests/eleven_action/test_controller.py`
- Create: `tests/eleven_action/test_hybrid_latch_controller.py`

**Interfaces:**

- `ControllerStep` produces `wrench`, `telemetry`, `latch_intent`, and `latch_reason`.
- `ElevenActionController.set_latched_contact_snapshot(snapshot)` stores the eligibility snapshot.
- `ElevenActionController.confirm_latched(backend_name)` and `confirm_unlocked()` acknowledge runtime transitions.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_view_latches_on_first_target_and_drift_gate_substep(controller, state_factory):
    start = state_factory(optical_axis=(0.0, 0.0, 1.0))
    assert controller.submit(1, start, physics_substep=0)
    first = controller.step(start, physics_substep=0)
    assert first.latch_intent is LatchIntent.UNLOCK

    inside = state_factory(optical_axis=controller.target_axis_world, support_drift_m=0.0019)
    reached = controller.step(inside, physics_substep=1)
    assert reached.latch_intent is LatchIntent.LOCK
    assert reached.latch_reason is LatchReason.VIEW_TARGET
    assert reached.wrench.force_world_n.tolist() == [0.0, 0.0, 0.0]
    assert reached.wrench.torque_world_nm.tolist() == [0.0, 0.0, 0.0]


def test_view_does_not_latch_when_only_angle_passes(controller, state_factory):
    # Target error is within 3 degrees but support drift is over 2 mm.
    state = state_factory(optical_axis=controller.target_axis_world, support_drift_m=0.0021)
    step = controller.step(state, physics_substep=1)
    assert step.latch_intent is LatchIntent.NONE


def test_camera_contact_has_priority_and_returns_completed_at_240(controller, state_factory):
    controller.observe_contact(camera_contact_sample(physics_substep=20))
    locked = controller.step(state_factory(), physics_substep=20)
    assert locked.latch_reason is LatchReason.CAMERA_CONTACT
    assert locked.telemetry.constrained
    final = advance_to_substep_240(controller, state_factory())
    assert final.telemetry.result is ActionResult.COMPLETED


def test_timeout_latches_actual_state_without_fourth_result(controller, state_factory):
    final = run_240_substeps_without_target_entry(controller, state_factory())
    assert final.latch_reason is LatchReason.ACTION_BOUNDARY
    assert final.telemetry.result is ActionResult.COMPLETED


def test_hold_and_rejected_move_remain_latched_for_240_substeps(controller, state_factory):
    hold = run_action(controller, ElevenActionId.HOLD_VIEW, state_factory())
    rejected = run_action(controller, ElevenActionId.MOVE_SIDE_POS, state_factory(tilt_deg=45.0))
    assert hold.substeps == rejected.substeps == 240
    assert hold.result is ActionResult.COMPLETED
    assert rejected.result is ActionResult.REJECTED
    assert all_zero_wrench(hold.records + rejected.records)
```

- [ ] **Step 2: Run the new controller tests and confirm failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/eleven_action/test_hybrid_latch_controller.py -q
```

Expected: FAIL because latch intents and the latched lifecycle are absent.

- [ ] **Step 3: Implement the state machine without Isaac imports**

Replace READY_HOLD's continuously active support wrench with a latched zero-wrench state. Preserve one request ID and 240-substep counting. Use the full target-axis error, not scalar motion from the start axis:

```python
target_error_deg = math.degrees(math.acos(np.clip(
    float(current_axis @ self._target_axis), -1.0, 1.0
)))
target_gate = (
    target_error_deg <= self.latch_profile.view_error_limit_deg
    and support_drift <= self.latch_profile.support_drift_limit_m
)
```

Once `_latched_during_action` becomes true, emit zero wrench for every remaining substep. Do not end early. At substep 240, guarantee a LOCK intent if no earlier latch occurred, set the existing result, and enter `LATCHED_READY`.

Use the immutable latched contact snapshot for MOVE submit:

```python
eligible = (
    tilt_deg >= self.profile.move_min_tilt_deg
    and self.latched_contact_snapshot.sidewall_contact
)
```

Preserve the accepted MOVE 60/120/60 schedule and compute/freeze direction at substep 60.

- [ ] **Step 4: Run all pure controller tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/eleven_action/test_controller.py \
  tests/eleven_action/test_hybrid_latch_controller.py \
  tests/eleven_action/test_trajectory.py \
  tests/eleven_action/test_geometry_and_contacts.py -q
```

Expected: PASS, with no change to action IDs, timing, direction geometry, or public result enumeration.

- [ ] **Step 5: Commit the pure lifecycle**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action \
  tests/eleven_action/test_controller.py tests/eleven_action/test_hybrid_latch_controller.py
git commit -m "feat: add hybrid latch lifecycle"
```

---

### Task 3: Build and gate the GPU runtime latch backend

**Files:**

- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/eleven_action_latch.py`
- Create: `scripts/eleven_action/probe_hybrid_latch_backend.py`
- Create: `tests/eleven_action/test_latch_runtime_adapter.py`
- Modify: `configs/eleven_action/hybrid_latched_profile.json` only if the approved fallback is selected.

**Interfaces:**

- `CapsuleLatchRuntime.lock_current(state, reason) -> LatchReadback`
- `CapsuleLatchRuntime.unlock_zeroed(state) -> LatchReadback`
- `CapsuleLatchRuntime.readback() -> LatchReadback`
- `LatchReadback` includes backend, latched, position, quaternion, linear velocity, angular velocity, locked position-axis mask, locked rotation-axis mask, and kinematic flag.

- [ ] **Step 1: Write failing adapter tests using fake USD attributes and a fake capsule writer**

```python
def test_dynamic_flags_lock_all_six_axes_and_zero_velocity(fake_capsule, fake_physx_api):
    runtime = CapsuleLatchRuntime.dynamic_lock_flags(fake_capsule, fake_physx_api)
    before = fake_capsule.state()
    result = runtime.lock_current(before, LatchReason.INITIAL)
    assert fake_physx_api.locked_pos_axis == 0b111
    assert fake_physx_api.locked_rot_axis == 0b111
    assert fake_capsule.last_written_velocity == pytest.approx([0.0] * 6)
    assert result.latched


def test_unlock_clears_flags_but_does_not_change_pose(fake_capsule, fake_physx_api):
    runtime = CapsuleLatchRuntime.dynamic_lock_flags(fake_capsule, fake_physx_api)
    runtime.lock_current(fake_capsule.state(), LatchReason.INITIAL)
    pose_before = fake_capsule.pose.copy()
    result = runtime.unlock_zeroed(fake_capsule.state())
    assert fake_physx_api.locked_pos_axis == 0
    assert fake_physx_api.locked_rot_axis == 0
    assert fake_capsule.pose == pytest.approx(pose_before)
    assert not result.latched


def test_kinematic_fallback_is_not_selected_silently(fake_capsule, fake_physx_api):
    with pytest.raises(RuntimeError, match="fallback requires tracked profile selection"):
        CapsuleLatchRuntime.auto_fallback(fake_capsule, fake_physx_api)
```

- [ ] **Step 2: Run the adapter tests and confirm failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/eleven_action/test_latch_runtime_adapter.py -q
```

Expected: FAIL because the runtime adapter is absent.

- [ ] **Step 3: Implement the dynamic-lock-flags backend and explicit fallback**

Use `PhysxSchema.PhysxRigidBodyAPI` `lockedPosAxis` and `lockedRotAxis` bit masks with `0b111` for lock and `0` for unlock. Use the existing capsule root-velocity writer only at latch/unlock boundaries. Never call the root-pose writer during an action or latch.

For the fallback, set `UsdPhysics.RigidBodyAPI.kinematicEnabled` only when the tracked profile explicitly selects `kinematic`; retain the current pose and write zero velocity. Do not move a kinematic target.

Every lock/unlock call must clear the permanent COM wrench before changing authority and must read back the selected attributes. Readback mismatch raises a true FAULT.

- [ ] **Step 4: Run unit tests before launching Isaac Sim**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/eleven_action/test_latch_runtime_adapter.py \
  tests/eleven_action/test_runtime_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the mandatory GPU feasibility and paired-release probe**

```bash
./run_isaaclab.sh -p scripts/eleven_action/probe_hybrid_latch_backend.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 \
  --device cuda:0 --backend dynamic_lock_flags --seed 20260819 --headless
```

The script must exercise at least ten stratified contact states, hold each for one simulated second, and run at least ten paired release trials per motion action. It must write a compact summary and external JSONL containing pre/post API poses, flag readbacks, velocities, first-0.05-second paired trajectories, GPU backend identity, and the CCD warning.

Pass requires all lock readbacks equal `0b111`, all unlock readbacks equal `0`, no API-time pose jump, no locked pose drift beyond numerical tolerance, paired position difference at most 0.5 mm, and paired target-axis difference at most 1 degree.

- [ ] **Step 6: Apply the explicit backend decision**

If dynamic lock flags pass, keep `"selected_backend": "dynamic_lock_flags"` and continue. If they fail, preserve that evidence, change only `selected_backend` to `kinematic`, run the identical command with `--backend kinematic`, and continue only if the fallback passes. If both fail, write the report as `needs_decision`, commit the evidence index/report, push, and stop before Task 4.

- [ ] **Step 7: Commit the accepted runtime backend**

```bash
git add configs/eleven_action/hybrid_latched_profile.json \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/eleven_action_latch.py \
  scripts/eleven_action/probe_hybrid_latch_backend.py \
  tests/eleven_action/test_latch_runtime_adapter.py
git commit -m "feat: add gated six-dof latch runtime"
```

---

### Task 4: Integrate latch, contact snapshot, and zero-wrench boundary into the action term

**Files:**

- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/eleven_action.py`
- Modify: `tests/eleven_action/test_action_term.py`
- Modify: `tests/eleven_action/test_runtime_contract.py`

**Interfaces:**

- `ElevenActionTerm.latched` reports verified runtime state.
- `ElevenActionTerm.policy_frame_ready` becomes true only after the boundary barrier.
- `ElevenActionTerm.latched_contact_snapshot` exposes a read-only diagnostic copy.
- `ElevenActionTerm.consume_policy_frame_ready()` consumes one boundary event; it does not return RGB itself.

- [ ] **Step 1: Write failing integration tests**

```python
def test_reset_finishes_latched_with_zero_wrench_and_frame_ready(action_term):
    action_term.reset()
    assert action_term.latched
    assert action_term.policy_frame_ready
    assert action_term.current_wrench_is_zero


def test_motion_action_unlocks_and_relocks_at_boundary(action_term):
    action_term.process_action_id(1)
    first = action_term.apply_one_substep()
    assert not first.latched
    final = action_term.run_to_240()
    assert final.latched
    assert final.result is ActionResult.COMPLETED
    assert action_term.policy_frame_ready


def test_rejected_move_never_unlocks(action_term_with_upright_snapshot):
    action_term_with_upright_snapshot.process_action_id(9)
    records = action_term_with_upright_snapshot.run_to_240()
    assert all(row.latched for row in records)
    assert records[-1].result is ActionResult.REJECTED


def test_move_uses_latched_contact_not_force_during_locked_wait(action_term):
    action_term.set_latched_contact_snapshot(sidewall_contact=True)
    action_term.clear_live_contact_force()
    action_term.process_action_id(9)
    assert action_term.controller.accepted_move
```

- [ ] **Step 2: Confirm the integration tests fail**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/eleven_action/test_action_term.py \
  tests/eleven_action/test_runtime_contract.py -q
```

- [ ] **Step 3: Integrate runtime ordering exactly**

In each physics substep use this order:

```python
state = self._read_state()
self._observe_contact_sensor(state)
output = self.controller.step_or_submit(state, physics_substep=self._physics_substep)
self._write_zero_or_commanded_wrench(output.wrench)
self._apply_latch_intent(output.latch_intent, state, output.latch_reason)
self._confirm_runtime_state_before_frame_event(output)
```

When LOCK is applied, snapshot contact classification from the most recent dynamic contact history before clearing it. While locked, do not let missing ContactSensor forces erase that snapshot. On UNLOCK, leave the snapshot immutable for eligibility/audit until the next latch.

Expose `policy_frame_ready` only after result publication, latch readback, zero-wrench readback, and zero velocity readback. Continue rejecting/countering extra action requests during all 240 substeps.

- [ ] **Step 4: Run integration and prior TASK-005 tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/eleven_action -q
```

Expected: all tests pass; update old READY_HOLD assertions to the approved latched semantics rather than weakening them.

- [ ] **Step 5: Commit the action-term integration**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/eleven_action.py \
  tests/eleven_action
git commit -m "feat: integrate latch at action boundaries"
```

---

### Task 5: Synchronize one-hertz policy RGB and keyboard visualization

**Files:**

- Modify: `scripts/eleven_action/teleop_eleven_action.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/eleven_action_keyboard.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_eleven_action_stomach_env_cfg.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Modify: `tests/eleven_action/test_keyboard.py`
- Create: `tests/eleven_action/test_policy_frame_barrier.py`
- Modify: `tests/eleven_action/test_task_cfg.py`

**Interfaces:**

- Register `Template-Robotarm-Magnetic-Eleven-Action-Stomach-Lab-v0`.
- `PolicyFrameEvent` carries frame index, simulation time, request/result, latch backend/reason, pose, velocities, RGB timestamp, and both profile digests.
- Keyboard requests remain one keypress per action; active requests are discarded without queueing.

- [ ] **Step 1: Write failing camera-barrier and task-registration tests**

```python
def test_policy_frame_event_requires_confirmed_latch(barrier):
    with pytest.raises(RuntimeError, match="frame requested before latch confirmation"):
        barrier.publish(latched=False, wrench_zero=True, velocity_zero=True)


def test_policy_frame_event_requires_zero_velocity(barrier):
    with pytest.raises(RuntimeError, match="nonzero velocity"):
        barrier.publish(latched=True, wrench_zero=True, velocity_zero=False)


def test_stomach_task_uses_same_latch_and_dynamic_profiles(flat_cfg, stomach_cfg):
    assert flat_cfg.actions.eleven_action.latch_profile_path == stomach_cfg.actions.eleven_action.latch_profile_path
    assert flat_cfg.actions.eleven_action.dynamic_profile_path == stomach_cfg.actions.eleven_action.dynamic_profile_path
```

- [ ] **Step 2: Confirm failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/eleven_action/test_policy_frame_barrier.py \
  tests/eleven_action/test_keyboard.py \
  tests/eleven_action/test_task_cfg.py -q
```

- [ ] **Step 3: Implement the capture barrier and continuous visualization**

Make frame zero occur only after reset latch confirmation. After every 240-substep cycle, consume exactly one policy-frame event and then update the camera sensor. Keep continuous rendering at selected 60/120/240 target FPS, default 120; do not treat continuous preview frames as policy observations.

Terminal output for every action must include action ID/name, result, constrained, latch backend/reason/substep, total substeps, pose, linear/angular velocity at policy capture, contact snapshot, VIEW target error/support drift, MOVE signed displacement, and measured wall FPS.

- [ ] **Step 4: Run focused tests and a five-action visual smoke test**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/eleven_action/test_policy_frame_barrier.py \
  tests/eleven_action/test_keyboard.py \
  tests/eleven_action/test_task_cfg.py -q

./run_isaaclab.sh -p scripts/eleven_action/teleop_eleven_action.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 \
  --device cuda:0 --render_fps 120
```

Expected visual sequence: locked frame zero, keypress, one-second action, locked next frame, observation pause, next keypress. No action may start before a new keypress.

- [ ] **Step 5: Commit RGB synchronization and stomach registration**

```bash
git add scripts/eleven_action/teleop_eleven_action.py \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/eleven_action_keyboard.py \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab \
  tests/eleven_action
git commit -m "feat: synchronize latched policy frames"
```

---

### Task 6: Implement reproducible flat and stress acceptance

**Files:**

- Create: `scripts/eleven_action/validate_hybrid_latched_flat.py`
- Create: `scripts/eleven_action/validate_hybrid_latched_release.py`
- Create: `scripts/eleven_action/stress_hybrid_latched_sequence.py`
- Create: `tests/eleven_action/test_hybrid_acceptance_summary.py`

**Interfaces:**

- `evaluate_hybrid_flat_samples(samples) -> dict`
- `evaluate_release_pairs(pairs) -> dict`
- `build_fixed_stress_sequence(seed, length=100) -> tuple[int, ...]`

- [ ] **Step 1: Write failing summary tests with pass and fail fixtures**

```python
def test_view_requires_ten_unblocked_target_latches():
    rows = [passing_view_row() for _ in range(10)]
    assert evaluate_hybrid_flat_samples(rows)["status"] == "pass"
    rows[0]["target_error_deg"] = 3.01
    assert evaluate_hybrid_flat_samples(rows)["status"] == "fail"


def test_planned_valid_move_is_reclassified_and_replaced():
    rows = [passing_move_row() for _ in range(10)]
    rows.append(planned_valid_but_actual_invalid_row())
    summary = evaluate_hybrid_flat_samples(rows)
    assert summary["valid_count"] == 10
    assert summary["reclassified_count"] == 1


def test_every_policy_frame_is_latched_and_zero_velocity():
    rows = [passing_action_row(action_id=i) for i in range(11)]
    rows[3]["capture_angular_speed_rad_s"] = 1.0e-4
    assert evaluate_hybrid_flat_samples(rows)["status"] == "fail"


def test_release_pair_limits_are_inclusive():
    pair = passing_release_pair(position_delta_m=0.0005, axis_delta_deg=1.0)
    assert evaluate_release_pairs([pair])["status"] == "pass"
```

- [ ] **Step 2: Confirm the summary tests fail**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/eleven_action/test_hybrid_acceptance_summary.py -q
```

- [ ] **Step 3: Implement deterministic validators**

Use seed `20260819`. Require at least ten actual-valid randomized samples per action, ten invalid samples per MOVE split across angle/contact causes, ten paired release samples per motion action, and a fixed 100-ID no-reset sequence with every ID at least five times.

Store every attempt, including constrained, reclassified, rejected, and FAULT rows. Never stop collecting a VIEW merely because ten attempts occurred; continue until ten unblocked valid rows exist or a fixed 30-attempt ceiling is reached and report failure.

Every row must include `action_id`, `result`, `substeps`, `latch_backend`, `latch_reason`, `latch_substep`, `capture_latched`, capture pose/velocities, target error, support drift, MOVE eligibility source, signed displacement, constrained, contact snapshot, dynamic profile digest, latch profile digest, GPU dynamics state, CCD state/warning, and measured wall FPS.

- [ ] **Step 4: Run pure summary tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/eleven_action/test_hybrid_acceptance_summary.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the validators**

```bash
git add scripts/eleven_action/validate_hybrid_latched_flat.py \
  scripts/eleven_action/validate_hybrid_latched_release.py \
  scripts/eleven_action/stress_hybrid_latched_sequence.py \
  tests/eleven_action/test_hybrid_acceptance_summary.py
git commit -m "test: add hybrid latch acceptance gates"
```

---

### Task 7: Run the blocking flat gates and freeze evidence

**Files:**

- Modify only if an implementation defect is proven: TASK-006 source/tests from Tasks 1 through 6.
- Create after execution: `handoffs/reports/TASK-006-hybrid-latched-eleven-action-controller-report.md`
- External only: `logs/hybrid_latched_task006/**`

**Interfaces:**

- Produces the blocking flat, paired-release, and 100-action decisions.

- [ ] **Step 1: Run the selected backend release gate**

```bash
./run_isaaclab.sh -p scripts/eleven_action/validate_hybrid_latched_release.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 \
  --device cuda:0 --seed 20260819 --headless
```

Expected: every motion action has ten passing pairs; maximum position delta is at most 0.0005 m and maximum target-axis delta is at most 1 degree.

- [ ] **Step 2: Run randomized flat acceptance with camera enabled**

```bash
./run_isaaclab.sh -p scripts/eleven_action/validate_hybrid_latched_flat.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 \
  --device cuda:0 --seed 20260819 --render_fps 120 --headless
```

Expected: all eleven action gates pass, every normal/rejected action has 240 substeps, all boundary frames are latched with zero velocity, no FAULT occurs, every unblocked VIEW is within 3 degrees and 2 mm at target latch, and valid MOVE success is at least 90 percent for both signs.

- [ ] **Step 3: Run the no-reset stress sequence**

```bash
./run_isaaclab.sh -p scripts/eleven_action/stress_hybrid_latched_sequence.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 \
  --device cuda:0 --seed 20260819 --actions 100 --render_fps 120 --headless
```

Expected: 100 results, each ID appears at least five times, no FAULT, no queued/preempted request, and 101 latched policy frames including frame zero.

- [ ] **Step 4: Apply the flat stop rule**

If any release, per-action, capture-barrier, or stress gate fails, write `Disposition: needs_decision`, include the exact failing rows and metrics, skip all stomach tasks, run regressions, commit, push, and stop. Do not loosen 3 degrees, 2 mm, 0.5 mm, 1 degree, one second, 0.9mg, or any physics setting.

- [ ] **Step 5: Record evidence hashes**

```bash
find logs/hybrid_latched_task006 -type f -print0 | sort -z | xargs -0 sha256sum
```

Record every evidence path, byte size, and SHA-256 in the report. Do not add the external files to Git.

---

### Task 8: Migrate unchanged to stomach and collect user-facing visualization evidence

**Files:**

- Do not change controller, profiles, assets, stomach mesh, thresholds, timing, or force.
- Append results only: `handoffs/reports/TASK-006-hybrid-latched-eleven-action-controller-report.md`

**Interfaces:**

- Reuses the exact flat stress ID sequence and both exact profile digests.

- [ ] **Step 1: Verify digest equality before launch**

```bash
./run_isaaclab.sh -p scripts/eleven_action/inspect_eleven_action_prerequisites.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Stomach-Lab-v0 \
  --device cuda:0 --headless
```

Expected: selected backend, dynamic profile SHA-256, latch profile SHA-256, action map, 240 Hz, and GPU dynamics match the flat run exactly.

- [ ] **Step 2: Run the same 100-ID sequence in stomach**

```bash
./run_isaaclab.sh -p scripts/eleven_action/stress_hybrid_latched_sequence.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Stomach-Lab-v0 \
  --device cuda:0 --seed 20260819 --actions 100 --render_fps 120 --headless
```

Expected: initialization succeeds, no true system FAULT, every action uses 240 substeps, every boundary RGB event is latched/zero-velocity, and profile/sequence hashes match flat. Low MOVE displacement and normal constrained VIEW are observations, not stomach tuning failures.

- [ ] **Step 3: Launch keyboard visualization for user review**

```bash
./run_isaaclab.sh -p scripts/eleven_action/teleop_eleven_action.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Stomach-Lab-v0 \
  --device cuda:0 --render_fps 120
```

Exercise all eleven keys at least once and at least one sequence containing VIEW, VIEW, MOVE, HOLD, opposite MOVE. Preserve an external terminal log and video or screenshots showing locked observation pauses and continuous rendering.

- [ ] **Step 4: Apply the stomach disposition rule**

If flat passed but stomach cannot initialize or reaches a true FAULT, report `partial`. Normal collision, constrained VIEW, low displacement, and user-disliked motion do not authorize adaptation and do not change the objective evidence.

---

### Task 9: Run regressions, finalize the report, and push for Windows review

**Files:**

- Modify: `docs/ELEVEN_ACTION_DYNAMIC_CONTROLLER.md`
- Complete: `handoffs/reports/TASK-006-hybrid-latched-eleven-action-controller-report.md`
- Move after final disposition: `handoffs/active/TASK-006-hybrid-latched-eleven-action-controller.md` to the repository's completed handoff location only if the established workflow requires it.

**Interfaces:**

- Report disposition is exactly `complete`, `partial`, or `needs_decision` under the spec stop rules.

- [ ] **Step 1: Run focused and regression suites**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/eleven_action -q --disable-warnings
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/local_primitives tests/dynamic_force -q --disable-warnings
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/ideal_surface tests/coverage \
  tests/action_layer/test_atomic_protocol.py \
  tests/action_layer/test_executor.py \
  tests/action_layer/test_safety.py \
  tests/action_layer/test_atomic_stomach_teleop_cfg.py \
  tests/action_layer/test_atomic_keyboard_protocol.py -q --disable-warnings
```

Expected: all suites pass. Record exact counts and durations; do not write only “tests passed.”

- [ ] **Step 2: Complete the report with required evidence**

Include base/head/branch, commit series, exact commands, selected/fallback backend decision, API readbacks, both profile digests, all eleven randomized metrics, paired release maxima, policy-frame velocity checks, result counts, 100-ID sequence/hash, flat/stomach comparison, actual wall FPS, GPU/CCD warning, regressions, deviations, unverified items, and every external artifact path/size/SHA-256.

- [ ] **Step 3: Verify forbidden-change scope**

```bash
git diff --name-only 67b7bf44747f08422add0cee7e6b94280bbeff6d..HEAD
git diff --check 67b7bf44747f08422add0cee7e6b94280bbeff6d..HEAD
git status --short
```

Expected: no asset, mesh, material, mass, inertia, gravity, solver, CCD, VLM, RL, reward, or coverage file changed; no whitespace errors; only intended evidence remains untracked outside Git.

- [ ] **Step 4: Commit final documentation and report**

```bash
git add docs/ELEVEN_ACTION_DYNAMIC_CONTROLLER.md \
  handoffs/active/TASK-006-hybrid-latched-eleven-action-controller.md \
  handoffs/reports/TASK-006-hybrid-latched-eleven-action-controller-report.md
git commit -m "docs: report TASK-006 hybrid latch validation"
```

- [ ] **Step 5: Push without merging**

```bash
git push -u origin feature/TASK-006-hybrid-latched-v1
```

Expected: Linux returns branch name, final HEAD, disposition, report path, selected latch backend, profile digests, key metrics, regression counts, and external evidence hashes to the Windows design side. Do not merge `main`.
