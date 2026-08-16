# Dynamic Capsule Force Teleoperation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated Isaac Lab stomach task in which a non-kinematic capsule receives a bounded world-frame force from six held keyboard directions and moves continuously under PhysX gravity, inertia, friction, damping, and stomach-wall contact.

**Architecture:** Preserve every existing task and add a dedicated three-dimensional force action term, held-key adapter, 60 Hz teleoperation loop, and diagnostic validators. The action term applies force at the capsule center of mass on every 240 Hz physics substep and never writes capsule pose or velocity outside reset; PhysX owns all runtime state evolution and contact response.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Isaac Lab/Isaac Sim 6.0 APIs pinned by the repository, USD/PhysX rigid-body and CCD APIs verified during preflight, Gymnasium, pytest, Kit keyboard events, and JSON/JSONL evidence.

## Global Constraints

- Work only on `feature/TASK-003-dynamic-capsule-force-teleop`, created from the exact head of `workflow/TASK-003-dynamic-capsule-force-teleop`.
- Register `Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0`; do not repurpose or change an existing task.
- Use one environment, `sim.dt=1/240`, `decimation=4`, `sim.render_interval=4`, and capsule-camera `update_period=1/30`.
- Keep the capsule non-kinematic with gravity enabled for every non-reset physics step.
- Permit direct capsule root-pose and root-velocity writes only through the existing reset event before normal stepping.
- Forbid pose writes, velocity writes, surface projection, pose clamping, penetration recovery, target tracking, hidden support forces, and surface constraints in every new runtime and validation path.
- Use exactly one normalized action vector `[Fx, Fy, Fz]` in world coordinates, clamped to unit norm and scaled by `force_weight_ratio * live_mass_kg * 9.81` newtons.
- Use `force_weight_ratio=0.5` by default and reject ratios outside `0 < ratio <= 2.0`.
- Apply force at the capsule center of mass and command exactly zero torque.
- Include no magnetic action term, magnetic collision bridge, ideal-surface action, robot joint action, VLM, RL, reward, coverage optimization, or deformable tissue.
- Reuse the delivered capsule collider, stomach collider, mass, inertia, gravity, damping, friction, restitution, velocity limits, and maximum depenetration velocity without tuning.
- Enable and verify scene-level and capsule-body CCD using the installed API; stop with `needs_decision` if this requires a shared asset edit or cannot be confirmed.
- Treat penetration, instability, nonfinite state, and unintended mesh escape as evidence, not states to correct.
- Keep logs, videos, screenshots, JSONL traces, and generated summaries outside Git and report their absolute paths, byte sizes, and SHA-256 hashes.
- Use test-driven development and commit after each independently passing task.
- The repository ignores `/tests/` and `docs/superpowers/`; use `git add -f` only for the explicitly named new test, spec, and plan files.

---

## Expected File Map

```text
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/dynamic_force.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/dynamic_force_action.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_dynamic_force_stomach_env_cfg.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/dynamic_force_keyboard.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py
scripts/dynamic_force/inspect_dynamic_force_prerequisites.py
scripts/dynamic_force/teleop_dynamic_force_stomach.py
scripts/dynamic_force/validate_dynamic_force_stomach.py
tests/dynamic_force/conftest.py
tests/dynamic_force/test_force_contract.py
tests/dynamic_force/test_dynamic_force_keyboard.py
tests/dynamic_force/test_dynamic_force_task_cfg.py
tests/dynamic_force/test_dynamic_force_preflight.py
docs/DYNAMIC_CAPSULE_FORCE_TELEOP.md
handoffs/reports/TASK-003-dynamic-capsule-force-teleop-report.md
```

## Task 1: Establish the Dynamic-Physics Preflight Gate

**Files:**
- Create: `scripts/dynamic_force/inspect_dynamic_force_prerequisites.py`
- Create: `tests/dynamic_force/conftest.py`
- Create: `tests/dynamic_force/test_dynamic_force_preflight.py`
- Output outside Git: `logs/dynamic_force_preflight/<timestamp>/prerequisites.json`

**Interfaces:**
- Consumes: the delivered stomach scene, capsule prim, contact sensor, installed Isaac Lab/PhysX APIs, and planned task configuration.
- Produces: `build_preflight_report(env, task_id, repository) -> dict` and `validate_preflight_report(report: dict) -> None`.

- [ ] **Step 1: Write the failing schema and real-dynamics tests**

```python
def test_gate_requires_true_dynamic_capsule_and_ccd(valid_report):
    report = valid_report()
    validate_preflight_report(report)
    assert report["capsule"]["kinematic_enabled"] is False
    assert report["capsule"]["gravity_enabled"] is True
    assert report["capsule"]["ccd_enabled"] is True
    assert report["physics"]["scene_ccd_enabled"] is True


def test_gate_rejects_runtime_pose_writer(valid_report):
    report = valid_report()
    report["runtime_contract"]["forbidden_calls"] = ["write_root_pose_to_sim"]
    with pytest.raises(ValueError, match="forbidden runtime state writer"):
        validate_preflight_report(report)
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force/test_dynamic_force_preflight.py -q
```

Expected: FAIL because `inspect_dynamic_force_prerequisites.py` does not exist.

- [ ] **Step 3: Implement the report schema and gate without changing simulation state**

The report must contain exact keys `repository`, `task`, `physics`, `capsule`, `stomach`, `contact_sensor`, `runtime_contract`, and `gate`. Validate the fixed rates, capsule shape and dimensions, finite positive mass and inertia, non-kinematic state, gravity, body and scene CCD, stomach static collision, contact sensor, and forbidden-call scan.

```python
def build_gate(report: dict) -> dict:
    failures: list[str] = []
    capsule = report["capsule"]
    physics = report["physics"]
    if capsule["kinematic_enabled"]:
        failures.append("capsule is kinematic")
    if not capsule["gravity_enabled"]:
        failures.append("capsule gravity is disabled")
    if not capsule["ccd_enabled"] or not physics["scene_ccd_enabled"]:
        failures.append("CCD is not active at scene and body levels")
    if report["runtime_contract"]["forbidden_calls"]:
        failures.append("forbidden runtime state writer")
    return {"status": "pass" if not failures else "needs_decision", "failures": failures}
```

- [ ] **Step 4: Inspect APIs and live properties without silently guessing version-dependent fields**

Use `UsdPhysics.RigidBodyAPI`, the installed `PhysxSchema.PhysxRigidBodyAPI`, the live rigid-body view, and the task configuration. Record the exact attribute names used for kinematic state, gravity, body CCD, scene CCD, mass, inertia, center of mass, collision enablement, and timing.

- [ ] **Step 5: Run the schema tests**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force/test_dynamic_force_preflight.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the preflight contract**

```bash
git add -f scripts/dynamic_force/inspect_dynamic_force_prerequisites.py tests/dynamic_force/conftest.py tests/dynamic_force/test_dynamic_force_preflight.py
git commit -m "test: define dynamic capsule physics preflight"
```

## Task 2: Implement the Pure Force and Held-Key Contracts

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/dynamic_force.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/dynamic_force_keyboard.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py`
- Create: `tests/dynamic_force/test_force_contract.py`
- Create: `tests/dynamic_force/test_dynamic_force_keyboard.py`

**Interfaces:**
- Produces: `validate_force_weight_ratio(value: float) -> float`.
- Produces: `normalize_force_direction(value: np.ndarray) -> np.ndarray`.
- Produces: `force_world_from_action(action: np.ndarray, mass_kg: float, force_weight_ratio: float) -> np.ndarray`.
- Produces: `DynamicForceCommandKind`, `DynamicForceCommand`, `DynamicForceKeyboard.direction -> np.ndarray`, and `DynamicForceKeyboard.key_event(key: str, is_down: bool) -> DynamicForceCommand | None`.

- [ ] **Step 1: Write the failing force-contract tests**

```python
def test_half_weight_force_uses_live_mass():
    force = force_world_from_action(np.array([1.0, 0.0, 0.0]), 0.0057, 0.5)
    np.testing.assert_allclose(force, [0.5 * 0.0057 * 9.81, 0.0, 0.0])


def test_diagonal_is_norm_limited():
    direction = normalize_force_direction(np.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 2.01])
def test_ratio_outside_contract_is_rejected(ratio):
    with pytest.raises(ValueError):
        validate_force_weight_ratio(ratio)
```

- [ ] **Step 2: Write the failing six-direction held-key tests**

```python
@pytest.mark.parametrize(
    ("key", "expected"),
    [("W", [1, 0, 0]), ("S", [-1, 0, 0]), ("A", [0, 1, 0]),
     ("D", [0, -1, 0]), ("Q", [0, 0, 1]), ("E", [0, 0, -1])],
)
def test_each_key_selects_one_world_axis(key, expected):
    keyboard = DynamicForceKeyboard()
    keyboard.key_event(key, True)
    np.testing.assert_array_equal(keyboard.direction, expected)


def test_release_and_opposites_return_zero():
    keyboard = DynamicForceKeyboard()
    keyboard.key_event("W", True)
    keyboard.key_event("S", True)
    np.testing.assert_array_equal(keyboard.direction, [0, 0, 0])
    keyboard.key_event("S", False)
    np.testing.assert_array_equal(keyboard.direction, [1, 0, 0])
    keyboard.key_event("W", False)
    np.testing.assert_array_equal(keyboard.direction, [0, 0, 0])
```

- [ ] **Step 3: Run both tests and verify missing implementation failures**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force/test_force_contract.py tests/dynamic_force/test_dynamic_force_keyboard.py -q
```

Expected: FAIL from missing modules or symbols.

- [ ] **Step 4: Implement the minimal pure force functions**

```python
GRAVITY_M_S2 = 9.81
DEFAULT_FORCE_WEIGHT_RATIO = 0.5


def normalize_force_direction(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    vector = np.clip(vector, -1.0, 1.0)
    norm = float(np.linalg.norm(vector))
    return vector if norm <= 1.0 else vector / norm


def force_world_from_action(action, mass_kg, force_weight_ratio):
    ratio = validate_force_weight_ratio(force_weight_ratio)
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("capsule mass must be finite and positive")
    return normalize_force_direction(action) * ratio * mass_kg * GRAVITY_M_S2
```

- [ ] **Step 5: Implement level-triggered keyboard state**

Use a set of held force keys. Recompute direction from all held keys after every press and release. Define the task-local enum `DynamicForceCommandKind` with `CLEAR`, `RESET`, `SNAPSHOT`, and `EXIT`, plus a frozen `DynamicForceCommand` dataclass. Return special commands only for `SPACE`, `BACKSPACE`, `F12`, and `ESCAPE`; force keys are read through the `direction` property and must not use rising-edge suppression. Do not add a new value to the shared atomic-action `CommandKind` enum.

```python
FORCE_KEYS = {
    "W": np.array([1.0, 0.0, 0.0]), "S": np.array([-1.0, 0.0, 0.0]),
    "A": np.array([0.0, 1.0, 0.0]), "D": np.array([0.0, -1.0, 0.0]),
    "Q": np.array([0.0, 0.0, 1.0]), "E": np.array([0.0, 0.0, -1.0]),
}
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force/test_force_contract.py tests/dynamic_force/test_dynamic_force_keyboard.py -q
```

Expected: PASS.

```bash
git add -f source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/dynamic_force.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/dynamic_force_keyboard.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py tests/dynamic_force/test_force_contract.py tests/dynamic_force/test_dynamic_force_keyboard.py
git commit -m "feat: add six-direction force input contract"
```

## Task 3: Add the Dynamic Force Action and Isolated Task

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/dynamic_force_action.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_dynamic_force_stomach_env_cfg.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Create: `tests/dynamic_force/test_dynamic_force_task_cfg.py`

**Interfaces:**
- Consumes: pure force functions from Task 2 and the existing scene asset named `capsule`.
- Produces: `DynamicForceAction(ActionTerm)` with `action_dim == 3`, `applied_force_world`, and `mass_kg` properties.
- Produces: `DynamicForceActionTermCfg(force_weight_ratio: float = 0.5, asset_name: str = "capsule")`.
- Produces: `RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg` and the frozen Gym task ID.

- [ ] **Step 1: Write failing task configuration and isolation tests**

```python
def test_dynamic_force_task_has_frozen_rates_and_action():
    cfg = RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg()
    assert cfg.scene.num_envs == 1
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.decimation == 4
    assert cfg.sim.render_interval == 4
    assert cfg.scene.capsule_camera.update_period == 1.0 / 30.0
    assert _term_names(cfg.actions) == ["dynamic_force"]
    assert cfg.actions.dynamic_force.force_weight_ratio == 0.5


def test_task_contains_no_forbidden_actuator():
    cfg = RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg()
    assert "magnetic_physics" not in _term_names(cfg.actions)
    assert "ideal_surface" not in _term_names(cfg.actions)
    assert "magnetic_collision_bridge" not in _term_names(cfg.events)
```

- [ ] **Step 2: Run the focused test and verify registration/configuration failure**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force/test_dynamic_force_task_cfg.py -q
```

Expected: FAIL because the task and action term are absent.

- [ ] **Step 3: Implement the action term without any state setter**

The constructor shall resolve the capsule, read one finite positive live mass, verify or set the task-local rigid body to non-kinematic, verify gravity, and enable body CCD. `process_actions` clamps and norm-limits the three-vector. `apply_actions` computes force from live mass and refreshes the capsule's permanent wrench composer in the world frame with zero torque at the center of mass.

```python
class DynamicForceAction(ActionTerm):
    @property
    def action_dim(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor) -> None:
        clipped = torch.clamp(actions, -1.0, 1.0)
        norm = torch.linalg.vector_norm(clipped, dim=-1, keepdim=True)
        self._processed_actions = clipped / torch.clamp(norm, min=1.0)

    def apply_actions(self) -> None:
        scale = self.cfg.force_weight_ratio * self._mass_kg * 9.81
        self._applied_force_world = self._processed_actions * scale
        zero_torque = torch.zeros_like(self._applied_force_world)
        self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._applied_force_world[:, None, :],
            torques=zero_torque[:, None, :],
            positions=None,
            body_ids=[0],
            env_ids=self._all_env_ids,
            is_global=True,
        )
```

Preflight must verify that `positions=None` means application at the center of mass for the installed wrench-composer API. If the installed API requires an explicit point, pass the live authored center-of-mass offset in the required local frame. Do not substitute the link origin unless preflight proves it equals the center of mass.

Do not call `write_root_pose_to_sim`, `write_root_velocity_to_sim`, `set_transforms`, or `set_velocities`. Do not add a contact-normal or damping force; passive authored damping remains the only non-contact damping.

- [ ] **Step 4: Implement the isolated task configuration**

Mirror the ideal task's isolated RGB observation pattern, but use the new three-dimensional action and only the reset event. Set `sim.physx.enable_ccd=True` through the installed configuration field verified in Task 1. Use timeout termination only; do not add a pose-based recovery or termination.

```python
@configclass
class DynamicForceActionsCfg:
    dynamic_force: mdp.DynamicForceActionTermCfg = mdp.DynamicForceActionTermCfg(
        force_weight_ratio=0.5
    )


def __post_init__(self) -> None:
    super().__post_init__()
    self.scene.num_envs = 1
    self.decimation = 4
    self.sim.dt = 1.0 / 240.0
    self.sim.render_interval = 4
    self.scene.capsule_camera.update_period = 1.0 / 30.0
    self.sim.physx.enable_ccd = True
```

- [ ] **Step 5: Add task registration and exports**

Register only `Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0` with `ManagerBasedRLEnv` and the new config entry point. Export `DynamicForceAction` and `DynamicForceActionTermCfg` through `mdp/__init__.py`.

- [ ] **Step 6: Add a static forbidden-writer test**

Read the new action module as text and assert that none of `write_root_pose`, `write_root_velocity`, `set_transforms`, or `set_velocities` occurs. This test complements live preflight; it does not inspect the allowed shared reset event.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force/test_dynamic_force_task_cfg.py tests/dynamic_force/test_force_contract.py -q
```

Expected: PASS.

```bash
git add -f source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/dynamic_force_action.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_dynamic_force_stomach_env_cfg.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py tests/dynamic_force/test_dynamic_force_task_cfg.py
git commit -m "feat: add dynamic capsule force task"
```

## Task 4: Build the Continuous Rendered Keyboard Launcher

**Files:**
- Create: `scripts/dynamic_force/teleop_dynamic_force_stomach.py`
- Modify: `tests/dynamic_force/test_dynamic_force_keyboard.py`
- Output outside Git: `logs/dynamic_force_teleop/<timestamp>/samples.jsonl`
- Output outside Git: `logs/dynamic_force_teleop/<timestamp>/session.json`
- Output outside Git: `logs/dynamic_force_teleop/<timestamp>/snapshots/`

**Interfaces:**
- Consumes: `DynamicForceKeyboard`, the new Gym task, `DynamicForceAction.applied_force_world`, capsule/contact state, and Kit keyboard events.
- Produces: a continuous `env.step(action)` loop and one JSONL record per 60 Hz environment step.

- [ ] **Step 1: Extend keyboard tests for Space and special commands**

```python
def test_space_clears_force_without_latching():
    keyboard = DynamicForceKeyboard()
    keyboard.key_event("W", True)
    command = keyboard.key_event("SPACE", True)
    assert command.kind is DynamicForceCommandKind.CLEAR
    np.testing.assert_array_equal(keyboard.direction, [0, 0, 0])


@pytest.mark.parametrize(
    ("key", "kind"),
    [("BACKSPACE", DynamicForceCommandKind.RESET),
     ("F12", DynamicForceCommandKind.SNAPSHOT),
     ("ESC", DynamicForceCommandKind.EXIT)],
)
def test_special_commands(key, kind):
    keyboard = DynamicForceKeyboard()
    assert keyboard.key_event(key, True).kind is kind
```

- [ ] **Step 2: Implement Kit press and release subscription**

The callback must forward both `KEY_PRESS` and `KEY_RELEASE`. It must not suppress repeated force-key state, and it must queue only special commands. The main loop reads `keyboard.direction` every step.

- [ ] **Step 3: Implement the continuous simulation loop**

Call `env.step` on every loop iteration, including when the force is zero. Never use `simulation_app.update()` as a substitute for advancing the physics task while idle.

```python
while simulation_app.is_running() and not exit_requested:
    direction = keyboard.direction
    action = torch.as_tensor(direction, device=env.unwrapped.device).reshape(1, 3)
    observation, reward, terminated, truncated, info = env.step(action)
    recorder.append(sample_from_env(env, term, direction))
    handle_special_commands()
```

- [ ] **Step 4: Add fixed-ratio and scripted smoke arguments**

Support `--force_weight_ratio 0.5`, `--max_steps 0`, and `--scripted_sequence "+x:0.5,zero:0.25,-x:0.5"`. The scripted path must use the same three-vector action as the keyboard and exists only for reproducible rendered smoke tests.

- [ ] **Step 5: Add 60 Hz diagnostics and state-change output**

Each JSONL row shall include `sim_time_s`, `step`, `reset_index`, `position_world_m`, `quaternion_wxyz`, `linear_velocity_world_m_s`, `angular_velocity_world_rad_s`, `direction_world`, `force_world_n`, `torque_world_nm`, and `contact_force_world_n`. Print force changes and a one-line state summary at 10 Hz; do not print every 240 Hz substep.

- [ ] **Step 6: Run pure launcher/keyboard tests and compile checks**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force/test_dynamic_force_keyboard.py -q
./run_isaaclab.sh -p -m compileall -q scripts/dynamic_force source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop
```

Expected: PASS with no syntax errors.

- [ ] **Step 7: Commit the launcher**

```bash
git add -f scripts/dynamic_force/teleop_dynamic_force_stomach.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/dynamic_force_keyboard.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py tests/dynamic_force/test_dynamic_force_keyboard.py
git commit -m "feat: add continuous capsule force teleoperation"
```

## Task 5: Validate Gravity, Contact, Six Forces, and Continuity

**Files:**
- Create: `scripts/dynamic_force/validate_dynamic_force_stomach.py`
- Modify: `tests/dynamic_force/test_dynamic_force_preflight.py`
- Output outside Git: `logs/dynamic_force_validation/<timestamp>/summary.json`
- Output outside Git: `logs/dynamic_force_validation/<timestamp>/samples.jsonl`

**Interfaces:**
- Consumes: new task, action term diagnostics, contact sensor, and read-only ideal spherocylinder/surface assessment utilities.
- Produces: one summary with `preflight`, `settling`, `directions`, `continuity`, `contact`, and `status`.

- [ ] **Step 1: Write failing validation-summary tests**

```python
def test_summary_requires_all_six_signed_directions(valid_summary):
    summary = valid_summary()
    assert set(summary["directions"]) == {"+x", "-x", "+y", "-y", "+z", "-z"}


def test_nonfinite_or_forbidden_writer_fails(valid_summary):
    summary = valid_summary()
    summary["continuity"]["nonfinite_samples"] = 1
    assert evaluate_summary(summary)["status"] == "fail"
    summary = valid_summary()
    summary["preflight"]["runtime_contract"]["forbidden_calls"] = ["set_transforms"]
    assert evaluate_summary(summary)["status"] == "needs_decision"
```

- [ ] **Step 2: Implement the no-input settling phase**

Reset once, command zero force, and step for 3.0 simulated seconds. Record the first and final state, confirm gravity remains enabled, record whether the capsule moved from its initial equilibrium, whether stomach contact was observed, maximum speed, maximum angular speed, nonfinite count, and read-only surface clearance. Do not require visible motion when the reset state is already a stable contact equilibrium, and do not reposition the capsule if contact fails.

- [ ] **Step 3: Implement six independent signed-force phases**

For each of `+x`, `-x`, `+y`, `-y`, `+z`, and `-z`, reset, settle for 1.0 second, apply exactly `0.5mg` for 0.5 second, then release to zero for 0.5 second. Record requested direction, expected force, measured action-term force, displacement, velocity change, contact-force range, clearance range, and maximum observed physics displacement.

- [ ] **Step 4: Define acceptance without assuming free-space motion inside the stomach**

Require exact wrench agreement within `1e-6 N`, zero commanded torque, finite states, active physics progression, no forbidden writers, and bounded per-step displacement. Report but do not fail a direction solely because wall contact blocks or redirects displacement.

Make the action term record read-only capsule positions at successive `apply_actions` calls so the validator receives 240 Hz substep displacement evidence. Compute the continuity bound from the authored maximum linear velocity and physics timestep rather than from the 60 Hz rendered frame difference:

```python
allowed_step_m = authored_max_linear_velocity_m_s * physics_dt_s + 1.0e-5
```

Fail sustained clearance decrease across consecutive samples, nonfinite state, or a displacement discontinuity above the bound. Report any crossing of a known mesh boundary separately as `boundary_escape`; do not repair it.

- [ ] **Step 5: Run the mandatory live preflight**

Run:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/inspect_dynamic_force_prerequisites.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0
```

Expected: `DYNAMIC_FORCE_PREFLIGHT_PASS`. If the result is `needs_decision`, stop before further implementation and write the report.

- [ ] **Step 6: Run deterministic headless acceptance**

Run:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/validate_dynamic_force_stomach.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 --seed 42 --force_weight_ratio 0.5 --headless
```

Expected: `DYNAMIC_FORCE_VALIDATION_PASS`, six direction records, finite state, exact wrench, observed physics progression, and no forbidden correction.

- [ ] **Step 7: Commit the validator**

```bash
git add -f scripts/dynamic_force/validate_dynamic_force_stomach.py tests/dynamic_force/test_dynamic_force_preflight.py
git commit -m "test: validate dynamic capsule force and contact"
```

## Task 6: Run Rendered Acceptance and Existing Regressions

**Files:**
- No tracked source changes unless a failing test identifies a TASK-003 defect.
- Output outside Git: `logs/dynamic_force_teleop/<timestamp>/`

**Interfaces:**
- Consumes: all completed TASK-003 components and delivered regression suites.
- Produces: rendered artifacts and exact command results for the report.

- [ ] **Step 1: Run a scripted rendered continuity smoke**

Run:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/teleop_dynamic_force_stomach.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 --force_weight_ratio 0.5 --scripted_sequence "+x:0.5,zero:0.25,-x:0.5,zero:0.25" --max_steps 120 --viz kit
```

Expected: the task initializes, advances 120 environment steps with four physics substeps each, renders continuously, writes 120 samples, and exits cleanly.

- [ ] **Step 2: Perform the manual keyboard checklist when a human is available**

Run:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/teleop_dynamic_force_stomach.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 --force_weight_ratio 0.5 --viz kit
```

Press and release W, S, A, D, Q, and E separately. Confirm that motion and camera imagery update continuously, force returns to zero on release, contact can block or redirect motion, and no visible pose snap occurs. If Linux automation cannot complete subjective review, mark it unverified rather than passing it by assertion.

- [ ] **Step 3: Run all focused TASK-003 tests**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/dynamic_force -q --disable-warnings
```

Expected: PASS.

- [ ] **Step 4: Run TASK-002 and delivered pure regressions**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface tests/coverage tests/action_layer/test_atomic_protocol.py tests/action_layer/test_executor.py tests/action_layer/test_safety.py tests/action_layer/test_atomic_stomach_teleop_cfg.py tests/action_layer/test_atomic_keyboard_protocol.py -q --disable-warnings
```

Expected: PASS with the same delivered test counts unless new TASK-003-only tests legitimately change collection totals.

- [ ] **Step 5: Run delivered live integration regressions**

Run:

```bash
./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py --num_envs 1 --coverage_samples 5
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_table.py --num_envs 1 --max_steps_per_action 60 --viz kit
./run_isaaclab.sh -p scripts/zero_agent.py --task Template-Robotarm-Magnetic-Table-Lab-v0 --num_envs 1 --max_steps 5 --viz kit
```

Expected: delivered coverage geometry, P0 stomach, eleven-action table, and legacy 9D table results remain passing.

- [ ] **Step 6: Run hygiene checks**

Run:

```bash
./run_isaaclab.sh -p -m compileall -q scripts/dynamic_force source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop
git diff --check
git status --short
```

Expected: compile and diff checks pass; only TASK-003 files and the required report are changed.

## Task 7: Document Operation and Deliver Evidence

**Files:**
- Create: `docs/DYNAMIC_CAPSULE_FORCE_TELEOP.md`
- Create: `handoffs/reports/TASK-003-dynamic-capsule-force-teleop-report.md`

**Interfaces:**
- Consumes: verified commands, observed results, preflight JSON, validation summary, rendered artifacts, and Git state.
- Produces: operator documentation and the authoritative Linux handoff report.

- [ ] **Step 1: Write operator documentation**

Document the real-dynamics contract, task ID, six keys, world-frame convention, default `0.5mg` force, rate separation, reset/snapshot/exit keys, commands, artifact locations, interpretation of contact-constrained motion, known 21 mesh boundary edges, and the absence of torque, magnetic actuation, tissue deformation, and pose correction.

- [ ] **Step 2: Write the report from observed evidence only**

The report must state status, planning base, implementation head before report, branch, exact task and API properties, capsule mass/inertia/collider, CCD evidence, rates, every command and observed result, exact applied force, settling/contact results, six direction summaries, continuity metrics, regressions, deviations, unverified claims, and external artifact paths with byte sizes and SHA-256 hashes.

- [ ] **Step 3: Explicitly classify collision behavior**

Use `complete` only when real dynamic motion and contact are observed without forbidden correction and all mandatory checks pass. Use `partial` if force application works but sustained penetration, instability, or unintended escape remains. Use `needs_decision` if the dynamic/CCD contract cannot be established without changing shared assets. Do not tune physics parameters inside this task to obtain a passing label.

- [ ] **Step 4: Commit documentation and report**

```bash
git add -f docs/DYNAMIC_CAPSULE_FORCE_TELEOP.md handoffs/reports/TASK-003-dynamic-capsule-force-teleop-report.md
git commit -m "docs: report dynamic capsule force teleoperation"
```

- [ ] **Step 5: Push without merging**

```bash
git push -u origin feature/TASK-003-dynamic-capsule-force-teleop
```

Expected: the feature branch is available to Windows for review. Linux does not merge it.

## Final Review Gate

Before reporting completion, verify that the implementation answers the user's actual requirement: the capsule is a real dynamic rigid body, not a kinematic target disguised behind small pose increments. Search all new runtime files for root-state setters, confirm the external wrench is active at every physics substep, confirm the six keyboard directions are level-triggered, confirm force release produces zero actuator force, confirm the viewer and camera rates are continuous, and confirm every penetration or instability is exposed in evidence rather than repaired.
