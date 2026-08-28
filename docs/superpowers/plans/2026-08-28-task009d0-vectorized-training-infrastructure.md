# TASK-009D-0 Vectorized Training Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The Linux implementation host does not currently provide these optional skills, so their absence is not a blocker; execute the same tasks manually in order and preserve every review gate.

**Goal:** Build a non-destructive, exact-GPU, synchronous multi-environment Isaac Lab task that preserves the accepted single-environment physics and produces isolated 120-second coverage episodes ready for TASK-009D-1 training.

**Architecture:** Keep the accepted TASK-009C task unchanged and register a new `ManagerBasedRLEnv` subclass. The new task vectorizes parameterized forces, RGB boundary synchronization, pose reset, exact visibility, area-weighted coverage, and raw coverage reward while sharing immutable stomach geometry. Training episodes reset all environments together, execute ten uncounted HOLD boundaries, record one positive `C0`, then execute exactly 1200 formal actions.

**Tech Stack:** Isaac Lab 3.0 source checkout, Isaac Sim 6.0.0.1 target configuration, Python 3.12, PyTorch 2.11, Warp mesh raycast, PhysX GPU, RTX camera, NumPy, Gymnasium, pytest.

**Spec:** `docs/design/2026-08-28-task009d0-vectorized-training-infrastructure-design.md`

## Global Constraints

- Code baseline is `7c4c5a18780b980ad3882ce75f1d64733fc3080d`; the Windows planning commits must remain descendants of this baseline.
- Preserve the existing task ID `Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0` and its implementation unchanged except for shared pure helpers that pass all legacy tests.
- Register the new task as `Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0`.
- Physics is 240 Hz, control/RGB/coverage is 10 Hz, and every action spans exactly 24 physics substeps.
- Formal episode duration is 120 seconds with 1200 actions and 1201 coverage points including `C0`; reset stabilization is ten HOLD boundaries outside the formal budget.
- Camera remains 1280 by 720 with the accepted 120-degree circular field of view. Do not change intrinsics, clipping, lighting, or the 70 mm visibility limit for performance.
- MOVE total force remains 0.70 to 1.40 mg, VIEW camera-end force remains 0.20 to 0.50 mg, and UP camera-end force remains 0.80 to 1.05 mg.
- Coverage remains exact area-weighted vertex coverage with first-hit occlusion, lumen-facing normal, frozen unreachable mask, 24529 raw target vertices, 17055 reachable positive-weight vertices, and 49047 triangles.
- Actor observations contain only RGB and the previous actual action encoded as six one-hot mode values plus one strength value. HOLD history strength is zero.
- Training reads only the 1000 training poses. Validation and test split access must fail in training mode.
- Normal collision, obstruction, zero new coverage, HOLD, low coverage, and timeout are valid outcomes. Non-finite state, RGB failure, frame mismatch, ray failure, decreasing cumulative coverage, or undefined lateral direction are fatal batch errors.
- Do not add CNN, GRU, Actor, Critic model, PPO, reward scaling, stagnation penalty, VLM, disturbance ranges, or asset edits in TASK-009D-0.
- Generated RGB, masks, timing traces, logs, and benchmark data stay under `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/`; Git receives only code, tests, frozen small configuration, manifests, documentation, and reports.

---

## File and Interface Map

`configs/task009d0/vectorized_training_candidates_v1.json` is the only pre-benchmark source for clocks, hashes, strength ranges, candidate environment counts, benchmark duration, and selection rules. `configs/task009d0/vectorized_training_frozen_v1.json` is created only after Gate 5 selects an environment count from measured evidence.

`runtime/task009d0_config.py` validates both configurations and loads the frozen pose library without trusting caller-provided coordinates. `runtime/task009d0_pose_batch.py` owns reproducible split-safe pose sampling.

`controllers/vectorized_parameterized_force.py` contains simulator-independent batched force mathematics. `mdp/vectorized_parameterized_force_action.py` is the Isaac Lab `ActionTerm` adapter and the only new module that writes Actor forces to PhysX.

`coverage/batched_accumulator.py` owns per-environment masks, frame deduplication, area sums, and row resets. `coverage/batched_visibility.py` owns GPU candidate, first-hit, incident-face, normal, and distance gates. `runtime/task009d0_coverage_runtime.py` connects the camera and environment-local coordinates to those pure coverage modules.

`mdp/task009d0_terms.py` exposes Actor RGB, previous action, raw new-coverage reward, and a separately named privileged diagnostic group. `task009d0_vector_env.py` owns the synchronous reset and 1200-step lifecycle. `robotarm_magnetic_task009d0_env_cfg.py` composes the new task without modifying the old one.

The four live scripts are independent: `validate_task009d0_single_env_parity.py`, `validate_task009d0_two_env_isolation.py`, `benchmark_task009d0_throughput.py`, and `validate_task009d0_long_soak.py`. `summarize_task009d0_throughput.py` is offline and is the only program allowed to select `num_envs`.

---

### Task 1: Freeze Configuration, Runtime Evidence, and Split-Safe Pose Sampling

**Files:**
- Create: `configs/task009d0/vectorized_training_candidates_v1.json`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task009d0_config.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task009d0_pose_batch.py`
- Create: `scripts/stomach_coverage/inspect_task009d0_prerequisites.py`
- Create: `tests/stomach_coverage/test_task009d0_config.py`
- Create: `tests/stomach_coverage/test_task009d0_pose_batch.py`

**Interfaces:**
- Consumes: `configs/task009b/pose_library_manifest_v1.json`, `configs/task009b/coverage_manifest_v1.json`, `configs/task009b/unreachable_region_v1.json`, `read_jsonl()`, `file_sha256()`, and `stable_record_is_valid()`.
- Produces: `load_task009d0_config(path: Path, *, frozen: bool = False) -> dict`, `PoseBatch`, `Task009D0PoseBatchSampler.sample(env_ids: np.ndarray, episode_indices: np.ndarray) -> PoseBatch`, and `resolve_explicit(env_ids: np.ndarray, pose_ids: Sequence[str]) -> PoseBatch`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_candidate_config_freezes_non_destructive_contract():
    cfg = load_task009d0_config(CONFIG_PATH)
    assert cfg["num_env_candidates"] == [1, 2, 4, 8]
    assert cfg["episode"]["formal_steps"] == 1200
    assert cfg["episode"]["hold_steps"] == 10
    assert cfg["camera"] == {"width": 1280, "height": 720, "hz": 10, "fov_deg": 120.0}
    assert cfg["coverage"]["max_distance_m"] == 0.07
    assert cfg["benchmark"]["warmup_steps"] == 50
    assert cfg["benchmark"]["measured_steps"] == 300
    assert cfg["benchmark"]["repeats"] == 3
    assert cfg["benchmark"]["minimum_free_memory_fraction"] == 0.20
    assert cfg["benchmark"]["near_tie_fraction"] == 0.10
```

- [ ] **Step 2: Write failing split and reproducibility tests**

```python
def test_training_sampler_is_reproducible_and_rejects_split_leakage(pose_fixture):
    first = Task009D0PoseBatchSampler(pose_fixture, authorized_split="train", training_seed=990009)
    second = Task009D0PoseBatchSampler(pose_fixture, authorized_split="train", training_seed=990009)
    ids = np.asarray([0, 1, 7], dtype=np.int64)
    episodes = np.asarray([3, 3, 3], dtype=np.int64)
    assert first.sample(ids, episodes).pose_ids == second.sample(ids, episodes).pose_ids
    assert len(set(first.sample(ids, episodes).rng_seeds.tolist())) == len(ids)
    with pytest.raises(ValueError, match="training sampler cannot access"):
        first.resolve_explicit(np.asarray([0]), ["validation-0006"])

def test_validation_loader_accepts_only_explicit_validation_ids(pose_fixture):
    loader = Task009D0PoseBatchSampler(pose_fixture, authorized_split="validation", training_seed=990009)
    batch = loader.resolve_explicit(np.asarray([0]), ["validation-0006"])
    assert batch.pose_ids == ("validation-0006",)
    with pytest.raises(ValueError, match="explicit pose split mismatch"):
        loader.resolve_explicit(np.asarray([0]), ["test-0001"])
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009d0_config.py \
  tests/stomach_coverage/test_task009d0_pose_batch.py
```

Expected: collection fails because the new modules and JSON configuration do not exist.

- [ ] **Step 4: Add the exact candidate configuration**

Write schema `robotarm_magnetic_lab.task009d0_vectorized_training`, version 1, task ID, exact code baseline, `[1,2,4,8]`, 4.0 m environment spacing, clocks, camera contract, force ranges, pose manifest path and hashes, unreachable-region path and hash, coverage manifest path and hash, 1200 formal steps, ten HOLD steps, threshold times `[0.80,0.90,0.95]`, 300-second audit duration, and the benchmark constants asserted above. Reject unknown top-level keys to prevent silent misspellings.

- [ ] **Step 5: Implement deterministic pose batches**

```python
from collections.abc import Sequence

@dataclass(frozen=True)
class PoseBatch:
    env_ids: np.ndarray
    pose_ids: Sequence[str]
    poses_world_xyzw: np.ndarray
    episode_indices: np.ndarray
    rng_seeds: np.ndarray

def derived_env_episode_seed(training_seed: int, env_id: int, episode_index: int) -> int:
    sequence = np.random.SeedSequence([int(training_seed), int(env_id), int(episode_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
```

Load all 1200 external records once, validate their file hash and manifest hash, partition by the record's actual `split`, and resolve only the constructor's authorized split. `sample()` is enabled only for the training split, requires one episode index per environment row, and derives each RNG seed from training seed, environment ID, and that row's episode index. Validation and test use `resolve_explicit()` with versioned pose IDs. Never accept caller coordinates.

- [ ] **Step 6: Implement the prerequisite inspector**

The inspector must record Python, Isaac Lab, Isaac Sim, torch, RSL-RL, Warp, driver, GPU name, total/free VRAM, PhysX device, camera device, raycast device, and `inspect.signature(Camera._update_buffers_impl)`. It writes one JSON file and returns nonzero if the required GPU/RTX/PhysX path is unavailable. It must not install or upgrade packages.

- [ ] **Step 7: Run tests and the read-only inspector**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009d0_config.py \
  tests/stomach_coverage/test_task009d0_pose_batch.py

./run_isaaclab.sh -p scripts/stomach_coverage/inspect_task009d0_prerequisites.py \
  --device cuda:0 \
  --output /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/prerequisites
```

Expected: tests pass; inspector status is `pass` and reports GPU PhysX, RTX camera, and Warp on `cuda:0`.

- [ ] **Step 8: Commit Task 1**

```bash
git add configs/task009d0 source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime \
  scripts/stomach_coverage/inspect_task009d0_prerequisites.py \
  tests/stomach_coverage/test_task009d0_config.py \
  tests/stomach_coverage/test_task009d0_pose_batch.py
git commit -m "feat: freeze task009d0 runtime and pose contract"
```

### Task 2: Add Simulator-Independent Batched Force Mathematics

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/vectorized_parameterized_force.py`
- Create: `tests/parameterized_force/test_vectorized_parameterized_force.py`

**Interfaces:**
- Consumes: `ParameterizedForceConfig`, `ParameterizedForceMode`, `GRAVITY_M_S2`, and scalar `parameterized_endpoint_forces()`.
- Produces: `BatchedEndpointForceCommand`, `batched_parameterized_endpoint_forces()`, and `batched_equivalent_com_wrench()`.

- [ ] **Step 1: Write scalar-equivalence and invalid-row tests**

```python
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_batched_modes_match_scalar_controller(alpha):
    modes = torch.arange(6, dtype=torch.int64)
    masses = torch.full((6,), 0.005735, dtype=torch.float64)
    axes = torch.tensor([[1.0, 0.0, 0.0]] * 6, dtype=torch.float64)
    batch = batched_parameterized_endpoint_forces(modes, torch.full((6,), alpha), masses, axes)
    for row, mode in enumerate(ParameterizedForceMode):
        scalar = parameterized_endpoint_forces(mode, alpha, mass_kg=float(masses[row]), camera_axis_world=axes[row].numpy())
        np.testing.assert_allclose(batch.camera_forces_world[row].numpy(), scalar.camera_force_world)
        np.testing.assert_allclose(batch.other_forces_world[row].numpy(), scalar.other_force_world)

def test_vertical_active_axis_reports_exact_environment_rows():
    with pytest.raises(ValueError, match=r"undefined lateral direction.*\[1\]"):
        batched_parameterized_endpoint_forces(
            torch.tensor([0, 1]), torch.tensor([0.0, 0.5]),
            torch.tensor([0.005735, 0.005735]),
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        )
```

- [ ] **Step 2: Run the test and verify missing-module failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force/test_vectorized_parameterized_force.py
```

Expected: FAIL because `vectorized_parameterized_force` does not exist.

- [ ] **Step 3: Implement batched forces without CPU copies**

```python
@dataclass(frozen=True)
class BatchedEndpointForceCommand:
    modes: torch.Tensor
    alpha: torch.Tensor
    force_ratios: torch.Tensor
    camera_forces_world: torch.Tensor
    other_forces_world: torch.Tensor
    directions_world: torch.Tensor

def batched_parameterized_endpoint_forces(
    modes: torch.Tensor,
    alpha: torch.Tensor,
    masses_kg: torch.Tensor,
    camera_axes_world: torch.Tensor,
    config: ParameterizedForceConfig = ParameterizedForceConfig(),
) -> BatchedEndpointForceCommand:
    modes = modes.to(dtype=torch.int64).reshape(-1)
    alpha = alpha.to(dtype=masses_kg.dtype).reshape(-1)
    masses = masses_kg.reshape(-1)
    axes = camera_axes_world.reshape(-1, 3)
    if not (len(modes) == len(alpha) == len(masses) == len(axes)):
        raise ValueError("batched parameterized-force rows must match")
    if torch.any((modes < 0) | (modes > 5)):
        raise ValueError("mode IDs must be in [0, 5]")
    if torch.any(~torch.isfinite(alpha) | (alpha < 0) | (alpha > 1)):
        raise ValueError("alpha must be finite and in [0, 1]")
    if torch.any(~torch.isfinite(masses) | (masses <= 0)):
        raise ValueError("masses must be finite and positive")
    axis_norm = torch.linalg.vector_norm(axes, dim=1)
    if torch.any(~torch.isfinite(axes)) or torch.any(axis_norm <= 1e-12):
        raise ValueError("camera axes must be finite and non-zero")
    axes = axes / axis_norm[:, None]
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=axes.dtype, device=axes.device)
    lateral = torch.linalg.cross(world_up.expand_as(axes), axes)
    lateral_norm = torch.linalg.vector_norm(lateral, dim=1)
    lateral_modes = (modes == 1) | (modes == 2) | (modes == 3) | (modes == 4)
    bad_rows = torch.flatnonzero(lateral_modes & (lateral_norm <= 1e-12))
    if bad_rows.numel():
        raise ValueError(f"undefined lateral direction at environment rows {bad_rows.tolist()}")
    lateral = lateral / lateral_norm.clamp_min(1e-12)[:, None]
    lateral = torch.where(((modes == 2) | (modes == 4))[:, None], -lateral, lateral)
    ratios = torch.zeros_like(alpha)
    ratios = torch.where((modes == 1) | (modes == 2), config.move_min_ratio + alpha * (config.move_max_ratio - config.move_min_ratio), ratios)
    ratios = torch.where((modes == 3) | (modes == 4), config.view_min_ratio + alpha * (config.view_max_ratio - config.view_min_ratio), ratios)
    ratios = torch.where(modes == 5, config.up_min_ratio + alpha * (config.up_max_ratio - config.up_min_ratio), ratios)
    target = ratios * masses * GRAVITY_M_S2
    directions = torch.where((modes == 5)[:, None], world_up.expand_as(axes), lateral)
    directions = torch.where((modes == 0)[:, None], torch.zeros_like(directions), directions)
    camera_scale = torch.where((modes == 1) | (modes == 2), 0.5 * target, target)
    camera = camera_scale[:, None] * directions
    other = torch.where(((modes == 1) | (modes == 2))[:, None], 0.5 * target[:, None] * directions, torch.zeros_like(camera))
    return BatchedEndpointForceCommand(modes, alpha, ratios, camera, other, directions)
```

Use tensor masks for HOLD, MOVE, VIEW, UP, positive, and negative modes. Validate the whole batch before producing output. Keep calculations on the input device and preserve float64 in pure equivalence tests.

- [ ] **Step 4: Implement equivalent COM wrench math**

`batched_equivalent_com_wrench()` takes `[E,3]` camera/other forces and `[E,3]` camera/other/com positions and returns `[E,3]` resultant force plus `[E,3]` torque. Test that MOVE, VIEW, and UP match the scalar cross-product definition; the ActionTerm still submits UP as an actual point force.

- [ ] **Step 5: Run new and legacy force tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force/test_vectorized_parameterized_force.py \
  tests/parameterized_force/test_baseline_audit.py
```

Expected: all tests pass and the legacy controller source is unchanged.

- [ ] **Step 6: Commit Task 2**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/vectorized_parameterized_force.py \
  tests/parameterized_force/test_vectorized_parameterized_force.py
git commit -m "feat: add batched parameterized force math"
```

### Task 3: Add the Vectorized Isaac Lab ActionTerm

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/vectorized_parameterized_force_action.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.pyi`
- Create: `tests/parameterized_force/test_vectorized_parameterized_force_action.py`

**Interfaces:**
- Consumes: Task 2 batched force functions, `camera_sphere_centers_local()`, and `set_forces_and_torques_index()`.
- Produces: `VectorizedParameterizedForceAction`, `VectorizedParameterizedForceActionTermCfg`, and property `previous_action_features: torch.Tensor` with shape `[E,7]`.

- [ ] **Step 1: Write failing ActionTerm contract tests**

```python
def test_previous_action_features_mask_hold_strength(fake_vector_action):
    fake_vector_action.process_actions(torch.tensor([[0.0, 0.8], [5.0, 0.25]]))
    expected = torch.tensor([[1,0,0,0,0,0,0.0], [0,0,0,0,0,1,0.25]], dtype=torch.float32)
    torch.testing.assert_close(fake_vector_action.previous_action_features.cpu(), expected)

def test_action_term_requires_exact_vector_shape(fake_vector_action):
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        fake_vector_action.process_actions(torch.zeros((1, 2)))
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force/test_vectorized_parameterized_force_action.py
```

Expected: FAIL because the ActionTerm is absent.

- [ ] **Step 3: Implement vector state and action parsing**

Allocate `_raw_actions` and `_processed_actions` as `[env.num_envs,2]`, mode IDs as int64, alpha as float32, mass as `[E]`, and local endpoint centers once. `process_actions()` validates shape and values, canonicalizes HOLD alpha to zero, clears residual composer rows before the new cycle, and updates the seven-dimensional history tensor. `reset(env_ids)` clears only the selected raw, processed, history, and composer rows; `reset(None)` clears all rows.

- [ ] **Step 4: Implement vectorized 240 Hz application**

Read root link pose and COM tensors for all environments without `.cpu()` or NumPy conversion. Rotate local endpoints with batched quaternion matrices. Submit MOVE/VIEW equivalent COM wrench rows with one indexed call and submit UP point-force rows with one indexed call using camera endpoint positions. HOLD rows remain cleared. Do not create Python telemetry objects per physics substep; retain device tensors for optional audit snapshots.

- [ ] **Step 5: Test indexed composer behavior with a fake composer**

The fake must prove mixed `[HOLD, MOVE, VIEW, UP]` batches write only the intended environment rows, UP receives positions, MOVE/VIEW receive torque, and a mode change to HOLD clears the previous force before the next substep.

- [ ] **Step 6: Run force and ActionTerm suites**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force/test_vectorized_parameterized_force.py \
  tests/parameterized_force/test_vectorized_parameterized_force_action.py \
  tests/parameterized_force/test_baseline_audit.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp \
  tests/parameterized_force/test_vectorized_parameterized_force_action.py
git commit -m "feat: add vectorized parameterized force action"
```

### Task 4: Add Per-Environment Area-Weighted Coverage State

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/batched_accumulator.py`
- Create: `tests/coverage/test_batched_accumulator.py`

**Interfaces:**
- Consumes: float64 vertex weights and boolean `[E,V]` visibility masks.
- Produces: `BatchedCoverageUpdate` and `BatchedCoverageAccumulator.update(frame_ids, visible_mask)`.

- [ ] **Step 1: Write failing monotonicity, deduplication, and row-reset tests**

```python
def test_duplicate_frame_is_ignored_per_row_and_reset_is_isolated():
    acc = BatchedCoverageAccumulator(torch.tensor([1.0, 2.0, 3.0]), num_envs=2, device="cpu")
    first = acc.update(torch.tensor([10, 10]), torch.tensor([[1,0,0], [0,1,0]], dtype=torch.bool))
    second = acc.update(torch.tensor([10, 11]), torch.tensor([[0,0,1], [0,0,1]], dtype=torch.bool))
    assert second.updated.tolist() == [False, True]
    before = acc.mask[1].clone()
    acc.reset_rows(torch.tensor([0]))
    assert not acc.mask[0].any()
    torch.testing.assert_close(acc.mask[1], before)
```

- [ ] **Step 2: Run and verify missing-module failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/coverage/test_batched_accumulator.py
```

Expected: FAIL because the new accumulator does not exist.

- [ ] **Step 3: Implement the tensor accumulator**

```python
@dataclass(frozen=True)
class BatchedCoverageUpdate:
    updated: torch.Tensor
    visible_count: torch.Tensor
    newly_covered_count: torch.Tensor
    visible_area_m2: torch.Tensor
    newly_covered_area_m2: torch.Tensor
    cumulative_area_m2: torch.Tensor
    coverage_fraction: torch.Tensor

class BatchedCoverageAccumulator:
    def __init__(self, weights: torch.Tensor, num_envs: int, device: str) -> None:
        self._weights = weights.to(device=device, dtype=torch.float64).reshape(-1)
        if self._weights.numel() == 0 or torch.any(self._weights < 0) or self._weights.sum() <= 0:
            raise ValueError("coverage weights must be nonnegative with positive total area")
        self._mask = torch.zeros((int(num_envs), len(self._weights)), dtype=torch.bool, device=device)
        self._last_frame = torch.full((int(num_envs),), -1, dtype=torch.int64, device=device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask.clone()

    def update(self, frame_ids: torch.Tensor, visible_mask: torch.Tensor) -> BatchedCoverageUpdate:
        frame_ids = frame_ids.to(device=self._mask.device, dtype=torch.int64).reshape(-1)
        visible = visible_mask.to(device=self._mask.device, dtype=torch.bool)
        if visible.shape != self._mask.shape or frame_ids.shape != self._last_frame.shape:
            raise ValueError("coverage frame and visibility shapes must match accumulator state")
        if torch.any(frame_ids < self._last_frame):
            raise RuntimeError("coverage frame IDs decreased")
        updated = frame_ids > self._last_frame
        effective_visible = visible & updated[:, None]
        previous = self._mask.clone()
        self._mask |= effective_visible
        newly = self._mask & ~previous
        self._last_frame = torch.where(updated, frame_ids, self._last_frame)
        visible_count = effective_visible.sum(dim=1)
        newly_count = newly.sum(dim=1)
        visible_area = (effective_visible.to(torch.float64) * self._weights).sum(dim=1)
        newly_area = (newly.to(torch.float64) * self._weights).sum(dim=1)
        cumulative_area = (self._mask.to(torch.float64) * self._weights).sum(dim=1)
        total_area = self._weights.sum()
        return BatchedCoverageUpdate(updated, visible_count, newly_count, visible_area, newly_area, cumulative_area, cumulative_area / total_area)

    def reset_rows(self, env_ids: torch.Tensor) -> None:
        rows = env_ids.to(device=self._mask.device, dtype=torch.int64).reshape(-1)
        self._mask[rows] = False
        self._last_frame[rows] = -1
```

Store masks as bool, weights and area sums as float64, and last frame IDs as int64 initialized to `-1`. Reject decreasing frame IDs and decreasing cumulative area. Duplicate frames return zero new area only for duplicated rows.

- [ ] **Step 4: Compare every row with the legacy NumPy accumulator**

Add a deterministic sequence of five visibility masks for three environments, run one legacy `CoverageAccumulator` per row, and assert exact mask equality plus float64 area equality within `1e-15` for the fixture.

- [ ] **Step 5: Run coverage tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/coverage/test_batched_accumulator.py \
  tests/coverage/test_coverage_accumulator.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/batched_accumulator.py \
  tests/coverage/test_batched_accumulator.py
git commit -m "feat: isolate batched coverage state"
```

### Task 5: Add Exact Batched GPU Visibility

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/batched_visibility.py`
- Create: `tests/coverage/test_batched_visibility.py`

**Interfaces:**
- Consumes: `ReferenceMesh`, environment-local camera centers/axes, and the frozen visibility constants.
- Produces: `build_incident_face_table()`, `batched_candidate_mask()`, `visible_from_batched_first_hits()`, and `BatchedWarpFirstHitRaycaster.query()` returning device tensors `[E,V]`.

- [ ] **Step 1: Write pure geometry equivalence tests**

```python
def test_batched_candidates_equal_scalar_for_each_camera():
    vertices = torch.tensor(FIXTURE_VERTICES, dtype=torch.float64)
    centers = torch.tensor([[0,0,0], [0.01,0,0]], dtype=torch.float64)
    axes = torch.tensor([[0,0,1], [0,0,1]], dtype=torch.float64)
    mask, distances = batched_candidate_mask(vertices, centers, axes)
    for row in range(2):
        scalar_ids, scalar_distances = candidate_vertices(FIXTURE_VERTICES, centers[row].numpy(), axes[row].numpy())
        assert torch.flatnonzero(mask[row]).tolist() == scalar_ids.tolist()
        np.testing.assert_allclose(distances[row, scalar_ids].numpy(), scalar_distances, atol=1e-12)
```

- [ ] **Step 2: Write incident-face, distance, and normal gate tests**

Use the existing two-triangle fixture and assert the batched boolean output exactly matches `visible_from_first_hits()` plus `camera_facing_first_hits(normal_sign=-1)` for every row. Include the inclusive 70 mm and 60-degree half-angle boundary cases.

- [ ] **Step 3: Run and verify tests fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/coverage/test_batched_visibility.py
```

Expected: FAIL because the module does not exist.

- [ ] **Step 4: Implement exact device-side gates**

Build a padded int64 incident-face table `[V,K]` with `-1` padding. Use float64 for candidate distances and cone comparisons to mirror the legacy scalar code. Use float32 only for Warp ray starts/directions and mesh inputs, matching the existing production raycaster. Compute visibility as the conjunction of candidate, valid first hit, incident face membership, `abs(hit_distance-target_distance) <= 1e-4 + epsilon`, and lumen-facing normal with sign `-1`.

- [ ] **Step 5: Implement the batched Warp adapter**

Expand local target vertices to `[E,V,3]`, build safe directions for zero-distance non-candidates, call `isaaclab.utils.warp.ops.raycast_mesh` once per control boundary, and return torch distances and face IDs without `.cpu()`. Mask non-candidates after the query. Record ray count as `E*V` and candidate counts separately.

- [ ] **Step 6: Run pure coverage suites**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/coverage/test_batched_visibility.py \
  tests/coverage/test_visibility_geometry.py \
  tests/coverage/test_batched_accumulator.py
```

Expected: all non-simulator tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/batched_visibility.py \
  tests/coverage/test_batched_visibility.py
git commit -m "feat: add exact batched GPU visibility"
```

### Task 6: Add RGB Synchronization and the Vector Coverage Runtime

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task009d0_coverage_runtime.py`
- Create: `tests/stomach_coverage/test_task009d0_coverage_runtime.py`

**Interfaces:**
- Consumes: Tasks 4 and 5, `reference_from_stage()`, camera tensors, `env.scene.env_origins`, and the frozen unreachable mask.
- Produces: `Task009D0RgbSynchronizer`, `Task009D0CoverageRuntime.capture_initial()`, `update_boundary()`, `reset_rows()`, and `latest_update`.

- [ ] **Step 1: Write frame-vector synchronization tests with a fake camera**

```python
def test_all_stale_frames_are_forced_once_and_partial_staleness_fails(fake_camera):
    sync = Task009D0RgbSynchronizer(num_envs=2)
    fake_camera.frame = torch.tensor([4, 4])
    assert sync.observe(boundary=1, camera=fake_camera).tolist() == [4, 4]
    fake_camera.frame = torch.tensor([4, 4])
    assert sync.observe(boundary=2, camera=fake_camera).tolist() == [5, 5]
    fake_camera.frame = torch.tensor([6, 5])
    with pytest.raises(RuntimeError, match="partial camera advancement"):
        sync.observe(boundary=3, camera=fake_camera)
```

- [ ] **Step 2: Write environment-local translation and C0 tests**

Create two fake environments with origins `[0,0,0]` and `[4,0,0]`, identical local camera poses, and identical visibility. Assert identical masks and coverage. Assert `capture_initial()` sets positive `C0`, returns zero new reward, and the first formal `update_boundary()` returns only post-action new area.

- [ ] **Step 3: Run and verify missing-module failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009d0_coverage_runtime.py
```

Expected: FAIL because the runtime does not exist.

- [ ] **Step 4: Implement shared local geometry and isolated state**

Read the accepted mesh once from `env_0`, subtract `env.scene.env_origins[0]` to create shared local vertices, and subtract each environment origin from camera positions before visibility. Keep camera quaternion/axis in the common world orientation because all cloned stomachs share rotation. Own separate raw and reachable batched accumulators.

- [ ] **Step 5: Implement boundary idempotence**

Cache the latest global control boundary and update object. Reward and observation terms may request the same boundary repeatedly; the runtime must return the cached result without another camera capture or coverage union. During reset stabilization, `stabilizing=True` suppresses accumulation. After ten HOLD boundaries, clear masks and call `capture_initial()` exactly once on the last RGB frame.

- [ ] **Step 6: Run runtime and legacy coverage tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009d0_coverage_runtime.py \
  tests/coverage/test_batched_visibility.py \
  tests/coverage/test_batched_accumulator.py \
  tests/coverage/test_coverage_accumulator.py \
  tests/coverage/test_visibility_geometry.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task009d0_coverage_runtime.py \
  tests/stomach_coverage/test_task009d0_coverage_runtime.py
git commit -m "feat: add vector coverage runtime and rgb sync"
```

### Task 7: Register the Non-Destructive Synchronous Vector Environment

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/task009d0_terms.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.pyi`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_task009d0_env_cfg.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/task009d0_vector_env.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Create: `tests/stomach_coverage/test_task009d0_environment_contract.py`
- Modify: `tests/stomach_coverage/test_environment_contract.py`

**Interfaces:**
- Consumes: Tasks 1, 3, and 6.
- Produces: Gym task `Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0`, Actor observation keys `policy.rgb` and `policy.previous_action`, separate `privileged` keys, raw reward `new_coverage`, and synchronous auto-reset.

- [ ] **Step 1: Write static non-destructive registration tests**

```python
def test_new_task_is_separate_and_old_task_registration_is_unchanged():
    source = REGISTRY.read_text(encoding="utf-8")
    assert 'id="Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"' in source
    assert 'id="Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0"' in source
    assert "Task009BTrainingEnv" in source
    assert "Task009D0VectorEnv" in source

def test_task009d0_cfg_preserves_physics_camera_and_assets():
    cfg = RobotarmMagneticTask009D0EnvCfg()
    assert cfg.sim.dt == 1 / 240
    assert cfg.decimation == 24
    assert cfg.scene.capsule_camera.width == 1280
    assert cfg.scene.capsule_camera.height == 720
    assert cfg.scene.capsule_camera.update_period == 0.1
    assert cfg.scene.stomach.spawn.usd_path == STOMACH_ASSET_USD_PATH

def test_training_mode_rejects_explicit_validation_pose_ids():
    cfg = RobotarmMagneticTask009D0EnvCfg()
    assert cfg.pose_split == "train"
    assert cfg.explicit_pose_ids is None
```

- [ ] **Step 2: Write observation leakage and lifecycle tests**

```python
def test_actor_group_has_only_rgb_and_previous_action():
    cfg = RobotarmMagneticTask009D0EnvCfg()
    assert set(vars(cfg.observations.policy)) >= {"rgb", "previous_action"}
    source = inspect.getsource(type(cfg.observations.policy))
    for forbidden in ("pose", "velocity", "contact", "coverage", "pose_id", "split"):
        assert forbidden not in source.lower()

def test_synchronous_constants_are_exact():
    assert FORMAL_STEPS == 1200
    assert RESET_HOLD_CYCLES == 10
```

- [ ] **Step 3: Run and verify tests fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009d0_environment_contract.py \
  tests/stomach_coverage/test_environment_contract.py
```

Expected: FAIL because the new environment files and registration are absent.

- [ ] **Step 4: Implement task terms**

`task009d0_rgb(env)` calls the runtime synchronizer and returns the existing circular RGB processing result. `task009d0_previous_action(env)` reads the vector ActionTerm's seven-dimensional history. `task009d0_new_coverage(env)` calls the idempotent runtime update and returns float32 `newly_covered_area_m2 / reachable_total_area_m2`. Privileged terms are placed only under `observations.privileged`; do not concatenate them into policy.

- [ ] **Step 5: Implement the new configuration**

Inherit the accepted parameterized-force stomach scene, replace only the action config with `VectorizedParameterizedForceActionTermCfg`, set `scene.num_envs` from the validated candidate configuration, retain 4.0 m spacing, set `episode_length_s=120.0`, and omit the base timeout termination because the subclass returns synchronized truncation. The default `pose_split` is `train`; validation/test launchers must explicitly set their matching split and pass versioned pose IDs, never coordinates. Do not attach an RSL-RL configuration in TASK-009D-0.

- [ ] **Step 6: Implement explicit and automatic synchronous reset**

```python
class Task009D0VectorEnv(ManagerBasedRLEnv):
    def reset(self, seed=None, env_ids=None, options=None):
        if env_ids is not None and len(env_ids) != self.num_envs:
            raise ValueError("formal task009d0 reset must include all environments")
        return self._reset_all_with_hold(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, extras = super().step(action)
        self._formal_step += 1
        if torch.any(terminated) or torch.any(truncated):
            raise RuntimeError("base environment terminated before synchronous horizon")
        if self._formal_step == FORMAL_STEPS:
            terminal = self._terminal_snapshot(obs)
            next_obs, reset_extras = self._reset_all_with_hold(seed=None, options=None)
            extras.update(reset_extras)
            extras["terminal_observation"] = terminal
            truncated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            obs = next_obs
        return obs, reward, terminated, truncated, extras
```

`_reset_all_with_hold()` calls the base reset, samples one train pose per environment, resets the action/composer, writes `[E,7]` poses and zero `[E,6]` velocities, forwards PhysX, verifies row-wise pose error, sets `stabilizing=True`, executes ten HOLD calls through `super().step`, clears episode length, formal step, previous action, and coverage rows, captures one positive `C0`, and returns the last HOLD RGB. Any failure raises and aborts the batch.

The environment owns `_episode_indices` as one int64 value per row. Supplying a new explicit training seed resets every episode index to zero. Each successful synchronous reset samples with the current row indices and then increments every row once. A test-only row-reset helper may alter one row for isolation evidence, but the formal training path only resets all rows together.

- [ ] **Step 7: Run the full pure regression set**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force \
  tests/coverage \
  tests/stomach_coverage \
  tests/runtime
```

Expected: all tests pass; old task contract tests remain unchanged except for an assertion that the new task is additive.

- [ ] **Step 8: Commit Task 7**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab \
  tests/stomach_coverage/test_task009d0_environment_contract.py \
  tests/stomach_coverage/test_environment_contract.py
git commit -m "feat: register synchronous vector coverage task"
```

### Task 8: Gate 2 Single-Environment Exact Parity

**Files:**
- Create: `scripts/stomach_coverage/validate_task009d0_single_env_parity.py`
- Create: `tests/stomach_coverage/test_task009d0_single_env_parity_protocol.py`

**Interfaces:**
- Consumes: new task with `num_envs=1`, the five TASK-009C validation poses, scalar force functions, scalar visibility functions, and batched runtime snapshots.
- Produces: a parity manifest with exact current/cumulative mask equality and bounded float64 area error.

- [ ] **Step 1: Write protocol rejection tests**

Build fixture records that intentionally change one visible bit, one frame ID, one force component, and one physics substep count. `validate_parity_records()` must reject each mutation and must not apply tolerances to boolean masks.

- [ ] **Step 2: Run and verify tests fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009d0_single_env_parity_protocol.py
```

Expected: FAIL because the validator does not exist.

- [ ] **Step 3: Implement the live parity validator**

Launch the new task in explicit validation mode. For each frozen validation pose `0006`, `0011`, `0015`, `0017`, and `0019`, resolve the versioned pose ID, reset the new task, and execute a fixed 60-boundary sequence that covers every mode at alpha 0.0, 0.5, and 1.0. On every boundary, compare the vector force tensor with the scalar controller, run scalar candidate/first-hit/normal coverage on the same camera state, and compare scalar versus batched current and cumulative masks exactly. Record frame IDs, 24 substeps, mask hashes, area error, pose, and finite-state flags.

- [ ] **Step 4: Run Gate 2**

Run:

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_single_env_parity.py \
  --headless --device cuda:0 \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate2_single_parity
```

Expected: five poses and 300 formal boundaries pass; every current/cumulative mask comparison is exact, every step has 24 substeps and one new RGB, and `C0` is positive.

- [ ] **Step 5: Run pure regressions again**

Run the protocol test plus `tests/parameterized_force`, `tests/coverage`, and `tests/stomach_coverage/test_environment_contract.py`. Expected: all pass.

- [ ] **Step 6: Commit Task 8**

```bash
git add scripts/stomach_coverage/validate_task009d0_single_env_parity.py \
  tests/stomach_coverage/test_task009d0_single_env_parity_protocol.py
git commit -m "test: prove task009d0 single environment parity"
```

### Task 9: Gates 3 and 4 Multi-Environment Isolation and Reset Sync

**Files:**
- Create: `scripts/stomach_coverage/validate_task009d0_two_env_isolation.py`
- Create: `scripts/stomach_coverage/validate_task009d0_gpu_reset_sync.py`
- Create: `tests/stomach_coverage/test_task009d0_isolation_protocol.py`

**Interfaces:**
- Consumes: Gate 2 accepted task and runtime.
- Produces: two-environment isolation evidence and GPU reset/frame evidence.

- [ ] **Step 1: Write isolation protocol tests**

The offline validator must reject any change in the untouched environment's coverage-mask hash, local pose, frame state, episode-index/RNG bookkeeping, previous action, or reward after another row is reset. It must also reject a pair of equal-action environments whose local trajectories differ beyond `1e-6 m` or quaternion absolute alignment below `1-1e-6`.

- [ ] **Step 2: Implement the two-phase live isolation run**

Phase A resets both environments to the same pose and executes an identical 100-boundary R3 sequence, comparing environment-local pose, velocity, RGB frame delta, force, visible mask, coverage mask, and reward each boundary. Phase B repeats from the same seeds; environment 0 receives a divergent sequence while environment 1 repeats Phase A. Environment 1 must reproduce its Phase A trace. Finally reset only row 0 of coverage, frame association, previous action, and episode-index bookkeeping without stepping physics and prove row 1 hashes are unchanged.

- [ ] **Step 3: Implement the GPU reset-sync run**

Use two distinct train poses. Verify pose writes precede HOLD, velocities and composer rows are zeroed, ten frame vectors increment exactly once, both final RGB tensors are finite, both `C0` values are strictly positive, episode length is zero, and devices are `cuda:0`. Repeat 20 synchronized resets to detect residual state.

- [ ] **Step 4: Run Gates 3 and 4**

Run:

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_two_env_isolation.py \
  --headless --device cuda:0 \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate3_isolation

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_gpu_reset_sync.py \
  --headless --device cuda:0 --resets 20 \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate4_reset_sync
```

Expected: both scripts report `pass` with zero retries, zero partial camera advancement, and zero cross-environment changes.

- [ ] **Step 5: Run protocol and environment tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009d0_isolation_protocol.py \
  tests/stomach_coverage/test_task009d0_environment_contract.py \
  tests/stomach_coverage/test_task009d0_coverage_runtime.py
```

Expected: all pass.

- [ ] **Step 6: Commit Task 9**

```bash
git add scripts/stomach_coverage/validate_task009d0_two_env_isolation.py \
  scripts/stomach_coverage/validate_task009d0_gpu_reset_sync.py \
  tests/stomach_coverage/test_task009d0_isolation_protocol.py
git commit -m "test: prove task009d0 environment isolation"
```

### Task 10: Gate 5 Throughput Benchmark and Frozen Parallel Count

**Files:**
- Create: `scripts/stomach_coverage/benchmark_task009d0_throughput.py`
- Create: `scripts/stomach_coverage/summarize_task009d0_throughput.py`
- Create: `tests/stomach_coverage/test_task009d0_throughput_summary.py`
- Create after successful Gate 5: `configs/task009d0/vectorized_training_frozen_v1.json`

**Interfaces:**
- Consumes: candidate configuration and one isolated Isaac Sim process per candidate/repetition.
- Produces: 12 benchmark manifests, a verified aggregate summary, and one selected `num_envs`.

- [ ] **Step 1: Write deterministic selection-rule tests**

```python
def test_near_tie_selects_smaller_candidate():
    rows = [row(4, throughput=39.0, free=0.30), row(8, throughput=42.0, free=0.25)]
    assert select_num_envs(rows, near_tie_fraction=0.10, minimum_free=0.20) == 4

def test_fast_candidate_with_insufficient_memory_is_rejected():
    rows = [row(4, throughput=30.0, free=0.25), row(8, throughput=50.0, free=0.19)]
    assert select_num_envs(rows, near_tie_fraction=0.10, minimum_free=0.20) == 4
```

- [ ] **Step 2: Implement one-process benchmark recording**

The benchmark accepts exactly one `--num_envs` and one `--repeat_index`. It runs 50 warm-up and 300 measured formal boundaries with deterministic R3 actions, records wall time for physics, RGB sync, coverage, and total boundary, aggregate environment transitions per second, CUDA free/total memory at every measured boundary, maximum process memory if available, candidate/ray counts, forced captures, and faults. It writes a manifest and SHA-256 inventory atomically.

- [ ] **Step 3: Implement strict offline aggregation**

Require exactly three valid manifests for each of 1, 2, 4, and 8 environments. Reject mismatched commits, configs, devices, clocks, boundary counts, or hashes. Compute median aggregate throughput and minimum free-memory fraction per candidate. Reject faulted or memory-ineligible candidates, apply the 10% near-tie rule, and write the frozen JSON with selected integer, source manifest hashes, and selection explanation.

- [ ] **Step 4: Run summary unit tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009d0_throughput_summary.py
```

Expected: all selection and malformed-manifest tests pass.

- [ ] **Step 5: Run the 12 isolated benchmark processes**

Run each pair independently so Isaac Sim and RTX state are recreated:

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/benchmark_task009d0_throughput.py \
  --headless --device cuda:0 --num_envs 1 --repeat_index 0 \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_throughput
```

Repeat with `repeat_index` 0, 1, 2 for each `num_envs` 1, 2, 4, 8. Do not place a shell loop in the evidence report; record every resolved command.

- [ ] **Step 6: Select and freeze the parallel count**

Run:

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/summarize_task009d0_throughput.py \
  --artifact_root /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_throughput \
  --write_frozen_config configs/task009d0/vectorized_training_frozen_v1.json
```

Expected: exactly one candidate is selected by the frozen rule; all evidence hashes verify; no camera, physics, coverage, or memory fault is present.

- [ ] **Step 7: Commit Task 10**

```bash
git add scripts/stomach_coverage/benchmark_task009d0_throughput.py \
  scripts/stomach_coverage/summarize_task009d0_throughput.py \
  tests/stomach_coverage/test_task009d0_throughput_summary.py \
  configs/task009d0/vectorized_training_frozen_v1.json
git commit -m "perf: freeze task009d0 parallel environment count"
```

### Task 11: Gate 6 Two-Episode Long Soak

**Files:**
- Create: `scripts/stomach_coverage/validate_task009d0_long_soak.py`
- Create: `tests/stomach_coverage/test_task009d0_long_soak_protocol.py`

**Interfaces:**
- Consumes: Gate 5 frozen configuration.
- Produces: two complete vector episodes with exact clocks, reset evidence, and no inherited coverage.

- [ ] **Step 1: Write strict episode-count tests**

```python
def test_two_episode_manifest_requires_exact_counts(valid_manifest):
    assert validate_soak(valid_manifest)["status"] == "pass"
    broken = copy.deepcopy(valid_manifest)
    broken["episodes"][1]["envs"][0]["coverage_points"] = 1200
    with pytest.raises(ValueError, match="1201 coverage points"):
        validate_soak(broken)
```

Also reject formal substeps other than 28800, HOLD substeps other than 240, nonzero episode length after reset, inherited final mask hash, non-monotonic coverage, repeated RGB, early done, or missing terminal observation.

- [ ] **Step 2: Implement the long-soak runner**

Load only `vectorized_training_frozen_v1.json`; disallow command-line `num_envs`. Run two consecutive 120-second synchronous episodes with deterministic per-environment R3 seeds. Record `C0`, all 1200 coverage values, nAUC, threshold times, frame vectors, formal/hold substeps, mask hashes before and after reset, action proportions, finite-state flags, and per-environment fault status. Do not write 720p RGB frames unless a fault occurs.

- [ ] **Step 3: Run Gate 6**

Run:

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_long_soak.py \
  --headless --device cuda:0 \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate6_long_soak
```

Expected: two episodes pass for every selected environment, each has 1201 coverage points and 28800 formal physics substeps, each inter-episode stabilization has 240 substeps, and the second `C0` is computed from a cleared mask.

- [ ] **Step 4: Run the final automated suite**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force \
  tests/coverage \
  tests/stomach_coverage \
  tests/runtime
```

Expected: all tests pass with no legacy regression.

- [ ] **Step 5: Commit Task 11**

```bash
git add scripts/stomach_coverage/validate_task009d0_long_soak.py \
  tests/stomach_coverage/test_task009d0_long_soak_protocol.py
git commit -m "test: validate task009d0 long soak"
```

### Task 12: Documentation, Evidence Audit, and Final Handoff

**Files:**
- Create: `docs/TASK009D0_VECTORIZED_TRAINING_INFRASTRUCTURE.md`
- Modify: `docs/PROJECT_RUN_LOG.md`
- Create: `handoffs/reports/TASK-009D0-vectorized-training-infrastructure-report.md`
- Modify: `handoffs/active/README.md`

**Interfaces:**
- Consumes: all Gate 1 through Gate 6 manifests and committed frozen configuration.
- Produces: reproducible user documentation and the authoritative Linux execution report.

- [ ] **Step 1: Write operational documentation**

Document the new task ID, exact non-destructive relationship to the old task, installation command, prerequisite inspection, unit tests, Gates 2 through 6 commands, frozen `num_envs`, Actor/privileged observation boundary, synchronous reset semantics, external artifact root, and known private-camera compatibility risk. Do not claim PPO or learning is implemented.

- [ ] **Step 2: Audit every evidence file**

Recompute SHA-256 and byte size for prerequisite JSON, five-pose parity manifest, isolation manifest, reset-sync manifest, all 12 throughput manifests, throughput summary, frozen config, soak manifest, and artifact inventory. Verify each manifest records the same feature branch and the commit that produced it.

- [ ] **Step 3: Write the Gate-ordered report**

The report begins with `complete`, `partial`, or `needs_input`, then records code baseline, Windows planning branch and HEAD, Linux feature branch and HEAD, actual runtime versions, modified files, every command, observed test count, Gate 1 through Gate 6 result, selected `num_envs` with raw throughput/memory table, deviations, unverified claims, and all external artifact paths, sizes, and hashes.

- [ ] **Step 4: Verify documentation does not overclaim**

Run:

```bash
rg -n "PPO|GRU|CNN|VLM|learning|训练完成|优于" \
  docs/TASK009D0_VECTORIZED_TRAINING_INFRASTRUCTURE.md \
  handoffs/reports/TASK-009D0-vectorized-training-infrastructure-report.md
```

Every match must explicitly say the component is not implemented or remains TASK-009D-1 scope. Also run `git diff --check` and the final automated test suite from Task 11.

- [ ] **Step 5: Commit documentation and report**

```bash
git add docs/TASK009D0_VECTORIZED_TRAINING_INFRASTRUCTURE.md \
  docs/PROJECT_RUN_LOG.md \
  handoffs/reports/TASK-009D0-vectorized-training-infrastructure-report.md \
  handoffs/active/README.md
git commit -m "docs: complete task009d0 vector infrastructure"
```

- [ ] **Step 6: Push and return exact delivery evidence**

```bash
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor 7c4c5a18780b980ad3882ce75f1d64733fc3080d HEAD
git push -u origin feature/TASK-009D0-vectorized-training-infrastructure
git ls-remote --heads origin feature/TASK-009D0-vectorized-training-infrastructure
```

Expected: worktree is clean, ancestry check succeeds, local and remote feature HEADs are identical, and the report contains the same full HEAD.

---

## Gate Stop and Return Rules

Gate 1 failure stops all simulator work. Gate 2 parity failure stops all multi-environment work. Gate 3 isolation failure stops reset and throughput work. Gate 4 reset/RGB failure stops benchmarking. Gate 5 may reject an individual candidate because of OOM or insufficient memory, but it returns `partial` if no candidate passes. Gate 6 failure returns `partial` and does not authorize TASK-009D-1.

Missing external pose-library data with a valid Git manifest returns `needs_input`. Package-version mismatch, missing GPU PhysX, missing RTX camera, or incompatible private camera API returns `partial` with exact observed evidence because it is an execution-environment incompatibility, not missing user data.

No failed Gate may be repaired by reducing image resolution, changing camera intrinsics, changing 70 mm visibility, changing ROI, deleting a pose, changing force ranges, replacing a seed, accepting approximate masks, or editing USD assets. Any required asset restructuring returns `needs_decision` to Windows before an asset file is touched.
