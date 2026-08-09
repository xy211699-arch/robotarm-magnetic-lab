# TASK-001 Linux implementation report

- Status: **complete**
- Planning/base commit: `c6aaa07141b8bf8c1a2281a5a0e6796413b4fcfa`
- Implementation head before this report commit: `db397ac744071c79bf9070e9a740df453421e430`
- Branch: `feature/TASK-001-p0-coverage-teleop`
- Task: `Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0`
- Date: 2026-08-09

The report commit necessarily follows the implementation head above. The pushed branch tip is the
authoritative delivery commit. Nothing was merged.

## Delivered

- Preflight selected exactly the rendered luminal surface. Its invisible collision duplicate has
  identical geometry. The denominator has 24,529 welded vertices and 49,047 triangles; geometry
  SHA-256 is `67b4e06a4f5cfc3b8d51e5411942226d4bcabd3a6a937a456057e408a990ad36`.
- Added CUDA-batched first-hit visibility: 50 mm range, circular 120-degree full FOV,
  incident-triangle membership, 0.1 mm hit tolerance, one update per unique 1 Hz frame.
- Added cumulative/resettable state, append-only records, deterministic 2D exports, isolated
  USD-context 3D point cloud, and exact mask/record/color/export consistency checks.
- Registered a one-environment scalar-action stomach task. Deployable observations remain joint
  position, joint velocity, and external-magnet pose; evaluator truth is isolated.
- Added the frozen keyboard protocol and common non-queuing session controller for interactive and
  scripted inputs. All eleven action IDs and key bindings are unchanged.
- Added launcher, validator, tests, and operator documentation. Clocks remain 240 Hz physics,
  20 Hz atomic control, 1 Hz recorded RGB/coverage, and up to 30 Hz engineering display.

## Validation

| Check | Command | Result |
|---|---|---|
| New suites | `./run_isaaclab.sh -p -m pytest tests/coverage tests/action_layer/test_atomic_stomach_teleop_cfg.py tests/action_layer/test_atomic_keyboard_protocol.py -q` | `26 passed` |
| Original stage-one pure | `./run_isaaclab.sh -p -m pytest tests/action_layer/test_atomic_protocol.py tests/action_layer/test_executor.py tests/action_layer/test_safety.py -q` | `10 passed in 0.07s` |
| Expanded pure runner | `./run_isaaclab.sh -p scripts/action_layer/validate_pure.py` | `total=14 failed=0` |
| GPU/scalar geometry | `./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all` | PASS; distances and face IDs agreed |
| Stomach integration | `./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py --num_envs 1 --coverage_samples 5` | 11/11 `DONE`; 12 unique frames; PASS |
| Existing table acceptance | `./run_isaaclab.sh -p scripts/action_layer/validate_atomic_table.py --num_envs 1 --max_steps_per_action 60 --viz kit` | 11 terminal, 11 `DONE`, exit 0 |
| Legacy 9D smoke | `./run_isaaclab.sh -p scripts/zero_agent.py --task Template-Robotarm-Magnetic-Table-Lab-v0 --num_envs 1 --max_steps 5 --viz kit` | shape `(1,9)`; 5 steps; exit 0 |
| Rendered smoke | `./run_isaaclab.sh -p scripts/action_layer/teleop_atomic_stomach_coverage.py --task Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0 --num_envs 1 --viz kit --max_idle_updates 2` | main and isolated coverage view created; evidence finalized; exit 0 |
| Performance | `./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py --num_envs 1 --coverage_samples 100` | 11/11 `DONE`; 100 unique frames; PASS |
| Hygiene | `git diff --check` | no errors |

100-frame results: median `0.0033557525002834154 s`, p95 `0.0069043277499986275 s`,
maximum `0.013250904999949853 s`; candidate/ray median `712`, maximum `717`; all
`100/100` updates completed below one second. Final coverage was `0.06200823515023034`
(informational only). Timing ends after CUDA hit tensors are copied to CPU, so GPU completion is
synchronized and included. Final mask, frame record, green point count, and `6.201%` 2D text agreed.

## Acceptance correction

The first 100-frame attempt stopped after 25 frames because the harness repeatedly submitted
`HOLD`, capturing small drive sag as a new target each second. This created artificial cumulative
drift. The corrected validator submits no extra action: after the eleven actions it keeps the final
executor target latched and advances only physics and unique recorded frames. This changes neither
the executor nor action semantics. The diagnostic evidence is retained.

## Post-delivery live-view correction

Manual review found that the original coverage points shared the simulation stage with the stomach
surface. They were coplanar with the rendered wall and could be hidden by depth testing even though
the mask and exported images were correct. Commit `1423fcd` moves only the engineering point cloud,
capsule marker, trajectory, and camera into a separate USD context, increases point diameter, and
adds a live `Coverage: percent (covered / total)` HUD. Each 1 Hz update also prints
`P0_COVERAGE frame=... covered=... percent=... new=...`.

The correction does not alter the mesh denominator, visibility rays, cumulative mask, physics,
actions, safety, or deployable observations. Targeted tests passed `24/24`; GPU stomach integration
again passed all eleven actions with 12 unique coverage frames, ending at `5.569%`; and the corrected
GUI startup/exit smoke returned zero without the previous viewport-menu cleanup exception.

## Post-delivery robot/ASM-to-stomach collision correction

Manual review found that the eleven actions did run the hard-safety monitor, but that monitor only
checked joint/device limits, workspace, XRDF ASM-to-arm self-clearance, and an optional flat ground.
The legacy `collision` termination was likewise ASM-to-arm only. The stomach triangle mesh was not
part of either predictive or runtime safety, so a robot/ASM path could cross the stomach wall.

The stomach atomic task now enables a mesh-specific checker without changing any action ID,
increment, trajectory meaning, capsule dynamics, or deployable observation. It transforms the dense
XRDF `world_collision` spheres (including the mounted ASM) into world coordinates and queries
unsigned distance to the exact active stomach collision mesh. Every action receives a 21-sample
swept-path precheck, and the current configuration is rechecked at each 20 Hz control update. A
5 mm buffer triggers `ENVIRONMENT_COLLISION` and holds the last safe target. Unsigned distance is
required because the current stomach is a thin open surface; it protects both sides without relying
on an invalid inside/outside sign. The capsule is deliberately excluded, so normal capsule-wall
contact remains valid.

Validation passed `16/16` pure action-layer tests. The live initial robot/ASM-to-wall clearance was
`53.741 mm`, and independent reset testing completed all `11/11` actions with `DONE`. The dedicated
guard test applied repeated `APPROACH`: live clearance fell through `42.956`, `32.045`, `21.091`,
and `10.662 mm`; request five was blocked as `ENVIRONMENT_COLLISION` at approximately `4.639 mm`.
After the hold settled, measured clearance remained positive at `4.274 mm`, so no geometric overlap
occurred. `scripts/action_layer/validate_stomach_collision_guard.py` reported `PASS`.

## Deviations and notes

- An isolated worktree `/tmp/robotarm-task001` preserved the original checkout. The launcher
  prepends that worktree source because the editable installation otherwise resolves the original.
- This Isaac Lab 3.0 build reserves but does not expose `--headless`; the validator translates the
  required public flag to `HEADLESS=1` before `AppLauncher` parsing.
- `pytest==8.3.5` was installed in Isaac Sim's user Python environment; no repository dependency or
  simulator source was changed.
- The first isolated-view smoke exposed Kit 110.1's asynchronous viewport teardown ordering. The
  corrected view retains hidden Hydra resources until `SimulationApp` shutdown; the repeated smoke
  exited cleanly. Existing UJITSO/cooking, Fabric VtValue, CPU powersave warnings remain non-blocking.
- Human subjective inspection of a long keyboard session remains for Windows/user review. P0 has
  no required final coverage threshold and makes no complete-coverage claim.

## External evidence

All evidence is outside Git. Every session manifest enumerates every sibling artifact (relative
path, byte size, SHA-256). The following rows identify the complete preflight and the four complete
session roots through those canonical manifests; the final-run primary artifacts are also listed.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_preflight/20260809_120549/prerequisites.json` | 6062 | `e38d3d981cc3d90b062182fb7e93334e2b438be44655e1604fbf20ee991e8c51` |
| `.../20260809_043038_781790Z/artifact_inventory.json` | 1319 | `47eefb0e08680ac7a3b5f54795daa721d690a7ae9663fa2cd5b0248fceb125fc` |
| `.../20260809_043203_523202Z/artifact_inventory.json` | 1312 | `197fd00cab10677c3a4438a399fb9349b4934fca159fa4baa99e609730abbe77` |
| `.../20260809_043515_115207Z/artifact_inventory.json` | 1320 | `328e5c2cf3ff97527a2d4c04d94285729650426b3aabde9229edbabb0ac24e7e` |
| `.../20260809_043803_867325Z/artifact_inventory.json` | 1321 | `d29948a9750c2cecc57f14cbfd10b178cd6d10dc28eeac50f07066b96e9ecdb6` |
| `.../20260809_043803_867325Z/actions.jsonl` | 13639 | `400a43cddbabef0db449ff8e914a7a1023f6cc9089434f73d75da5b4420d19bf` |
| `.../20260809_043803_867325Z/coverage_mask.npy` | 24657 | `59a69fcd5621e85f12aeccfd34f497554ee833d80d374ff59ed959d350ed473c` |
| `.../20260809_043803_867325Z/coverage_timings_s.npy` | 928 | `85b4812723d3e5e5415e929a415320deb78d6e66ba97b4e63061be266e98c59a` |
| `.../20260809_043803_867325Z/frames.jsonl` | 53158 | `e50c00811a8cd9b33eacf7f09727929a272078a9219fc4bd61fb6bdb8d759535` |
| `.../20260809_043803_867325Z/metadata.json` | 1372 | `a221231e60dad37ebc88398eb87452b480476cfe41316aedde508fef3b04c189` |
| `.../20260809_043803_867325Z/snapshot_0001_exit.json` | 532 | `b64cc3850989b1a59df288f58e6e452379548205f75adabd2552209cbb1aa4a7` |
| `.../20260809_043803_867325Z/snapshot_0001_exit.png` | 93000 | `dac75a3ef79b241de68cceacbefc5d0d31396b0f295b9ea79dd4c81f8e13916b` |
| `.../20260809_043803_867325Z/summary.json` | 392 | `ed33b43f449f74e3d99152f511061b5676963b9a9ed9c4bc91216d5733f2d7f4` |
| `.../20260809_043803_867325Z/trajectory_world_m.npy` | 2528 | `57b6814a0e82421ebf30abf8ff5185dd6e0136be0ba683ef403e68e758421309` |

The abbreviated `...` prefix in this table is
`/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop`.

## Delivery

Push this feature branch to writable `origin`; do not merge. Windows reviews the diff and evidence
and authorizes subsequent work.
