# TASK-003 Linux Execution Report

## Disposition

- Status: `partial`
- Branch: `feature/TASK-003-dynamic-capsule-force-teleop`
- Contract branch: `workflow/TASK-003-dynamic-capsule-force-teleop`
- Planning head: `027c558aebb142151d64d36769d3a62a7a9ffd80`
- Contract lineage base: `0e3ae452c403b141b20ca1e4700c3d37dc7f2b90`
- Validated implementation head before this report: `bc9d5681abaa88b1af1ce1b3b6937620c4f785a6`
- Task: `Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0`

The isolated task implements the requested real-dynamics interface. Dynamic motion,
gravity, stomach contact, exact six-direction COM force, zero commanded torque,
continuous rendering, and absence of runtime pose correction were observed. The
result is `partial`, not `complete`, because the frozen delivered physics produced
sustained measured penetration and one small per-step continuity-bound exceedance.
No physics parameter was tuned to change that result.

## Implemented scope

- Added pure force normalization/scaling and a level-triggered held-key adapter.
- Added a dedicated Isaac Lab action term that reads live mass and applies only a
  world-frame force at the capsule center of mass on all 240 Hz physics substeps.
- Added an isolated one-environment task with only the three-dimensional
  `dynamic_force` action, RGB observation, reset event, and timeout termination.
- Added a Kit launcher with W/S, A/D, Q/E, clear/reset/snapshot/exit controls,
  60 Hz environment loop, 30 Hz capsule-camera view, and external JSON evidence.
- Added a strict live preflight, three-second zero-input settling test, six signed
  force phases, read-only surface-clearance measurement, focused tests, and docs.

## Live preflight

The final preflight returned `DYNAMIC_FORCE_PREFLIGHT_PASS`.

| Property | Live value |
|---|---|
| Capsule rigid body | non-kinematic; gravity enabled |
| Collider | USD `Capsule`, local axis `Z` |
| Radius / cylinder / total length | `0.0065 / 0.0120000001 / 0.0250000001 m` |
| Mass | `0.0057349997 kg` |
| Inertia diagonal | `[3.1337407e-7, 3.9471431e-7, 1.9233103e-7] kg m²` |
| Authored max linear velocity | `0.25 m/s` |
| CCD | scene enabled; capsule body enabled |
| Rates | 240 Hz physics, 60 Hz environment/render interval, 30 Hz camera |
| Stomach collider | static, collision enabled |
| Collision mesh | 24,529 vertices; 49,047 triangles |
| Collision topology | 73,581 edges; 21 boundary; 0 nonmanifold |
| Collision geometry SHA-256 | `d0096ee5fc3dfaeaba218ca8cae9eace290203632b9f4dfb026480c9e43c5977` |
| Contact sensor | present |
| Runtime state writers | none found |
| COM wrench semantics | installed API states `positions=None` acts at body COM |

The installed Isaac Lab physics manager disables scene CCD under GPU Dynamics.
The task therefore forces one-environment CPU PhysX while RTX rendering remains on
the GPU. This is a task-local API compatibility decision; no delivered physical
material, friction, damping, restitution, depenetration, or velocity value changed.

## Deterministic validation

Command:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/validate_dynamic_force_stomach.py \
  --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 \
  --seed 42 --force_weight_ratio 0.5 --headless
```

Observed result: `DYNAMIC_FORCE_VALIDATION_FAIL`. This maps to task disposition
`partial` under the contract because the force interface works but frozen collision
behavior shows sustained penetration/continuity instability.

Zero-input settling ran for 3.0 simulated seconds. Gravity remained enabled,
stomach contact was observed, states remained finite, displacement was
`0.001031 m`, maximum linear speed was `0.012391 m/s`, and maximum angular speed
was `0.938258 rad/s`.

All six active phases applied the expected signed `0.5mg` force. Magnitude was
`0.0281301737 N`; maximum error from the double-precision expectation was
`3.31e-10 N`; commanded torque was exactly zero.

| Direction | Net displacement XYZ (m) | Contact-force norm range (N) | Clearance range (m) | Max 240 Hz step (m) |
|---|---|---|---|---:|
| `+X` | `[0.006576, 0.009531, -0.000059]` | `[0.045291, 0.078441]` | `[-0.001517, -0.000892]` | `0.0005570` |
| `-X` | `[-0.000168, -0.002474, 0.000097]` | `[0.037451, 0.085869]` | `[-0.001651, -0.000194]` | `0.0007836` |
| `+Y` | `[0.007061, 0.009757, -0.000078]` | `[0.045291, 0.083532]` | `[-0.001982, -0.000914]` | `0.0006213` |
| `-Y` | `[-0.001263, -0.013498, 0.001647]` | `[0.039352, 0.123703]` | `[-0.001945, -0.000830]` | `0.0010656` |
| `+Z` | `[0.001070, -0.000835, -0.000103]` | `[0.027415, 0.056819]` | `[-0.001871, -0.001281]` | `0.0000597` |
| `-Z` | `[0.001067, -0.000824, -0.000089]` | `[0.045291, 0.085322]` | `[-0.001847, -0.001281]` | `0.0000597` |

There were no nonfinite samples and no measured boundary escape. Minimum measured
surface clearance was `-0.00198198 m`. Six sustained-clearance-decrease windows
were found in `-X/-Y` phases. The maximum substep displacement was
`0.00106555 m`, exceeding the authored bound
`0.25*(1/240)+1e-5 = 0.00105167 m` by `1.39e-5 m`.

## Rendered acceptance

The required scripted Kit run exited cleanly after exactly 120 environment samples
and 2.0 simulated seconds. It showed changing capsule pose, changing contact force,
force release to zero, a 240/60/60/30 simulated-Hz configuration, and no
state-setter recovery. Measured wall-clock environment throughput was 9.90 Hz; this
is performance telemetry, not simulated cadence. No human performed the separate
long-duration subjective smoothness/visible-snap review, so that claim remains unverified.

## Test and regression results

- TASK-003 focused tests: `47 passed`.
- TASK-002 and delivered pure regressions: `87 passed`.
- Coverage geometry: `COVERAGE_GEOMETRY_PASS`.
- P0 stomach: `actions=11 done=11 status=PASS`.
- Eleven-action table: `actions=11 terminal=11 done=11`.
- Legacy 9D table: action shape `(1,9)`, five steps completed.
- Compile, `git diff --check`, and worktree hygiene: passed.

Exact commands used for these results:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/inspect_dynamic_force_prerequisites.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 --num_envs 1 --headless
./run_isaaclab.sh -p scripts/dynamic_force/validate_dynamic_force_stomach.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 --seed 42 --force_weight_ratio 0.5 --headless
./run_isaaclab.sh -p scripts/dynamic_force/teleop_dynamic_force_stomach.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 --force_weight_ratio 0.5 --scripted_sequence "+x:0.5,zero:0.25,-x:0.5,zero:0.25" --max_steps 120 --viz kit
./run_isaaclab.sh -p -m pytest tests/dynamic_force -q --disable-warnings
./run_isaaclab.sh -p -m pytest tests/ideal_surface tests/coverage tests/action_layer/test_atomic_protocol.py tests/action_layer/test_executor.py tests/action_layer/test_safety.py tests/action_layer/test_atomic_stomach_teleop_cfg.py tests/action_layer/test_atomic_keyboard_protocol.py -q --disable-warnings
./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py --num_envs 1 --coverage_samples 5
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_table.py --num_envs 1 --max_steps_per_action 60 --viz kit
./run_isaaclab.sh -p scripts/zero_agent.py --task Template-Robotarm-Magnetic-Table-Lab-v0 --num_envs 1 --max_steps 5 --viz kit
./run_isaaclab.sh -p -m compileall -q scripts/dynamic_force source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop
git diff --check
```

## External evidence

Sizes are bytes and hashes are SHA-256. Files remain outside Git.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/dynamic_force_preflight/20260816_032301_373270Z/prerequisites.json` | 4370 | `d6d16739015c8919e9760eeaca480947fafdc1d53043172b894821b35b8624d2` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/dynamic_force_validation/20260816_031556_762173Z/summary.json` | 13830 | `8c6f03bc2cfcf8df80b50b6c603065c28412f6aee124a8201672d49fe1282ec7` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/dynamic_force_validation/20260816_031556_762173Z/samples.jsonl` | 638135 | `d3972492f8111b5baae6a0db16e88a7702aba65dde785c226a5acf77afc9b331` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/dynamic_force_teleop/20260816_031837_089300Z/session.json` | 599 | `7511df8db6a56f4d5df3b2ec2586a54d92eb5ede2f8f3f1808d9f375e9570b65` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/dynamic_force_teleop/20260816_031837_089300Z/samples.jsonl` | 78082 | `2137dac7450d400eba01fc7f2c5f64a2f9d779336ea11398756140c0708c3bf4` |

## Deviations and unverified claims

- The optional Superpowers skill named by the plan was unavailable. The frozen
  repository contract and execution plan were read completely and executed directly.
- CPU PhysX is used because this installed Isaac Lab version disables scene CCD
  under GPU Dynamics. This does not change physical parameters.
- Clearance is a read-only five-point centerline measurement against 512 nearest
  triangle-centroid candidates; it never writes or corrects capsule state.
- The validation is intentionally failing on sustained penetration and one small
  continuity excess; the report does not relabel those observations as passing.
- Long-duration subjective keyboard smoothness is unverified.
- No claim is made for magnetic control, tissue deformation, fluid effects,
  autonomous navigation, RL/VLM integration, or hardware-calibrated realism.

## Review boundary

Linux did not merge this branch. Windows should review the implementation and
evidence before authorizing a later collision/material calibration task or merge.
