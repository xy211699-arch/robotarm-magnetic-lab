# TASK-002 Linux Execution Report

## Disposition

- Status: `complete`
- Branch: `feature/TASK-002-ideal-surface-controller`
- Base/planning commit: `98598b3ac82fd43aa4b5af01a0ab7790086fdd09`
- Validated implementation head before this report: `302567ae29192d7ac61f49f675e2485bf4693e0b`
- Delivery head: the report commit at the pushed branch tip (reported to Windows after push)
- Task: `Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0`
- Action schema: `ideal_surface_v1`, scalar IDs `0..14`

TASK-002 was implemented in an isolated worktree and branch created from the exact
planning commit. The existing eleven-action magnetic executor, magnetic physics,
camera calibration, assets, coverage semantics, safety thresholds, tasks, and
prior evidence were not changed.

## Implemented scope

- Added a pure NumPy ideal-surface package with the frozen fifteen-action contract,
  minimal state-dependent mask, spherocylinder support geometry, mesh adjacency and
  exact closest-point projection, active surface anchors, contact classification,
  local frames, quintic trajectories, and the `READY`/`EXECUTING`/`TERMINAL_FAULT`
  state machine.
- Added an Isaac Lab action term that writes a smooth target at every 240 Hz physics
  substep and emits exactly one result at each 1.0 s Actor boundary.
- Added the isolated one-environment stomach task. Actor observations contain only
  the existing 1 Hz capsule RGB. It has no magnetic-physics action term and no
  magnetic collision bridge.
- Added the fifteen-action keyboard adapter, Kit coverage launcher, read-only
  prerequisite inspector, pure geometry acceptance script, deterministic stomach
  validator, focused tests, and user documentation.
- Reused the approved P0 CUDA first-hit coverage evaluator without changing its
  mesh, ray, visibility, accumulation, or information-isolation semantics.

## Mandatory preflight result

The preflight gate returned `pass`.

| Item | Verified value |
|---|---|
| Capsule collider | one unambiguous USD `Capsule`, local axis `+Z` |
| Radius | `0.0065 m` |
| Cylindrical half-length | `0.006000000052154064 m` |
| Tip-to-tip length | `0.025000000104308126 m` |
| Camera convention | ROS: local `+Z` optical, local `-Y` image-up |
| Camera axes in capsule frame | optical `[0, 0, -1]`, image-up `[0, 1, 0]` |
| Approved luminal mesh | 24,529 vertices, 49,047 triangles |
| Mesh geometry SHA-256 | `85ddd3e79438509364245c87be9a9564d1bf9ca29afb2c922fc013b2f7561d09` |
| Topology | 73,581 edges, 21 boundary edges, 0 nonmanifold edges |
| Inward convention | winding sign `-1`, confirmed at initial contact |
| Initial support gap | `-0.00011030260127639201 m`, valid triangle `19696` |
| Root-pose API | `RigidObject.write_root_pose_to_sim`, quaternion `wxyz` |
| Root-velocity API | `RigidObject.write_root_velocity_to_sim` |

## Validation commands and observed results

1. Mandatory preflight:

   ```bash
   ./run_isaaclab.sh -p scripts/ideal_surface/inspect_ideal_surface_prerequisites.py \
     --task Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0 \
     --output logs/ideal_surface_preflight
   ```

   Result: `IDEAL_SURFACE_PREFLIGHT status=pass`.

2. Focused pure tests:

   ```bash
   ./run_isaaclab.sh -p -m pytest tests/ideal_surface -q --disable-warnings
   ```

   Result: `46 passed`.

3. Pure geometry/action acceptance:

   ```bash
   ./run_isaaclab.sh -p scripts/ideal_surface/validate_ideal_surface_geometry.py
   ```

   Result: `IDEAL_SURFACE_GEOMETRY_PASS`; all nine checks passed, including eight
   unique start directions, absolute 15° start tilt, 15° tilt/rise/precession,
   4.0 mm roll arc, upright residual range, and contact/boundary-limited `DONE`.

4. Rendered launcher startup/exit smoke:

   ```bash
   ./run_isaaclab.sh -p scripts/ideal_surface/teleop_ideal_surface_stomach_coverage.py \
     --task Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0 \
     --num_envs 1 --seed 42 --viz kit
   ```

   Result: task, Kit view, controller, and coverage visualization initialized and
   exited cleanly; final snapshot and artifact inventory were written. This smoke
   intentionally closed before a manual action, so its coverage is 0%.

5. Deterministic long stomach acceptance:

   ```bash
   ./run_isaaclab.sh -p scripts/ideal_surface/validate_ideal_surface_stomach.py \
     --task Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0 \
     --seed 42 --random_actions 1000 \
     --output logs/ideal_surface_validation
   ```

   Result: `PASS`. The run issued 75 directed acceptance actions plus 1,000 valid
   seeded random actions. All 1,075 requests produced 1,075 unique results; all
   fifteen IDs were observed; all eight start directions ended at 15°; no nonfinite
   pose, nonadjacent jump, unexplained hard failure, or hard penetration occurred.
   There were 636 unique 1 Hz coverage updates and final informational coverage was
   `0.12144808186228546`. Coverage time was median `0.0034425725 s`, p95
   `0.0076054872 s`, maximum `0.0108820020 s`; every update was below 1 s.

6. Delivered coverage and stage-one pure regressions:

   ```bash
   ./run_isaaclab.sh -p -m pytest \
     tests/coverage \
     tests/action_layer/test_atomic_protocol.py \
     tests/action_layer/test_executor.py \
     tests/action_layer/test_safety.py \
     tests/action_layer/test_atomic_stomach_teleop_cfg.py \
     tests/action_layer/test_atomic_keyboard_protocol.py \
     -q --disable-warnings
   ```

   Result: `41 passed`.

7. Delivered GPU/scalar coverage geometry:

   ```bash
   ./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all
   ```

   Result: `COVERAGE_GEOMETRY_PASS`; GPU and scalar hit distance both 0.02 m and
   face ID both 0 for the hit case.

8. Delivered P0 stomach integration:

   ```bash
   ./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py \
     --num_envs 1 --coverage_samples 5
   ```

   Result: `P0_VALIDATION actions=11 done=11 ... status=PASS`; initial robot/ASM
   stomach clearance was 0.053741 m and no regression collision occurred.

9. Delivered eleven-action table acceptance:

   ```bash
   ./run_isaaclab.sh -p scripts/action_layer/validate_atomic_table.py \
     --num_envs 1 --max_steps_per_action 60 --viz kit
   ```

   Result: `ATOMIC_VALIDATION actions=11 terminal=11 done=11`.

10. Legacy 9D table smoke:

    ```bash
    ./run_isaaclab.sh -p scripts/zero_agent.py \
      --task Template-Robotarm-Magnetic-Table-Lab-v0 \
      --num_envs 1 --max_steps 5 --viz kit
    ```

    Result: action space remained `(1, 9)` and the five-step smoke completed.

11. Final hygiene:

    ```bash
    ./run_isaaclab.sh -p -m compileall -q \
      source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface \
      source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/ideal_surface_action.py \
      source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/ideal_surface_keyboard.py \
      scripts/ideal_surface
    git diff --check
    ```

    Result: both completed with exit code 0.

## External evidence artifacts

These files remain outside Git. Sizes are bytes; hashes are SHA-256.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `logs/ideal_surface_preflight/20260815_104000_842563Z/prerequisites.json` | 4759 | `67441752f8a7fd2ed7293c4674004ed789673da4e433ec43dd041dc255714165` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/actions.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/artifact_inventory.json` | 1312 | `d127505f0cee80fdc3fc297a9b92e795e96028960e2e2ef12bbd9d40471a7824` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/coverage_mask.npy` | 24657 | `97bb0a0719f4abbb87c0a4fa8cc58dcc087aef8c02eda7d1b6279e77f6044f90` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/coverage_timings_s.npy` | 128 | `fdee2f2368bf2af9c942f32cce9d982e48dfc46889bf923e99bc9ac834a4ba46` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/frames.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/metadata.json` | 1385 | `bae8a66da26a96e48c8fcd9e288a4ea4bbc6c7d3da4b4b1424cc8876383292de` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/snapshot_0001_exit.json` | 498 | `c8b60cb854188b4bd66cfd26e3386591b4f62f3f5c37cdc08445b2131216797d` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/snapshot_0001_exit.png` | 86000 | `922a7d3f42e2350ac685240605f0403f752a781a80a6a161759e83c97114f61f` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/summary.json` | 325 | `c2d91dad09378c0aaa02fee86cf1bd25a1b4ac9cd6e4efd1ec9258ed8d2ddfb7` |
| `logs/ideal_surface_coverage_teleop/20260815_110305_176489Z/trajectory_world_m.npy` | 128 | `4aa7aa40d1bbd6bba4570a87b12a7a2be0c4643337cc363349524c7c66ef8fd0` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/actions.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/artifact_inventory.json` | 1647 | `7c2c8602a23fc26009ece84b901e2e9a7c77702072a04d40d9e9b10c25b54a16` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/coverage_mask.npy` | 24657 | `00fa8b3943abd28a8f7e07da8953f11bcfa82e732a02d81b6bf22d3b503d6cfa` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/coverage_timings_s.npy` | 5216 | `08673a5d01396abdec64c09ea01936115287888d2ffff5aae72860c68a901548` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/frames.jsonl` | 337679 | `973366a7e644d2f99958acf32b35bc02ae0c30d4241eb06861e495676977eace` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/ideal_surface_records.jsonl` | 577971 | `95231cb12e48fb5db2f0d85ce4857d3c186e4c3cedad5150dd136d751daa724c` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/ideal_surface_validation.json` | 666 | `5844f23340c3b32a9d8349866dc8db4fefba5e65e1776288e69693c8abf74b08` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/metadata.json` | 1385 | `ea2d6882060bfc6130b671f33be26d2fe7909914037a47e8836260e9915fde5d` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/snapshot_0001_validation.json` | 527 | `a730349200cd243c5b0d44b422364e9bc6602a0fbc8ef4e95f2943613d7510c9` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/snapshot_0001_validation.png` | 94197 | `8867e7ceec806c5278275f571790d9a549d1fe9b9ebdbc41108c743d7d82c0aa` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/summary.json` | 399 | `3c8f0af229a580ea3baaf261a079c7a183e6264f9798d33f0e5b017f2a5bd251` |
| `logs/ideal_surface_validation/20260815_131706_218645Z/trajectory_world_m.npy` | 15392 | `85872d02c9eb97ffdfb9d02db34540249408f65cd048bb0db627f65035cfc9a0` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/actions.jsonl` | 13370 | `78dc14d6ac7e3a9f5b8cfd5faf538d2b3349bef26324abe2338356843acf2243` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/artifact_inventory.json` | 1319 | `e14078b41076931d2c833de6fbcb4f05063aabb27a1940fbb6547f8f6c760ceb` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/coverage_mask.npy` | 24657 | `b384697caaaeff6ab4a69554b7ed7da73dcd2e0ea6fad0726c47f0b7fa02b578` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/coverage_timings_s.npy` | 216 | `ced276f0d10bda51839f72b9baffb4b869d8b3b41a3729f008049f3c8b7aea5a` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/frames.jsonl` | 5799 | `71990420b3a8ecc4349116721fcff5538d76190d843549b5909a58458333ecc4` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/metadata.json` | 1378 | `5ae2882c6131f58f921c2b8a2e5894523de78a005d887436dca55380402d1af4` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/snapshot_0001_exit.json` | 519 | `4921e88eeea525e06a747cdd3c05a8615c276b894018fdc3ff5b8b1cb9c4baaa` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/snapshot_0001_exit.png` | 90622 | `ad938afd13e6e99daf2eb5a116a21b9bb154000b0d200e3a32e91153738389ef` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/summary.json` | 390 | `ac9d897905c9326b50f70ceecba62c38febd18db28707cef7d11d16c05175c80` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/p0_coverage_teleop/20260815_141230_920269Z/trajectory_world_m.npy` | 392 | `f3fed93b7cad2545d5745f9affb7dd38c95e83e5915d03ffed726ee45574b3c3` |
| `logs/action_layer/stage1_atomic_results.jsonl` | 11329 | `d22afe1d3d1dda05c2a3684709c53c6597ed3e6dd78817f35998d436dd39e370` |

## Deviations and unverified claims

- No functional deviation from the approved design or task contract is known.
  Exact vectorized closest-point queries were added to keep the 1,000-action run
  practical; scalar/vectorized equivalence is covered by focused tests.
- The optional Superpowers execution skills named by the plan were not installed in
  this environment. The approved plan was executed directly with task-scoped TDD and
  independent commits; this did not change scope or acceptance criteria.
- The rendered launcher smoke proves startup, rendering, snapshot, and clean exit,
  but no long subjective human keyboard session was performed by Linux automation.
- Final coverage is informational and was not optimized. No claim is made that the
  ideal kinematic trajectory is magnetically realizable or transfers to hardware.
- TASK-002 does not validate magnetic tracking, tissue deformation, stochastic slip,
  autonomous coverage planning, reinforcement learning, VLM integration, rewards,
  or physical calibration.

## Review boundary

Windows should review this branch and evidence before authorizing any merge or later
Actor/VLM/training changes. Linux did not merge the feature branch.
