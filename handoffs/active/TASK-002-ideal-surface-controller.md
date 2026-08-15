# TASK-002: Idealized Capsule Surface Controller

**Status:** Approved for Linux implementation by the user on 2026-08-15.

**Planning branch:** `workflow/TASK-002-ideal-surface-controller`

**Required Linux branch:** `feature/TASK-002-ideal-surface-controller`

**Delivered-code lineage:** `d7c2119dedc678684d6b46f1f090ce24fb72bbd5` from `origin/feature/TASK-001-p0-coverage-teleop`

**Design authority:** `docs/superpowers/specs/2026-08-15-idealized-capsule-surface-controller-design.md`

**Execution plan:** `docs/superpowers/plans/2026-08-15-ideal-surface-controller.md`

**Required report:** `handoffs/reports/TASK-002-ideal-surface-controller-report.md`

## Objective

Implement a separate Isaac Lab task in which a privileged ideal controller moves the capsule continuously along the approved stomach inner surface from one of fifteen deterministic scalar action IDs. The task shall reuse the delivered 1 Hz camera and occlusion-aware complete-inner-surface coverage evaluator so the project can evaluate high-level exploration independently of magnetic localization and magnetic actuation tracking.

This task validates the ideal motion layer and its manual evaluation interface. It does not authorize reinforcement-learning training, VLM integration, reward optimization, magnetic closed-loop control, camera recalibration, stomach-asset replacement, or modification of the delivered eleven-action magnetic executor.

## Base and Branch Rules

Linux shall fetch `workflow/TASK-002-ideal-surface-controller` and record its exact head as `base_commit` before editing. Linux shall create an isolated worktree and branch `feature/TASK-002-ideal-surface-controller` from that exact head. Linux shall not implement directly on `main`, on the Windows planning branch, or on the prior TASK-001 feature branch.

Linux shall preserve unrelated work. If the required planning head cannot be established without overwriting local changes, report `blocked` and stop.

## Mandatory Preflight Gate

Linux shall inspect the actual capsule rigid body, collider geometry, capsule long axis, camera optical and image-up axes, approved luminal mesh, inward normal convention, initial surface contact, and live root-pose/root-velocity write APIs.

Linux shall report `needs_decision` and stop before controller implementation if the collision geometry is not an unambiguous spherocylinder, if the capsule or camera axes cannot be confirmed, if the approved mesh or its normal orientation is ambiguous, if the initial pose has no valid surface contact, or if continuous kinematic pose and velocity targets cannot be written through a verified Isaac Lab/PhysX API.

Linux may not silently approximate an unknown capsule shape, guess the inward normal, substitute an unverified quaternion convention, or attach the controller to a different stomach surface.

## Authorized Changes

Linux may add the isolated pure ideal-surface controller package, new Isaac action term, dedicated ideal-surface stomach task, task registration, fifteen-action keyboard adapter, manual coverage launcher, preflight and validation scripts, focused tests, user documentation, and required report.

Linux may make the smallest export changes needed to expose the new package and task. Linux may modify `.gitignore` only when newly generated TASK-002 artifacts are not already excluded.

## Forbidden Changes

Linux shall not modify any existing `AtomicAction` ID, action template, magnetic executor state transition, magnetic force model, robot/ASM safety threshold, existing task behavior, P0 coverage geometry or visibility semantics, camera calibration, stomach or robot asset, prior report, or prior acceptance result.

Linux shall not feed coverage, rays, capsule pose, contact geometry, surface normals, active triangles, or controller diagnostics into Actor observations. Linux shall not add RL training, VLM modules, reward terms, automatic coverage planning, dynamic tissue, stochastic slip, magnetic actuation, or recovery motion.

## Frozen Fifteen-Action Contract

The schema version is `ideal_surface_v1`. The scalar action IDs are:

```text
0   HOLD
1   START_TILT_000
2   START_TILT_045
3   START_TILT_090
4   START_TILT_135
5   START_TILT_180
6   START_TILT_225
7   START_TILT_270
8   START_TILT_315
9   TILT_MORE
10  RISE
11  PRECESS_POS
12  PRECESS_NEG
13  ROLL_POS
14  ROLL_NEG
```

The eight initial directions are defined in the local stomach tangent plane relative to the projected camera image-up axis. Positive azimuth follows the right-hand rotation about the inward surface normal.

`START_TILT_*` is valid only in the logical upright state and targets an absolute 15-degree tilt. `TILT_MORE` and `RISE` change the current tilt by 15 degrees along the current tilt plane. `PRECESS_POS/NEG` hold tilt and change azimuth by positive or negative 15 degrees. `ROLL_POS/NEG` use right-hand axial roll and an ideal no-slip surface arc length of 4.0 mm.

## Frozen State and Contact Contract

The Actor boundary is one action every 1.0 simulated second. The controller updates a smooth quintic target at every physics substep and returns exactly one result at the action boundary. Early contact or boundary limitation latches the last safe pose and holds it until the same one-second boundary.

Logical upright entry is at 5 degrees and exit is at 8 degrees with a 0.1-second stability window. Any normal support contact is not side contact. Stable side contact requires separated longitudinal barrel contacts or near-contacts satisfying the approved design thresholds for 0.1 seconds.

Contact-limited and open-boundary-limited motions return `DONE` with flags. A masked manual request returns a normal no-effect result and does not become a device hard failure. `HARD_FAILURE` is reserved for nonfinite state, lost valid surface, nonadjacent surface jump, or actual penetration above `0.05 r_eff`.

The main controller states remain `READY`, `EXECUTING`, and `TERMINAL_FAULT`. Upright, side contact, contact limitation, boundary limitation, and no effect remain orthogonal flags and shall not expand the main state machine.

## Task and Information Isolation

The new Gym task ID is `Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0`. It uses one environment, 240 Hz physics, `decimation=240`, 1 Hz capsule RGB, scalar action shape `(1,)`, and the existing stomach scene.

The new task must not include the magnetic-physics action term or magnetic collision bridge. Capsule pose and mesh truth are authorized only inside the new ideal controller, privileged Critic/evaluator channel, visualization, validation, and offline evidence path. The existing deployable magnetic observation and action contracts remain unchanged.

## Required Acceptance

Linux shall pass pure tests for all fifteen IDs, minimal masks, upright hysteresis, local frames, surface adjacency, disconnected-fold rejection, boundary clipping, spherocylinder support, active anchors, side-contact detection, quintic interpolation, unique start directions, 15-degree tilt and precession, 4.0 mm rolling, request de-duplication, fixed one-second completion, normal saturation, and hard-failure containment.

Linux shall pass task-registration and observation-isolation tests, a rendered startup/exit smoke, all-action stomach integration, and a deterministic 1,000-valid-action stomach run from the approved initial pose. The long run must show finite poses, no nonadjacent jumps, no unexplained hard failures, no hard penetration, exactly one result per request, and one coverage update per unique 1 Hz frame. Final coverage is informational and has no pass threshold in TASK-002.

Linux shall rerun all delivered coverage tests, stage-one pure action-layer tests, P0 GPU/scalar geometry validation, P0 stomach integration, eleven-action table acceptance, and the legacy 9D table smoke.

## Delivery Contract

The final report shall state one of `complete`, `partial`, `needs_decision`, or `blocked`. It shall include base commit, head commit, branch, capsule dimensions and axes, camera axes, surface geometry hash, verified pose-write APIs, every validation command and observed result, all deviations, unverified claims, and every external evidence artifact with path, byte size, and SHA-256.

Linux shall push `feature/TASK-002-ideal-surface-controller` to writable `origin` and shall not merge it. Windows remains responsible for reviewing the diff and evidence and authorizing any later Actor, VLM, or training work.
