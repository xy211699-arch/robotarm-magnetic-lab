# Hybrid Latched Eleven-Action Controller Design

## 1. Purpose and research boundary

TASK-006 replaces TASK-005's continuously active READY_HOLD and terminal settling behavior with `hybrid_latched_v1`. Motion inside an accepted action remains force/torque driven and is integrated by GPU PhysX. At observation boundaries the controller is allowed to clear velocity and hard-lock all six capsule degrees of freedom so every 1 Hz policy RGB frame is captured from a stationary pose.

This is a simulation-privileged ideal actuator for validating the visual exploration policy. It is not evidence that a physical magnetic capsule can stop instantaneously, and it must not be described as a pure dynamic controller or a realizable magnetic stabilization law.

Implementation baseline is Linux TASK-005 commit `67b7bf44747f08422add0cee7e6b94280bbeff6d`. TASK-005's failed randomized report remains immutable evidence; TASK-006 receives a new profile, report, branch, and acceptance result.

## 2. Frozen scope

The public action space remains exactly eleven scalar discrete IDs:

| ID | Action |
|---:|---|
| 0 | `HOLD_VIEW` |
| 1 | `VIEW_UP` |
| 2 | `VIEW_UP_RIGHT` |
| 3 | `VIEW_RIGHT` |
| 4 | `VIEW_DOWN_RIGHT` |
| 5 | `VIEW_DOWN` |
| 6 | `VIEW_DOWN_LEFT` |
| 7 | `VIEW_LEFT` |
| 8 | `VIEW_UP_LEFT` |
| 9 | `MOVE_SIDE_POS` |
| 10 | `MOVE_SIDE_NEG` |

The nine-grid camera mapping, keyboard mapping, relative 15 degree VIEW cone half-angle, 240 Hz physics, strict one-second/240-substep action boundary, continuous rendering target, `0.9mg` shared MOVE force, and three public action results `COMPLETED`, `REJECTED`, and `FAULT` remain unchanged. No action mask, VLM, Actor, Critic, reward, coverage, action chunk, teacher, dataset, robot arm, magnet, asset, mass, inertia, material, gravity, solver, or CCD change is in scope.

GPU dynamics remains enabled. The known runtime warning that sweep CCD is disabled under GPU dynamics is recorded, not fixed or hidden. A result may not claim CCD was active.

## 3. Unified one-hertz cycle

Every policy cycle follows the same order:

```text
LATCHED_READY
  -> capture one policy RGB frame
  -> accept exactly one action ID
  -> unlock only if the action requires motion
  -> execute/count exactly 240 physics substeps
  -> ensure six-DOF latch and zero wrench
  -> confirm zero linear/angular velocity
  -> publish COMPLETED or REJECTED
  -> expose the next policy RGB frame
```

The episode begins by obtaining a finite reset state, clearing velocity and wrench, latching it, and only then exposing frame zero. Rendering and the physics clock continue while latched. Keyboard visualization may render at 60, 120, or 240 FPS, but only the explicit one-second boundary frame is exposed as the policy RGB observation.

An early VIEW latch does not shorten the action. The controller stays latched while the remaining substeps are counted. A rejected MOVE also occupies all 240 substeps while remaining latched. No new request is accepted, cached, queued, or allowed to preempt the current one.

Camera capture has an explicit barrier: latch application, zero-wrench application, zero-velocity confirmation, and fresh camera sensor update must complete in that order. A camera frame produced before latch confirmation is not a policy frame.

## 4. Latch authority and backend selection

The latch profile records `preferred_backend=dynamic_lock_flags`, `fallback_backend=kinematic`, and a single `selected_backend`. The selected value begins as `dynamic_lock_flags` and may be changed to `kinematic` only after the failed preferred-backend evidence is written. The preferred backend keeps the capsule dynamic and applies all six PhysX rigid-dynamic lock flags: linear X/Y/Z and angular X/Y/Z. On latch it copies the current actual pose for auditing, clears the current COM wrench, writes zero linear and angular velocity, applies all six lock flags, and verifies the flags and zero velocity by readback. It never projects or snaps the capsule to a target pose.

On unlock it clears the old COM wrench, writes zero linear and angular velocity, clears all six lock flags, verifies the readback, and lets the new action generate its first wrench in the same physics substep. There is no unlocked zero-control gap.

The Linux executor must first run a minimal GPU runtime gate. Static USD attribute presence is insufficient: the test must show that lock flags can be changed while the CUDA simulation is active, that the pose remains fixed for one simulated second, and that unlock does not create an extra trajectory transient.

If and only if this runtime gate fails, the executor may select the approved fallback backend: dynamic-to-kinematic switching while retaining the exact current pose and zero velocity, followed by kinematic-to-dynamic switching at the next action boundary. Because kinematic/static contact reporting may disappear, MOVE eligibility still uses the latched contact snapshot defined below. If both backends fail their runtime gates, TASK-006 stops with `needs_decision`.

Per-substep pose overwrites, repeated teleportation, temporary joints, sleep-only locking, target pose projection, penetration correction, surface snap, and hidden recovery are forbidden.

## 5. VIEW and HOLD behavior

At action acceptance, each VIEW freezes the camera frame from the locked RGB observation. The selected nine-grid direction defines one immutable target optical axis at a relative 15 degree cone half-angle. The target is not recomputed while the capsule moves.

VIEW unlocks and uses the existing TASK-005 force/torque path. At the first physics substep satisfying both conditions below, it clears wrench and immediately latches the current actual six-DOF pose:

\[
\arccos(\mathbf u_{\mathrm{actual}}\cdot\mathbf u_{\mathrm{target}})\leq3^\circ
\]

\[
\lVert\Delta\mathbf p_{\mathrm{support,tangent}}\rVert\leq2\,\mathrm{mm}
\]

No angular-speed threshold and no multi-substep dwell are required. This intentionally removes kinetic energy when the moving trajectory first passes through the accepted terminal region.

Real camera-hemisphere contact has priority over the target gate. The first detected camera contact immediately clears wrench and latches the actual pose even when the angle or support-drift criterion is not met. The action remains a normal `COMPLETED` action with `constrained=true` in telemetry.

If neither target latch nor camera-contact latch occurs, the controller runs to the strict one-second boundary, latches the actual terminal pose, and returns `COMPLETED`. Target error and support drift remain validation telemetry only; no fourth action result and no `target_reached` field are added.

`HOLD_VIEW` never invokes the TASK-005 dynamic hold law. It keeps the already latched pose for all 240 substeps and returns `COMPLETED`. If reset recovery presents an unlocked finite state, HOLD first clears velocity/wrench and latches that actual state.

## 6. MOVE behavior and latched contact snapshot

MOVE eligibility is evaluated once at action acceptance. Tilt must be at least 60 degrees. Sidewall eligibility comes from a contact snapshot captured at the most recent latch event, not from nonzero contact forces generated while locked. The snapshot stores the actual latest dynamic contact classification and its source physics substep; it is immutable until the next latch.

An ineligible MOVE remains locked, counts 240 substeps, applies zero wrench, returns `REJECTED`, and then exposes the next locked RGB frame. It is not a FAULT and does not allow an early policy decision.

An eligible MOVE unlocks and retains TASK-005's exact schedule: 0.25 seconds free, 0.5 seconds fixed COM force, and 0.25 seconds free. At 0.25 seconds it computes and freezes the signed movement direction from the then-current capsule axis and local surface tangent plane. Torque remains zero. Reaching 5 mm does not trigger an early latch. At substep 240 it clears wrench, zeros velocity, latches the actual pose, and returns `COMPLETED`.

The shared force remains:

\[
\mathbf F=0.9mg\,\mathbf d_{\mathrm{move}}
\]

No positive/negative direction-specific retuning is allowed unless a new plan is approved after TASK-006 reports failure.

## 7. Public results, telemetry, and FAULT policy

The only public results remain `COMPLETED`, `REJECTED`, and `FAULT`. `REJECTED` remains exclusive to a MOVE whose latched start state fails the tilt or latched-sidewall prerequisite. Normal camera contact, early latch, failure to enter the target gate, finite slip, finite oscillation before latch, low displacement, and direction degeneracy remain `COMPLETED` and are represented by telemetry.

`FAULT` remains restricted to NaN/Inf, missing rigid-body state, backend readback mismatch, impossible lifecycle transition, program exception, or unrecoverable numerical state. Ordinary stomach contact instability is not a FAULT.

Internal telemetry may add `latched`, `latch_backend`, `latch_reason`, `latch_substep`, `policy_frame_ready`, latched contact classification, lock/unlock readback, and paired-release diagnostics. These are not additional action results and must not become privileged Actor observations. The future Actor receives RGB, action history, and one of the existing three result values only.

## 8. Release-safety acceptance

Release safety uses paired trials from identical pose, zero velocity, contact state, action ID, profile, and random seed. The baseline member executes directly from the state. The latch member holds the same state for one simulated second, unlocks, and executes the same action.

During the first 0.05 seconds after action start, the paired maximum differences must satisfy:

\[
\max_t\lVert\mathbf p_{\mathrm{latched}}(t)-\mathbf p_{\mathrm{direct}}(t)\rVert\leq0.5\,\mathrm{mm}
\]

\[
\max_t\arccos(\mathbf u_{\mathrm{latched}}(t)\cdot\mathbf u_{\mathrm{direct}}(t))\leq1^\circ
\]

The lock/unlock API call itself must not change pose. Each motion action receives at least ten paired randomized starts. A backend that violates any paired gate is not accepted even if its final action metric passes.

## 9. Flat quantitative acceptance

The flat task is the blocking gate. Every action receives at least ten stratified randomized initial states. Every normal action consumes exactly 240 physics substeps, every policy RGB frame is captured only while the backend confirms six-DOF latch and zero velocity, and no unclassified status is permitted.

Each unblocked VIEW sample must latch through the target gate with terminal target-axis error at most 3 degrees and support tangential drift at most 2 mm. Camera-contact samples are reported separately and are not used to satisfy the ten unblocked samples. HOLD must show no measurable six-DOF drift during the one-second interval.

Each MOVE receives at least ten eligible and ten ineligible starts. Ineligible starts must return `REJECTED`; eligible starts must return `COMPLETED`, and at least 90 percent must achieve signed displacement of at least 5 mm before the terminal latch. A planned-valid sample that fails the actual latched predicate is reclassified and replaced; it is not counted as an eligible controller failure.

After per-action gates pass, run a fixed no-reset 100-action sequence with every action ID appearing at least five times. Requests issued while an action is active must be discarded, and every boundary frame must satisfy the latch/camera barrier.

## 10. Stomach migration and visualization

Only after all flat gates pass may the executor migrate the same code and the same frozen dynamic and latch profile digests to the existing stomach scene. No stomach-specific gain, force, threshold, latch, timing, collision, mesh, or reset adaptation is permitted.

The stomach run uses the same fixed 100-ID sequence and then launches keyboard visualization. Continuous rendering remains selectable at 60, 120, or 240 FPS, default 120 FPS. The terminal prints action ID, result, latch reason/backend, substeps, pose/velocity at policy capture, contact snapshot, VIEW error/drift, MOVE signed displacement, actual wall FPS, and both profile digests. No HUD is required.

Stomach MOVE is observational and does not receive a second 5 mm tuning gate. The user will judge the visual effect after Linux provides reproducible launch commands and evidence.

## 11. Stop conditions and evidence

If dynamic lock flags fail the runtime gate, record the exact API, warning/error, readbacks, trajectories, and hashes before trying the kinematic fallback. If both backends fail, return `needs_decision`. If flat action or release-safety gates fail, return `needs_decision` and do not run stomach migration. If flat passes but the unchanged stomach task cannot initialize or reaches a true system FAULT, return `partial`. Only flat pass, release pass, no-reset stress pass, same-digest stomach run, regression pass, and evidence completion permit `complete`.

The report must include base/head/branch, exact commands, selected backend and fallback decision, dynamic/latch profile digests, per-action randomized summaries, paired release metrics, policy-frame latch/velocity checks, status counts, wall FPS, GPU CCD warning, regression results, stomach evidence status, external artifact paths, byte sizes, and SHA-256 values. Large logs, images, videos, datasets, and simulation caches remain outside normal Git history.
