# TASK-008 Six-Action Dynamic-Force Controller Design

## Purpose and authority

This document freezes the first implementation of a force-driven six-action capsule controller. It supersedes TASK-005, TASK-006, TASK-007, and TASK-007-R1 as execution authority. Those branches remain evidence and history only.

The implementation starts from `06b15caf9a69bc9c20f85522ce4abbb32c8b9245`, the final recorded head of `origin/feature/TASK-003-dynamic-capsule-force-teleop`. TASK-003 supplies the non-kinematic capsule, gravity, contact, CCD configuration, RGB camera, continuous rendering, and direct PhysX wrench path. The new task shall not import the eleven-action controller, latch controller, virtual magnet, magnetic arm, pose setter, or ideal-surface motion controller.

The TASK-003 report is historical evidence, not a clean physics certificate. It reported correct force application and finite dynamics, but its disposition was `partial` because minimum measured clearance reached approximately `-1.982 mm` and one substep continuity bound was exceeded. TASK-008 may tolerate ordinary overlap, bounce, sliding, and contact variation, but it must preserve this fact and must not claim that TASK-003 already proved penetration-free contact.

## Scope

TASK-008 implements only the six-action dynamic controller, the synchronous one-second invocation interface, flat-table calibration and held-out acceptance, a keyboard one-action-at-a-time stomach viewer, and synchronized external, capsule-RGB, and coverage displays. It does not train a VLM, temporal Actor, Critic, reinforcement-learning policy, or expert dataset.

The deployable policy boundary is one discrete action ID. Capsule pose, velocity, contact, mass properties, and mesh truth are simulation-privileged implementation and evaluation inputs. They must not be added to the future Actor observation. The policy observation remains the action-boundary RGB image and action history.

## Fixed action vocabulary

The action space contains exactly six IDs:

| ID | Name | Meaning |
|---:|---|---|
| 0 | `HOLD` | Apply no active capsule force for one simulated second. |
| 1 | `MOVE_POS` | Apply the positive horizontal lateral force at both hemisphere centers. |
| 2 | `MOVE_NEG` | Apply the negative horizontal lateral force at both hemisphere centers. |
| 3 | `VIEW_POS` | Apply the positive horizontal lateral force only at the camera-side hemisphere center. |
| 4 | `VIEW_NEG` | Apply the negative horizontal lateral force only at the camera-side hemisphere center. |
| 5 | `UP` | Apply world-up force only at the camera-side hemisphere center. |

Every valid ID is executable. There is no action mask, posture eligibility gate, `REJECTED`, automatic substitution, target pose, target distance, early success termination, hidden HOLD conversion, or support-point constraint.

An out-of-range or non-integral API value is an interface error before physics starts. It is not an action outcome available to the Actor.

## Capsule geometry and action frame

The capsule camera is mounted on the local negative-Z end in the TASK-003 asset. Define the camera-directed long-axis unit vector as

\[
\mathbf u_{\mathrm{cam},t}
=
\frac{\mathbf p_{\mathrm{camera\ sphere},t}-\mathbf p_{\mathrm{COM},t}}
{\left\|\mathbf p_{\mathrm{camera\ sphere},t}-\mathbf p_{\mathrm{COM},t}\right\|}.
\]

Preflight must verify the authored capsule radius, cylindrical height, link frame, center of mass, and camera-side sign. For the recorded TASK-003 geometry, the radius is approximately `6.5 mm`, the cylindrical height is approximately `12 mm`, and the hemisphere centers lie at local `Z=+/-6 mm`. Runtime code must derive these values from verified geometry or one verified task-local geometry record instead of inferring them from the camera optical-center offset.

The positive horizontal lateral direction is recomputed on every 240 Hz physics substep:

\[
\mathbf d_t
=
\frac{\mathbf z_{\mathrm{world}}\times\mathbf u_{\mathrm{cam},t}}
{\left\|\mathbf z_{\mathrm{world}}\times\mathbf u_{\mathrm{cam},t}\right\|},
\qquad
\mathbf d^-_t=-\mathbf d_t.
\]

The approved reset and validation population excludes a long axis that is effectively parallel to world Z. Runtime code must nevertheless reject a non-finite or zero-norm calculation before applying a non-finite wrench. It may not invent a surface-normal direction, magnetic direction, or pose correction.

## Endpoint force semantics

Let `r_move`, `r_view`, and `r_up` denote independent ratios of live capsule weight `mg`. Their initial values are all `0.9`, and their accepted search range is `(0, 3.0]`.

MOVE applies

\[
\mathbf F_{\mathrm{camera}}=\frac{1}{2}r_{\mathrm{move}}mg\,\sigma\mathbf d_t,
\qquad
\mathbf F_{\mathrm{other}}=\frac{1}{2}r_{\mathrm{move}}mg\,\sigma\mathbf d_t,
\]

where `sigma` is `+1` for `MOVE_POS` and `-1` for `MOVE_NEG`. The total commanded MOVE force is therefore `r_move*mg`; at the default, each end receives `0.45mg` and the total is `0.9mg`.

VIEW applies

\[
\mathbf F_{\mathrm{camera}}=r_{\mathrm{view}}mg\,\sigma\mathbf d_t
\]

only at the camera-side hemisphere center. UP applies

\[
\mathbf F_{\mathrm{camera}}=r_{\mathrm{up}}mg\,\mathbf z_{\mathrm{world}}
\]

only at that center.

The authoritative physical meaning is force at the specified point. If the installed Isaac Lab/PhysX API supports point-force application, use it directly with zero explicit couple torque. If it exposes only a body-COM wrench, compose the exactly equivalent generalized wrench

\[
\mathbf F=\sum_i\mathbf F_i,
\qquad
\boldsymbol\tau_{\mathrm{COM}}=\sum_i(\mathbf p_i-\mathbf p_{\mathrm{COM}})\times\mathbf F_i.
\]

The implementation must use exactly one of these paths and record which path was verified. It must never apply both point forces and the equivalent lever-arm torque, because that would double-count torque.

## One-second timing contract

Physics runs at `240 Hz`, the environment and engineering control loop at `60 Hz`, the capsule camera and coverage evaluator at `30 Hz`, rendering targets `60 FPS`, and the future Actor decides at `1 Hz`.

Each macro action contains exactly 240 physics substeps. MOVE and VIEW apply zero active force during substeps `0..47`, apply their endpoint force during substeps `48..191`, and apply zero active force during substeps `192..239`. This is exactly `0.2 s` wait, `0.6 s` force, and `0.2 s` wait.

UP applies its endpoint force during all substeps `0..239`. HOLD applies no active force during all substeps. Gravity, contact, friction, inertia, and authored passive damping remain active during every phase.

The environment remains a 60 Hz stepping environment so rendering and 30 Hz coverage can update during a macro. A synchronous macro runner owns the 60 environment updates that make one Actor transition. It returns only after one simulated second.

At the one-second UP boundary, the runner completes the final physics step, renders and records the boundary RGB image while the UP force remains the current commanded wrench, then clears the active wrench without advancing physics. Model inference occurs while simulation time is paused. MOVE, VIEW, and HOLD follow the same boundary capture and acknowledgement sequence, although their active force is already zero at the boundary.

Wall-clock VLM inference time is not part of the simulated one-second action. The returned transition is

\[
I_k\rightarrow a_k\rightarrow 1\ \mathrm{s\ of\ simulated\ dynamics}\rightarrow I_{k+1}.
\]

## Keyboard inspection contract

The stomach launcher uses rising-edge commands: one key press submits one action, repeated key events do not resubmit it, and no second action is accepted while the current one-second macro is running.

The default mapping is `Space=HOLD`, `D=MOVE_POS`, `A=MOVE_NEG`, `E=VIEW_POS`, `Q=VIEW_NEG`, and `W=UP`. `Backspace` resets the episode and coverage, `F12` saves a synchronized snapshot, and `Escape` exits.

After an action and boundary capture finish, physics stepping pauses while Kit rendering and UI event processing continue. The external view, capsule RGB, and coverage view remain visible until the next key press. This pause is a keyboard inspection mechanism and is not a HOLD action or a training transition.

## Flat-table calibration and acceptance

Every randomized trial begins with an authorized reset-only pose and velocity write. The reset samples a safe table position, a uniform horizontal heading, and a uniform long-axis roll while excluding suspension, initial penetration, and a near-world-vertical long axis. Active force is zero while the capsule settles. The settle operation is outside the one-second action and is recorded separately.

Calibration and final acceptance use disjoint seed manifests. Each non-HOLD action receives 20 calibration trials per evaluated force candidate. The selected profile then receives 20 new held-out trials per action exactly once. A final action passes with at least 16 successes out of 20. MOVE positive and negative share `r_move`; VIEW positive and negative share `r_view`; both directions in a shared group must pass.

MOVE records the force-onset state at `t=0.2 s`. With `d_0.2` and `p_0.2` frozen for evaluation only, MOVE succeeds when

\[
(\mathbf p_{1.0}-\mathbf p_{0.2})\cdot\mathbf d_{0.2}\ge 0.005\ \mathrm m.
\]

No rolling angle, slip ratio, lateral-error, contact-point, or final-speed condition is part of MOVE acceptance.

VIEW records `u_0.2` and `d_0.2` at force onset. Define `k_0.2=normalize(u_0.2 cross d_0.2)`. The signed commanded-plane angle at `t=1.0 s` is

\[
\Delta\theta_{\mathrm{view}}
=
\operatorname{atan2}\left(
\mathbf k_{0.2}\cdot(\mathbf u_{0.2}\times\mathbf u_{1.0}),
\mathbf u_{0.2}\cdot\mathbf u_{1.0}
\right).
\]

VIEW succeeds when the signed angle in the requested direction is at least `15 deg`. Translation and support-point drift are not acceptance conditions.

UP records the initial horizontal long-axis projection as its no-crossing reference. At `t=1.0 s`, it succeeds only when the camera side is at least `45 deg` above the world horizontal plane and the continuously sampled camera-directed long axis never crosses world vertical to the opposite side. The boundary image must be captured before UP force release.

HOLD passes its contract when the recorded active capsule force is zero for all 240 substeps. It is not required to make the dynamic body motionless.

The calibration sequence begins at `0.9mg`, multiplies by `1.25` until the group passes or the next value would exceed `3.0mg`, evaluates `3.0mg` as the final coarse bound, and then performs three deterministic midpoint refinements between the nearest lower failure and first pass. The smallest refined candidate that passes the 20 calibration trials is selected. A candidate with any FAULT is ineligible even if at least 16 other samples pass.

FAULT is reserved for non-finite state, physics execution failure, the capsule completely traversing the table, or the capsule leaving the derived table test region. Bounce, oscillation, sliding, ordinary contact switching, insufficient displacement, insufficient angle, and mild solver overlap are ordinary pass/fail observations rather than FAULT.

If no candidate up to `3.0mg` passes, Linux must complete the authorized sweep, preserve every result, and return `partial`. It must not change action duration, collision geometry, friction, damping, mass, inertia, gravity, or the acceptance threshold.

## Stomach migration and coverage display

Stomach execution is a qualitative keyboard inspection only. It receives the exact selected flat-table force profile and timing with no stomach calibration, no stomach success percentage, and no automatic scene-specific adjustment.

The Kit application must show three synchronized panels: the normal external simulation view, the live 30 Hz capsule RGB view, and a coverage-only stomach view. The panels may be docked or laid out according to the installed Kit UI API, but they must be simultaneously visible and must represent the same simulation timeline.

Coverage accumulates on every unique 30 Hz capsule-camera frame, including frames recorded inside a one-second macro. Actor observation remains one boundary frame per second; coverage frequency does not change Actor frequency.

A vertex is visible only when it lies inside the circular camera field of view, is no farther than `50 mm`, passes the first-hit occlusion test, and the first-hit stomach face is camera-facing according to the authored mesh orientation. The normal test uses the first-hit triangle normal and camera-to-hit ray direction and requires a strictly camera-facing dot-product sign with a small numerical tolerance. Existing coverage code remains unchanged by default; TASK-008 explicitly enables this additional normal gate.

Coverage is the cumulative union of valid visible vertices over the episode. The coverage view colors uncovered and covered vertices differently and displays the cumulative covered count and percentage. Coverage truth is evaluator-only and must not enter the future Actor observation.

## Delivery and evidence

Linux implements on `feature/TASK-008-six-action-dynamic-force-controller` created from the exact fetched head of `workflow/TASK-008-six-action-dynamic-force`. It does not merge.

The report must include the planning head, implementation head, branch, verified point-force or equivalent-wrench API path, live geometry and mass, selected force profile, every calibration candidate, calibration and held-out seed manifests, per-action `success/20` counts, all FAULTs, timing evidence, boundary capture ordering evidence, test results, stomach launch command, synchronized-view evidence, and external artifact paths with byte counts and SHA-256 hashes.

Linux may report `complete` when every flat held-out action passes, no selected candidate contains a FAULT, the synchronous timing and RGB boundary tests pass, and the stomach three-view launcher is operational. Human judgment of whether the stomach motion looks useful remains explicitly unverified until Windows review. Linux reports `partial` if the interface is implemented but a flat gate or required launcher gate does not pass after exhausting the authorized search.
