# TASK-007-R1 Generalized-Wrench and Surface-Frame Closed-Loop Design

## 1. Purpose and evidence boundary

TASK-007-R1 is a corrective continuation of TASK-007. It does not replace the finite-magnet virtual-actuator architecture, relax the one-second action contract, or start a new controller family. Its purpose is to remove two code-confirmed sources of directional asymmetry, introduce one direction-independent terminal controller, execute a real multi-candidate calibration, and then resume the quantitative gates that TASK-007 did not reach.

The exact implementation base is the Linux report commit `29c459637b0d36c8a289ed2f8553e8e896120aa3` on `feature/TASK-007-virtual-magnet-closed-loop`. The underlying implementation HEAD recorded by that report is `1e1b9f2975b115edbc40dfd843c4eaf5f3425439`. Existing TASK-007 code, tests, manifests, report, and external evidence hashes remain immutable baseline evidence.

The TASK-007 report established that one VIEW direction and stable HOLD could pass single development samples, both MOVE signs could produce approximately 5 mm displacement but missed terminal angular-speed acceptance, and most VIEW directions missed their angular target. It did not run formal 20-sample held-out, disturbance, stomach, or 100-action gates. Therefore R1 must not describe the old result as a generalized-controller failure or a stomach failure.

## 2. Frozen constraints

The capsule remains a gravity-enabled, non-kinematic dynamic rigid body. Motion may be caused only by the repository-local finite-magnet model's force and couple torque together with PhysX gravity, contact, friction, and configured passive damping. R1 may not write capsule pose or velocity during actions, lock degrees of freedom, teleport, project, create a temporary joint, or inject an arbitrary desired wrench directly.

The virtual external magnet remains a smooth, non-colliding analytical 6-DOF pose source. It has no rigid-body or collision API and does not use the mechanical arm or Ball joints. Every controller-originated force and couple torque applied through PhysX must come from the finite-magnet forward model at the current virtual-magnet pose.

The public action IDs remain exactly `0 HOLD_VIEW`, `1 VIEW_UP`, `2 VIEW_UP_RIGHT`, `3 VIEW_RIGHT`, `4 VIEW_DOWN_RIGHT`, `5 VIEW_DOWN`, `6 VIEW_DOWN_LEFT`, `7 VIEW_LEFT`, `8 VIEW_UP_LEFT`, `9 MOVE_SIDE_POS`, and `10 MOVE_SIDE_NEG`. Public results remain exactly `COMPLETED`, `REJECTED`, and `FAULT`.

Physics and finite-magnet evaluation remain 240 Hz, feedback remains 60 Hz, and RGB/action boundaries remain 1 Hz. Every request occupies exactly 1.000 seconds or 240 physics substeps. VIEW remains a relative 15-degree target. MOVE remains a signed 5 mm target with a 4-to-6 mm pass interval and the existing unsigned 45-degree tilt plus recent-sidewall-contact prerequisite.

The last 0.1-second mean linear speed must remain at most 2 mm/s and mean angular speed at most 0.1 rad/s. VIEW target error remains at most 3 degrees and support tangent drift at most 2 mm. No direction-specific gains, thresholds, timings, force limits, or scene-specific controller parameters are permitted.

## 3. Correct generalized magnetic wrench

Magpylib returns the force on the capsule magnet and a magnetic couple torque about the target magnet's reference center. PhysX receives that force at the capsule magnet center, which is offset from the capsule center of mass. The equivalent magnetic wrench about the capsule center of mass is therefore

\[
\mathbf F_{\mathrm{COM}}=\mathbf F_{\mathrm{mag}},
\]

\[
\boldsymbol\tau_{\mathrm{COM}}
=
\boldsymbol\tau_{\mathrm{couple}}
+
(\mathbf p_{\mathrm{mag}}-\mathbf p_{\mathrm{COM}})\times\mathbf F_{\mathrm{mag}}.
\]

The numerical pose Jacobian and inverse residual must use the six-vector

\[
\mathbf w_{\mathrm{mag,COM}}
=
[\mathbf F_{\mathrm{mag}},\boldsymbol\tau_{\mathrm{COM}}].
\]

The physical application path must remain force at `p_mag` plus the original magnetic couple torque at that point. It must not apply `tau_COM` as the API couple torque, because doing so would count the lever-arm moment twice. The bridge must expose both representations and assert their equivalence.

Telemetry must distinguish `magnetic_force_world_n`, `magnetic_couple_torque_world_nm`, `magnetic_lever_torque_world_nm`, `magnetic_generalized_torque_com_world_nm`, and the actual PhysX force/couple/position triplet. Tests must fail if these fields are conflated.

Known passive drag may be subtracted when converting a desired rigid-body generalized wrench into a desired magnetic generalized wrench. Contact reaction is not blindly cancelled. The controller records a bounded estimate of the contact moment about the center of mass and treats it as a measured disturbance. Any compensation gain for that estimate is shared across all directions, bounded, filtered, and disabled in the first generalized-wrench diagnostic so the exact lever-arm correction can be evaluated independently.

## 4. Surface-action-frame nominal magnet pose

The TASK-007 nominal magnet position was fixed in the capsule's full rotation frame. Because long-axis roll is intentionally uncontrolled, this made an uncontrolled state rotate the actuator reference. R1 replaces that reference with a local surface-action frame frozen at action acceptance.

Let `n` be the frozen inward surface normal. Let `q` be the action's preferred tangent direction. The surface-action frame is

\[
\mathbf e_1=\operatorname{normalize}\left(\mathbf q-(\mathbf q\cdot\mathbf n)\mathbf n\right),
\]

\[
\mathbf e_2=\operatorname{normalize}(\mathbf n\times\mathbf e_1),
\qquad
R_s=[\mathbf e_1\;\mathbf e_2\;\mathbf n].
\]

For VIEW, `q` is the frozen target optical-axis displacement projected into the surface tangent plane. If that projection is degenerate, the controller tries the frozen desired swing axis, then frozen camera right, then the world basis axis least aligned with `n`. For MOVE, `q` is the frozen signed movement direction. HOLD preserves the last valid surface-action frame; reset constructs it from projected camera right with the same deterministic fallback.

One shared profile vector `nominal_position_surface_m` and one shared rotation offset `nominal_quaternion_surface_xyzw` define the nominal virtual-magnet pose:

\[
\mathbf p_{m,\mathrm{nom}}
=
\mathbf p_{\mathrm{COM}}
+
R_s\mathbf r_{m,\mathrm{surface}},
\]

\[
R_{m,\mathrm{nom}}=R_sR_{m,\mathrm{offset}}.
\]

The frame axes are frozen for the one-second action, while the origin follows the current capsule center of mass at each 60 Hz update. This preserves the frozen action direction and prevents the virtual magnet from separating when the capsule translates. The same profile vector and rotation offset apply to all eleven actions. There are no per-ID offsets.

Every new nominal target is reached through the existing smooth 240 Hz interpolation and trust region. Action acceptance may not teleport the virtual magnet. Quaternion sign continuity and finite separation limits remain mandatory.

## 5. Direction-independent prescribed-time terminal control

The fixed low stabilization torque cap introduced in TASK-007 produced a tracking-versus-terminal-speed tradeoff. R1 replaces it with one state-dependent law shared by all VIEW directions and HOLD. It does not select gains by action ID or image direction.

At every 60 Hz update, the controller computes the minimal swing error vector `e`, twist-removed angular velocity `omega_perp`, and time remaining `T_r`. During the final 0.2 seconds, the reference angular acceleration is

\[
\boldsymbol\alpha^*
=
\frac{6\mathbf e}{T_r^2}
-
\frac{4\boldsymbol\omega_\perp}{T_r},
\]

with `T_r` floored at one feedback interval before applying finite shared acceleration and torque limits. The desired rigid-body torque uses the live world-frame inertia tensor:

\[
\boldsymbol\tau^*_{\mathrm{rigid}}
=
I_w\boldsymbol\alpha^*
+
\boldsymbol\omega\times(I_w\boldsymbol\omega).
\]

The desired magnetic generalized torque is obtained after subtracting known passive drag and, only when enabled after its isolated diagnostic, a bounded filtered fraction of the estimated contact moment. The finite-magnet pose inverse then attempts to realize that magnetic generalized wrench. Saturation remains telemetry and validation evidence, not a `FAULT`.

The first 0.8 seconds retain the quintic optical-axis reference and reference angular velocity. The final law changes only the way the controller reaches the same target with near-zero terminal speed. MOVE receives the same remaining-time endpoint stabilization principle for translation and passive angular damping, but no active long-axis roll or tilt target is introduced.

## 6. Directional covariance requirements

The pure magnetic model, generalized-wrench transform, surface-action frame, inverse solver, and controller command must be equivariant under a common rotation about a flat surface normal. Rotating the capsule state, virtual-magnet state, camera frame, normal, and requested image-relative action by a common yaw must rotate force, torque, nominal pose, and response without changing their scalar errors.

Unit tests must verify this property for yaw rotations in 45-degree increments. A live diagnostic must run all eight VIEW actions from rotationally matched flat initial states and log desired generalized torque, finite-model generalized torque, lever-arm torque, applied API tuple, contact-moment estimate, angular acceleration, target error, terminal speed, magnet relative pose, and solver saturation.

The directional development gate is five independent samples for every valid action class with at least four passes per class. Constrained VIEW samples are reported and replaced; they do not fill the five-sample quota. This gate is a development screen and does not replace the formal 20-sample held-out acceptance.

## 7. Real calibration search

The current calibration executable evaluates exactly one checked-in profile despite accepting a search-budget argument. R1 must implement a deterministic multi-candidate search. The executable receives a base profile and emits immutable candidate profiles, candidate digests, seed manifests, per-direction metrics, ranking, rejection reasons, and the selected profile.

The first search stage uses at least 32 deterministic space-filling candidates over the authorized controller parameters. The second stage performs bounded local refinement around the best candidates, up to the existing maximum budget of 200 candidates. Early termination is allowed only after the five-sample-per-action development gate passes. A `needs_decision` return for control performance requires at least 32 distinct finite candidates, completion of local refinement for the best finite candidates, and a written failure taxonomy.

The ranking is lexicographic: true `FAULT` count, number of action classes meeting the development gate, worst action-class pass rate, worst VIEW terminal error, worst boundary angular speed, worst MOVE displacement error, then mean performance. An average score may not hide a failed direction.

Authorized search variables are the shared orientation/translation gains, prescribed-time limits, contact-disturbance compensation gain, inverse weights/damping/regularization/trust regions, filter constants, finite magnetic force/couple limits, and shared surface-frame nominal pose. Capsule mass, inertia, geometry, gravity, friction, contact materials, magnetic material constants, stomach geometry, and PhysX configuration remain fixed.

Any sample used to select or modify a profile is development data. Formal held-out manifests are generated only after the selected profile digest is frozen. If later stomach development changes that digest, both flat formal gates must be regenerated and rerun.

## 8. Acceptance order

R1 first preserves the TASK-007 baseline report and reproduces one eight-direction diagnostic with the old profile. It then runs isolated generalized-wrench diagnostics, surface-frame covariance diagnostics, prescribed-time terminal diagnostics, and the five-sample development gate in that order. Each structural stage records its own profile/code digest so its effect is attributable.

After development passes, flat no-disturbance validation uses 20 new held-out samples for every action class and requires at least 16 passes per class. Invalid MOVE is a separate 20-sample class for each sign and must execute one-second HOLD with `REJECTED`.

Only after flat no-disturbance passes does flat disturbance validation run 20 new held-out samples per action with at least 16 passes. Every disturbed trial has an identical feedback-disabled pair, and the report must demonstrate a changed magnet command and better endpoint recovery under feedback.

Only after both flat gates pass may stomach development and held-out validation begin. The stomach scene receives no injected disturbance and no scene-specific controller branch. Named development regions may tune the single shared profile; final held-out regions use 20 samples per action and require at least 16 passes. Any profile change forces complete flat revalidation with new manifests.

Finally, flat and stomach each run the same no-reset 100-action ID sequence. The flat sequence may include small disturbances; the stomach sequence may not. Finite low-effect actions do not stop the sequence. Only a true `FAULT` stops it.

## 9. Reporting and stop conditions

Large logs, videos, screenshots, and caches stay outside Git and are reported with absolute paths, byte sizes, and SHA-256 values. Compact per-stage JSON summaries, seed manifests, profile candidate tables, and the final Markdown report are committed. The previous `/tmp/task007-*.json` files are copied into a durable Linux evidence directory or summarized into committed compact JSON before further tuning; missing files are reported explicitly rather than reconstructed.

R1 returns `complete` only after generalized-wrench equivalence, covariance tests, real calibration, both flat gates, stomach held-out acceptance, both 100-action sequences, keyboard visualization, and regressions complete. It returns `partial` if both flat gates pass but stomach remains below threshold after authorized shared-profile tuning. It returns `needs_decision` only after the defined minimum multi-candidate search and diagnosis are exhausted or a new design-changing ambiguity is demonstrated.

The one-second duration, 0.1 rad/s terminal angular-speed limit, action definitions, or physics parameters may not be relaxed inside R1. If R1 still fails after its authorized search, the report must preserve enough directional and generalized-wrench evidence for the Windows side to decide whether those frozen constraints should change.
