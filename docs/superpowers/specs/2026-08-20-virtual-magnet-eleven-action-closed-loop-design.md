# Virtual-Magnet Eleven-Action Closed-Loop Controller Design

## 1. Purpose and research boundary

TASK-007 replaces the unsuccessful direct-force and rigid-latch controller attempts with a closed-loop magnetic controller derived from the validated open-loop magnetic implementation. The capsule remains a gravity-enabled, non-kinematic dynamic rigid body. Its motion is produced only by the finite-magnet model's force and torque together with PhysX gravity, contact, friction, and damping. The controller must never write capsule pose or velocity during an action, make the capsule kinematic, teleport it, lock its degrees of freedom, or inject an arbitrary desired wrench directly into the capsule.

The mechanical arm and the three-axis Ball assembly are not part of this task. They are replaced by a non-colliding virtual external magnet whose finite position and orientation are the actuator variables. The virtual magnet may be rendered as a debug object, but it has no collision or rigid-body role. The analytical magnet model computes the wrench produced at the capsule magnet, and that computed wrench is the only controller-originated wrench applied to the capsule.

This is an ideal simulation controller for validating the eleven-action interface and the later visual exploration policy. It is not evidence that a real robot arm can reproduce the virtual magnet trajectory, satisfy hardware limits, or avoid the patient. Capsule truth feedback is permitted inside this simulation-only controller, but it must not enter the future Actor observation.

The immutable implementation baseline is commit `bff26174ebd1aff2800883a0afdd5295f4f222d1` on `workflow/OPEN-LOOP-MAGNETIC-CONTROLLER-linux-response`. The old open-loop implementation and its archived evidence remain the regression reference; TASK-007 must not rewrite or reinterpret those results.

## 2. Why the open-loop implementation is retained

The old controller achieved the most convincing motion because it did not directly prescribe capsule motion. It oriented a finite external magnet to establish magnetic alignment, translated the field source to create a lateral gradient, and allowed surface friction to convert capsule translation into passive rolling. The capsule stayed a dynamic rigid body throughout. This coupling is retained.

Its principal limitation is open-loop execution. The prescribed arm and Ball trajectories do not correct for the capsule's actual pose, velocity, contact, or displacement. When contact delays the capsule, the external magnet can continue along its independent world trajectory and separate from the capsule. TASK-007 removes that independent trajectory: the virtual magnet's nominal pose follows the current capsule pose, while closed-loop corrections are solved around that moving relative pose.

## 3. Time contract and lifecycle

PhysX and analytical magnetic-wrench evaluation run at 240 Hz. The closed-loop target is recomputed at 60 Hz. Between 60 Hz updates, the virtual magnet pose follows a smooth interpolation evaluated at 240 Hz. A virtual magnet pose must never jump between physics substeps.

The policy and RGB boundary remain exactly 1 Hz. Every accepted request occupies exactly 1.000 seconds, equal to 240 physics substeps. A request submitted while another action is executing is discarded immediately and is not queued. The only internal no-request value is `-1`; it is not part of the Actor action space.

Every action uses the same lifecycle:

```text
READY at a 1 Hz RGB boundary
  -> accept one scalar discrete action ID
  -> freeze the action's start frame and targets
  -> execute 240 physics substeps
  -> publish one of COMPLETED, REJECTED, or FAULT
  -> sample the next 1 Hz RGB frame
  -> return to READY
```

The final 0.2 seconds of VIEW and MOVE are an active magnetic stabilization interval, not a pose lock. HOLD uses magnetic feedback for the full second. The terminal RGB frame is valid only if the last 0.1-second mean capsule linear speed is at most 2 mm/s and mean angular speed is at most 0.1 rad/s. A finite miss of this stability gate is an action-validation failure or low-effect event, not automatically a `FAULT`.

## 4. Public action and result contracts

The public action space is exactly eleven scalar discrete IDs:

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

The nine VIEW directions use the camera-image coordinate convention. `HOLD_VIEW` is the center cell. The eight non-center targets are the eight neighboring cells in clockwise image order beginning at image up.

The only public results are `COMPLETED`, `REJECTED`, and `FAULT`. `REJECTED` is used only when a requested MOVE fails its start-state eligibility predicate. A rejected MOVE still consumes all 240 substeps by executing HOLD. Normal camera-end contact, finite tracking error, ordinary wall obstruction, low displacement, finite oscillation, and inverse-solver saturation are not `FAULT`. They are represented by telemetry and may make a validation sample fail.

`FAULT` is reserved for non-finite state or command values, missing required simulator state, finite-magnet model failure, an impossible controller lifecycle transition, or an unrecoverable program exception. A `FAULT` truncates the episode and is excluded from policy training data.

## 5. Controller inputs and privilege boundary

The closed-loop controller may continuously read the capsule magnet center, capsule rigid-body pose, linear velocity, angular velocity, contact point and classification, contact force, local surface normal, and the surface mesh or equivalent query structure. These values are permanently authorized for this ideal simulation controller and for evaluation.

The controller must expose to the future Actor only the requested action ID history and the public result. Simulator truth, controller error, surface query, virtual magnet pose, magnetic wrench, contact truth, saturation flags, and validation metrics remain internal telemetry or privileged Critic/evaluation inputs. The design does not modify VLM, Actor, Critic, coverage, reward, or training code.

## 6. Finite-magnet actuation and inverse solution

The pure finite-magnet computation and its authoritative configuration must first be made repository-local. The required source is the external extension documented by `handoffs/reports/OPEN_LOOP_MAGNETIC_CONTROLLER_HANDOFF.md`. The migration must preserve the recorded SHA-256 values, required licenses, and a temporary verified fallback to the legacy location until numerical regression passes. Only the pure magnetic model and configuration needed by TASK-007 are migrated; visualization and XRDF code are retained only if a concrete runtime dependency is demonstrated.

Let the virtual magnet pose be \(\mathbf q_m\in SE(3)\), the measured capsule state be \(\mathbf x_c\), and the finite-model capsule wrench be

\[
\mathbf w_m=f_{\mathrm{mag}}(\mathbf q_m,\mathbf x_c)=[\mathbf F_m,\boldsymbol\tau_m].
\]

The outer loop computes a bounded desired magnetic wrench \(\mathbf w^*\) from the frozen action target and the current capsule error. The inner loop numerically differentiates the same finite-magnet model around the current virtual magnet pose to obtain a 6-by-6 pose-to-wrench Jacobian. It solves a weighted damped least-squares problem with a regularization term toward a nominal capsule-relative magnet pose:

\[
\Delta\boldsymbol\xi=
\arg\min_{\Delta\boldsymbol\xi}
\left\|W\left(J\Delta\boldsymbol\xi-(\mathbf w^*-\mathbf w_m)\right)\right\|^2
+\lambda^2\left\|\Delta\boldsymbol\xi\right\|^2
+\lambda_r^2\left\|\Delta\boldsymbol\xi-\Delta\boldsymbol\xi_{\mathrm{rel}}\right\|^2.
\]

Translation and rotation perturbations use explicit SI units and independent finite-difference steps. The solution has per-update translation, rotation, total separation, orientation, wrench, and condition-number limits. Quaternion updates must be normalized and sign-continuous. The nominal relative pose is recomputed from the current capsule state, so the virtual magnet follows the capsule rather than continuing along an independent world path.

HOLD and VIEW weight orientation error and magnetic torque more strongly than tangential position. MOVE weights signed tangent displacement and tangential magnetic force more strongly while suppressing unnecessary torque. The old coupling ramp, filtering, damping, and force/torque limits are the initial numerical baseline, but controller parameters may be tuned by the authorized simulation calibration loop. Physics parameters may not be randomized or silently changed.

If the numerical inverse is ill-conditioned or a step reaches a trust-region limit, the controller keeps the last finite virtual-magnet target, records `solver_saturated=true`, and tries again at the next 60 Hz update. This is not a `FAULT` unless it leads to non-finite values or a model failure.

## 7. HOLD behavior

At acceptance, HOLD freezes the current camera optical axis, the local inward surface normal, and the capsule center-of-mass tangent-plane anchor. During the full second it magnetically tracks the frozen optical axis first and the tangent anchor second. It does not control rotation about the capsule long axis and does not impose a normal-position target.

HOLD succeeds quantitatively when the terminal optical-axis error is at most 3 degrees, tangent-anchor drift is at most 2 mm, the action remains finite, and the boundary stability gate passes. It remains `COMPLETED` even when one of these validation metrics misses its threshold, unless a true `FAULT` occurs.

## 8. VIEW behavior

At acceptance, VIEW freezes the start camera frame \((\mathbf r_{\mathrm{up}},\mathbf r_{\mathrm{right}},\mathbf u_0)\), local inward surface normal, and tangent anchor. For the selected image direction, the controller forms a unit direction \(\mathbf d_i\) in the start camera image plane and freezes the relative 15-degree optical-axis target:

\[
\mathbf u^*=\cos(15^\circ)\mathbf u_0+\sin(15^\circ)\mathbf d_i.
\]

The first 0.8 seconds follow a quintic zero-end-velocity progress profile from \(\mathbf u_0\) to \(\mathbf u^*\). The final 0.2 seconds hold the target with magnetic feedback. Orientation error is corrected by minimal swing; twist about the optical axis is neither planned nor actively controlled. Tangent-anchor correction is subordinate to the optical-axis objective, and no normal-position target is imposed.

If the camera hemisphere contacts the wall and further target rotation would move the camera end into the wall, the inward component of the target swing is cancelled and the controller holds the current finite orientation for the remaining substeps. The action returns `COMPLETED` with `constrained=true`. A later request that is not geometrically blocked is evaluated normally; no permanent blocked state is created.

An unblocked VIEW sample passes quantitative acceptance when the terminal optical-axis target error is at most 3 degrees, tangent-anchor drift is at most 2 mm, and the boundary stability gate passes. Constrained samples are reported separately and are not used to fill the required unblocked-success quota.

## 9. MOVE behavior

MOVE eligibility is evaluated once at action acceptance. The unsigned capsule long-axis tilt relative to the local inward surface normal is

\[
\theta=\arccos\left(\left|\mathbf u_{\mathrm{axis}}\cdot\mathbf n_{\mathrm{in}}\right|\right).
\]

MOVE is eligible only when \(\theta\ge45^\circ\) and a cylinder-sidewall contact has been observed within the preceding 0.05 seconds, equal to 12 physics substeps. No minimum contact-force magnitude, force-duration dwell, multi-point contact, velocity dwell, or additional stability predicate is permitted. If either prerequisite is absent, the controller freezes a HOLD target, executes HOLD for the full second, and returns `REJECTED`.

For an eligible MOVE, the local normal and signed direction are computed once and frozen for the full second:

\[
\mathbf h=\operatorname{normalize}\left(\mathbf u_{\mathrm{axis}}-(\mathbf u_{\mathrm{axis}}\cdot\mathbf n)\mathbf n\right),
\]

\[
\mathbf d_{\pm}=\pm\operatorname{normalize}(\mathbf n\times\mathbf h).
\]

The target is a signed 5 mm tangent displacement. The first 0.8 seconds use a quintic displacement reference; the final 0.2 seconds stabilize the endpoint. The controller does not parallel-transport or recompute the frozen direction, does not actively hold capsule tilt, and does not command roll about the long axis. Rolling must arise passively from the finite magnetic gradient, translation, and contact friction as in the old controller. Passive roll sign and accumulated angle are mandatory telemetry, but the blocking displacement gate is 4 to 6 mm in the commanded signed direction.

The virtual magnet's capsule-relative regularization remains active throughout MOVE. With no obstruction, the magnet follows the capsule without a large relative-position error. With ordinary obstruction, the same bounded displacement feedback naturally requests more tangential magnetic force, but no obstacle state machine, rerouting, normal unloading, explicit barrier-crossing mode, active tilt change, or independent magnet world trajectory is added. Remaining blockage is a finite low-effect result and not a `FAULT`.

## 10. Parameter authority and tuning rules

TASK-007 uses one versioned controller profile and one versioned physics/profile snapshot. The final flat and stomach acceptance runs must record their SHA-256 digests. The same final controller profile, code, action definitions, and physics parameters must be used in both scenes. Scene-specific controller branches, gains, force limits, timing, contact rules, or thresholds are forbidden.

Controller gains, inverse-solver weights, damping, trust-region limits, virtual-magnet nominal relative pose, magnetic force/torque limits, and filters may be tuned through repeatable simulation. Capsule mass, inertia, gravity, geometry, contact materials, stomach geometry, PhysX solver, and magnetic material constants remain fixed. V1 does not randomize these physics parameters.

Calibration failures are experimental evidence, not an immediate task failure. The executor must diagnose and continue tuning within the authorized design. A sample used to change a parameter is no longer held out and must be replaced. `needs_decision` is returned only after the authorized search and diagnosis are exhausted and a reproducible structural failure remains.

## 11. Quantitative acceptance sequence

Flat no-disturbance validation is the first blocking gate. Each of the eleven actions receives 20 independent held-out valid initial states and needs at least 16 passing samples. Invalid MOVE starts are a separate 20-sample class per sign and must return `REJECTED` for the full one-second boundary. Randomization covers pose, contact state, and preceding action history while keeping physics parameters fixed.

Only after the no-disturbance gate passes does flat disturbance validation begin. Each action again receives 20 independent held-out samples and needs at least 16 passes. Disturbances are applied only in the flat scene and consist of randomized initial state errors plus finite mid-action external force and torque impulses unknown to the controller. A paired feedback-disabled run from the same state and seed must show that the closed-loop virtual-magnet command changes after the disturbance and recovers better than the open-loop baseline; this is required evidence that the implementation is actually closed loop.

Only after both flat gates pass may stomach development begin. Some named stomach regions may be used to tune the single shared profile. Final stomach acceptance uses new held-out regions, 20 samples per action, and at least 16 passes per action. The stomach run receives no injected disturbance. If stomach tuning changes the shared profile, the complete flat no-disturbance and disturbance gates must be rerun with the new digest before final stomach acceptance.

After per-action acceptance, run one fixed no-reset 100-action sequence in the flat scene and one in the stomach scene. Each sequence must contain every ID, opposite VIEW pairs, consecutive MOVE requests, invalid/rejected MOVE requests, and MOVE-to-VIEW and MOVE-to-HOLD transitions. A finite low-effect action does not stop a sequence; only a true `FAULT` stops it. The flat sequence may contain small disturbances. The stomach sequence must not.

## 12. Keyboard visualization

The keyboard interface is strictly one key press, one one-second action, terminal result, user observation, then the next key press. It does not auto-repeat, queue, or launch scripted sequences. Continuous rendering remains enabled independently of the 1 Hz policy boundary, with a selectable target of 60, 120, or 240 FPS and a default of 120 FPS. Physics remains 240 Hz even if wall-clock rendering cannot sustain the selected target.

The terminal prints action ID/name, lifecycle, public result, constrained/low-effect/saturation flags, substep count, current and target optical axis error, tangent drift or signed displacement, passive roll, capsule speed, virtual-magnet relative pose, actual magnetic wrench, contact classification, and measured rendering FPS. No additional HUD is required.

## 13. Evidence and disposition

The final report must identify branch, base, head, exact commands, hardware/software versions, finite-model dependency manifest, controller and physics digests, calibration ranges and selected values, per-action trial tables, invalid-MOVE rejection tables, disturbance comparisons, 100-action sequence summaries, stomach development and held-out region separation, regression results, warnings, and user-facing visualization commands.

Large JSONL logs, videos, screenshots, datasets, caches, and rendered output remain outside Git. The report records their absolute Linux paths, byte sizes, and SHA-256 values. Compact machine-readable summaries and manifests may be committed.

The final disposition is `complete` only when dependency regression, both flat gates, stomach held-out gates, both 100-action sequences, keyboard visualization, and regression tests finish. It is `partial` when flat acceptance passes but the stomach gate remains below threshold after authorized tuning or a true stomach runtime failure prevents completion. It is `needs_decision` only for a reproducible structural blocker that remains after the authorized tuning and diagnosis have been exhausted.
