# TASK-007 Virtual-Magnet Eleven-Action Closed-Loop Controller

## Execution authority

Windows design branch: `workflow/TASK-007-virtual-magnet-closed-loop`

Linux implementation branch: `feature/TASK-007-virtual-magnet-closed-loop`

Exact implementation base: `bff26174ebd1aff2800883a0afdd5295f4f222d1`

Authoritative design: `docs/superpowers/specs/2026-08-20-virtual-magnet-eleven-action-closed-loop-design.md`

Authoritative implementation plan: `docs/superpowers/plans/2026-08-20-virtual-magnet-eleven-action-closed-loop.md`

Required return report: `handoffs/reports/TASK-007-virtual-magnet-eleven-action-closed-loop-report.md`

## Linux execution instruction

Fetch `origin/workflow/TASK-007-virtual-magnet-closed-loop`, create `feature/TASK-007-virtual-magnet-closed-loop` from the fetched planning head, verify that the planning commit's parent is the exact base above, and read both authoritative documents completely before editing. Execute the implementation plan manually in order. The Linux session is not required to install or invoke any `superpowers` skill; the directory name is historical only.

Do not merge TASK-005, TASK-006, or `main`. The controller must be built from the validated open-loop magnetic-controller baseline while preserving that baseline and its evidence for regression.

## Frozen research boundary

The capsule remains a gravity-enabled, non-kinematic dynamic rigid body. It may move only under the repository-local finite-magnet model's force and torque plus PhysX gravity, contact, friction, and damping. Do not write capsule pose or velocity during action execution, lock degrees of freedom, teleport, project, or inject an arbitrary desired wrench.

The mechanical arm and Ball joints are removed from the TASK-007 actuation path. A non-colliding virtual external magnet supplies a finite 6-DOF analytical pose and may have a debug Xform, but it is not a rigid body. Its pose is updated by closed-loop capsule truth feedback and remains regularized to a nominal capsule-relative pose so it follows the capsule rather than drifting along an independent world trajectory.

This is a simulation-privileged ideal controller. Capsule truth may be used internally but may not enter the future Actor observation. No VLM, Actor, Critic, reward, coverage, action-chunk, teacher, or dataset implementation is authorized.

## Frozen action contract

The public action is one scalar ID: `0 HOLD_VIEW`, `1 VIEW_UP`, `2 VIEW_UP_RIGHT`, `3 VIEW_RIGHT`, `4 VIEW_DOWN_RIGHT`, `5 VIEW_DOWN`, `6 VIEW_DOWN_LEFT`, `7 VIEW_LEFT`, `8 VIEW_UP_LEFT`, `9 MOVE_SIDE_POS`, and `10 MOVE_SIDE_NEG`. The only public results are `COMPLETED`, `REJECTED`, and `FAULT`. An active-state request is discarded without queueing.

Every action occupies exactly 1.000 seconds or 240 physics substeps. Magnetic-wrench evaluation and interpolation run at 240 Hz; feedback targets update at 60 Hz; RGB/action boundaries remain 1 Hz. VIEW and MOVE use 0.8 seconds of quintic motion plus 0.2 seconds of active magnetic stabilization. The last 0.1-second mean speeds must be at most 2 mm/s linear and 0.1 rad/s angular for quantitative pass.

HOLD freezes and magnetically tracks the starting optical axis, local inward normal, and center-of-mass tangent anchor. VIEW freezes the start camera frame and commands the selected image direction at a relative 15-degree cone half-angle, using minimal swing without active twist. Camera-end wall contact cancels only further inward swing, completes normally, and sets `constrained=true`.

MOVE is eligible only when unsigned long-axis tilt relative to the inward normal is at least 45 degrees and cylinder-sidewall contact occurred within the previous 0.05 seconds. No force threshold, stability dwell, or multi-point-contact gate may be added. An ineligible MOVE executes one second of HOLD and returns `REJECTED`.

An eligible MOVE freezes the local tangent direction `±normalize(n × h)` and commands a signed 5 mm displacement. Acceptance is 4 to 6 mm. Direction is not recomputed or parallel-transported. Tilt and long-axis roll are not actively controlled; rolling must arise passively from finite magnetic-gradient translation and contact friction. Do not add obstacle crossing, rerouting, normal unloading, active tilt, or an independent magnet world trajectory.

## Tuning and acceptance authority

Controller gains, inverse-solver weights/damping/limits, nominal magnet-relative pose, filters, and magnetic wrench limits may be tuned. Mass, inertia, geometry, gravity, contact materials, magnetic material constants, stomach geometry, and PhysX are fixed and not randomized. One final controller profile must serve both scenes; stomach-only code branches or gains are forbidden.

Calibration failures are evidence and must trigger diagnosis and further authorized tuning, not an immediate failure return. If a validation sample informs tuning, discard it and generate a new held-out sample. Use `needs_decision` only after the permitted search and diagnosis are exhausted and a reproducible structural blocker remains.

The gate order is mandatory. First, run flat no-disturbance validation with 20 held-out samples per action and at least 16 passes per action, plus separate invalid-MOVE samples. Second, run flat disturbed validation with 20 held-out samples per action, at least 16 passes, and paired feedback-disabled evidence proving closed-loop correction. Third, use named stomach development regions if tuning is needed and then validate new held-out regions with 20 samples per action and at least 16 passes. Any stomach-driven profile change invalidates and requires rerunning both flat gates.

Finally run one no-reset 100-action sequence in flat and one in stomach, using the same ID file and covering all actions, opposite VIEWs, consecutive MOVE, rejected MOVE, MOVE-to-VIEW, and MOVE-to-HOLD. Flat may include small disturbances; stomach may not. Only true `FAULT` stops a sequence.

## Return contract

Push `feature/TASK-007-virtual-magnet-closed-loop` without merging. The final response must include the implementation HEAD, disposition, report path, finite-model manifest and hashes, controller/physics profile digests, per-action flat and stomach success counts, paired disturbance evidence, both 100-action sequence outcomes, regression counts, exact visualization commands, and external evidence paths with byte sizes and SHA-256 values.

Return `complete` only when every gate is complete. Return `partial` when both flat gates pass but stomach remains below threshold after authorized tuning or a true stomach runtime failure prevents completion. Return `needs_decision` only for an exhausted reproducible structural blocker. Ordinary low-effect motion, wall obstruction, finite oscillation, or solver saturation is not a system `FAULT`.
