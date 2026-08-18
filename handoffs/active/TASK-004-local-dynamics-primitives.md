# TASK-004: Closed-Loop Local Dynamic Capsule Primitives

**Status:** Approved for Linux implementation by the user on 2026-08-18.

**Planning branch:** `workflow/TASK-004-local-dynamics-primitives`

**Required Linux branch:** `feature/TASK-004-local-dynamics-primitives`

**Base lineage:** `06b15caf9a69bc9c20f85522ce4abbb32c8b9245` from `origin/feature/TASK-003-dynamic-capsule-force-teleop`

**Design authority:** `docs/superpowers/specs/2026-08-18-local-dynamics-primitives-design.md`

**Execution plan:** `docs/superpowers/plans/2026-08-18-local-dynamics-primitives.md`

**Required report:** `handoffs/reports/TASK-004-local-dynamics-primitives-report.md`

## Objective

Implement four closed-loop local capsule primitives using only bounded force and torque applied directly to the existing non-kinematic dynamic capsule: side-lying to upright, upright to side-lying, upright to 30 degrees, and one full conical revolution at 30 degrees. Each primitive must complete in less than 10 seconds of simulated time.

## Test Order

Linux shall implement and quantitatively accept the controller in the flat-table environment first. Only after that gate passes shall Linux instantiate the same controller and the same parameter set in the existing TASK-003 stomach scene for rendered visual review.

The stomach reset is already the selected approximately horizontal region. Do not search for another region, estimate a surface normal, or reorient the capsule before the action sequence.

## Frozen Physical-Control Boundary

Outside the standard episode reset, capsule motion may arise only from gravity, contact, passive authored physics, and the TASK-004 center-of-mass force/torque command. No root pose, root velocity, transform, or velocity setter may occur in the action term, controller, launcher, validator, contact response, or failure path.

Do not call the robot, ASM, external magnet, magnetic action, ideal-surface action, or legacy collision bridge. Do not add pose projection, surface following, hidden recovery, or automatic penetration correction.

## Frozen Coordinate Meaning

The camera is on local `-Z`. The directed capsule axis points from the non-camera end to the camera end and is `u = R(q)[0,0,-1]`. Upright means `u` aligned with world `+Z`; side-lying means a 90-degree world tilt; the intermediate and cone angle is 30 degrees from world `+Z`. The default arbitrary direction is world `+X`.

The side-to-upright primitive shall favor the non-camera end as its pivot through the shared horizontal anchor controller. Flat acceptance must prove that the camera hemisphere is not selected as the load-bearing support; the controller may not enforce this by teleporting or collision filtering.

## No Spatial-Margin Logic

No action performs space-availability, sweep, obstacle-margin, boundary-margin, clearance, ray-cast, nearest-triangle, or mesh-distance checks. Contact is allowed. A collision does not independently fail or abort an action. If contact blocks the requested motion, the ordinary posture/time criteria determine the result.

## Shared-Controller Requirement

The flat and stomach task configurations shall call the same `make_local_primitive_action_cfg()` factory. The stomach task may replace only scene/reset/viewer details inherited from TASK-003. It may not override controller gains, limits, timing, tolerances, coordinate semantics, trajectory generation, or force composition.

The controller shall use world vertical, not a measured stomach normal. No task-ID branch, stomach-local frame, surface query, or stomach-specific tuning is allowed.

## Flat Acceptance

The flat validator shall execute the four required sequences from the normal side-lying reset, using completed primitives to create upright and 30-degree prerequisites rather than directly writing those states. It shall verify command latching, 240 Hz feedback, bounded force/torque, finite state, target angle, stable hold, per-primitive completion time, actual cone azimuth coverage, and non-camera support during the rise.

Every hard timeout must be strictly less than 10 seconds. Contact with the plane is expected and is not a failure.

## Stomach Visual Acceptance

After flat acceptance, Linux shall run the same sequences in `Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0` with continuous external rendering and the 30 Hz capsule-camera view. Linux shall not add an automated stomach clearance or geometry gate. The report records quantitative tracking telemetry, artifacts, collisions, saturation, and timeouts, while the user decides whether the visible result is usable.

A satisfactory run establishes only that the unchanged controller is usable at the current approximately horizontal initial region. It does not establish arbitrary-wall or arbitrary-fold robustness.

## Authorized Changes

Linux may add focused pure controller modules, one center-of-mass force/torque action term, isolated flat and stomach task configurations and registrations, pure and live tests, a flat validator, a shared rendered launcher, operator documentation, and the required report. Linux may make the smallest export changes needed to expose those additions.

Linux may tune only shared controller gains, wrench limits, preload, and sub-10-second trajectory times within the explicit ranges in the design authority. Every attempt and final value must be reported. Once flat acceptance passes, parameters are frozen before the stomach run.

## Forbidden Changes

Linux shall not modify any USD/USDZ asset, capsule geometry, camera, physical material, mass, inertia, gravity, damping, restitution, CCD or solver configuration, stomach placement, TASK-003 placement, previous controller, previous report, VLM/RL code, rewards, or coverage logic.

Linux shall not add a local normal estimator, surface mesh controller, clearance probe, swept-volume test, collision avoidance, magnetic actuation, direct state setter, or stomach-only adaptation.

## Delivery Contract

Linux shall push `feature/TASK-004-local-dynamics-primitives` without merging. The report shall include planning base, implementation head, branch, exact commands and observed results, final shared controller parameters, all calibration attempts, flat primitive metrics, stomach rendered artifacts, deviations, unverified claims, and external artifact paths with byte sizes and SHA-256 hashes.
