# TASK-003: Dynamic Capsule Six-Direction Force Teleoperation

**Status:** Approved for Linux implementation by the user on 2026-08-16.

**Planning branch:** `workflow/TASK-003-dynamic-capsule-force-teleop`

**Required Linux branch:** `feature/TASK-003-dynamic-capsule-force-teleop`

**Base lineage:** `0e3ae452c403b141b20ca1e4700c3d37dc7f2b90` from `origin/feature/TASK-002-ideal-surface-controller`

**Design authority:** `docs/superpowers/specs/2026-08-16-dynamic-capsule-force-teleop-design.md`

**Execution plan:** `docs/superpowers/plans/2026-08-16-dynamic-capsule-force-teleop.md`

**Required report:** `handoffs/reports/TASK-003-dynamic-capsule-force-teleop-report.md`

## Objective

Implement an isolated Isaac Lab stomach task in which the existing capsule is a true non-kinematic dynamic rigid body and a human can hold six keyboard directions to apply a bounded, constant-magnitude world-frame force. PhysX, rather than direct pose assignment or surface projection, must integrate the motion and resolve gravity, friction, damping, inertia, and stomach-wall contact.

The task must provide a continuous external view and 30 Hz capsule-camera view so motion can be judged without the one-frame-per-second sampling artifact of TASK-002.

## Real-Dynamics Requirement

Linux shall treat physical state evolution as the central acceptance requirement. Outside reset, no code in the new task may call a root-pose or root-velocity setter for the capsule. No controller may snap, project, clamp, or recover the capsule pose. The task may apply only the requested keyboard force at the capsule center of mass with zero commanded torque.

The capsule must remain subject to gravity and must react to static stomach contact. A collision or penetration failure must be recorded and reported, not repaired by hidden kinematic logic.

## Frozen Interface

The task ID is `Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0`. It uses one environment, 240 Hz physics, 60 Hz environment and keyboard updates, approximately 60 Hz Kit rendering, and 30 Hz capsule-camera updates.

The action is a normalized world-frame vector `[Fx, Fy, Fz]`. The default magnitude is `0.5 * live_capsule_mass * 9.81 N`. `W/S` select `+X/-X`, `A/D` select `+Y/-Y`, and `Q/E` select `+Z/-Z`. Opposite directions cancel and diagonal input is norm-limited. Key release removes the corresponding force immediately at the next 60 Hz step.

## Authorized Changes

Linux may add the isolated dynamic-force action term, task configuration and registration, held-key adapter, Kit launcher, prerequisite inspector, deterministic validator, focused tests, documentation, and required report. Linux may make the smallest export changes needed to expose these additions.

Linux may enable CCD for the new task through verified task-local configuration or runtime API. Linux may not alter shared USD assets to do so without returning `needs_decision`.

## Forbidden Changes

Linux shall not modify the ideal-surface controller, magnetic executor, legacy bridge, robot or stomach assets, capsule geometry, camera calibration, material coefficients, damping, mass, inertia, velocity limits, maximum depenetration velocity, coverage semantics, VLM code, RL code, rewards, or prior evidence.

Linux shall not add torque keys, pose targets, velocity targets, surface-normal forces, surface constraints, automatic recovery, mesh sealing, deformable tissue, fluid effects, or magnetic actuation in this task.

## Mandatory Preflight Gate

Linux shall verify the live capsule prim, spherocylinder dimensions, mass, inertia, center of mass, non-kinematic state, gravity state, body-level CCD, scene-level CCD, contact sensor, static stomach collider, geometry hash, topology statistics, and frozen timing configuration.

Linux shall also scan the new runtime path for forbidden capsule pose and velocity writes outside the reset path and verify that no magnetic or ideal-surface action term is present.

If any required property is ambiguous or cannot be configured without changing shared assets, Linux shall report `needs_decision` and stop before implementing behavior that could be mistaken for real dynamics.

## Required Acceptance

Linux shall pass pure keyboard and force-contract tests, task registration and isolation tests, preflight, a no-input gravity/contact settling run, deterministic six-direction force runs, a rendered continuous-motion smoke, and all regressions named by the execution plan.

The six-direction evidence shall report the exact requested and applied wrench, pose and velocity evolution, maximum per-physics-step displacement, contact force, and minimum measured surface clearance. Contact-constrained motion need not match free-space displacement, but every state must remain finite and every applied force must match the contract.

The rendered result shall demonstrate 240 Hz physics, 60 Hz environment updates, approximately 60 Hz Kit rendering, and 30 Hz capsule-camera updates. If no human performs the subjective smoothness review, the report shall mark that claim unverified.

## Delivery Contract

The final report shall state `complete`, `partial`, `needs_decision`, or `blocked`; record planning base, implementation head, branch, every command and observed result; distinguish measured facts from interpretations; enumerate deviations and unverified claims; and list external artifacts with absolute paths, byte sizes, and SHA-256 hashes.

Linux shall push `feature/TASK-003-dynamic-capsule-force-teleop` and shall not merge it. Windows remains responsible for reviewing evidence and authorizing any later physics tuning, torque control, magnetic model, or VLM work.
