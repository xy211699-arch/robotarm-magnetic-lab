# TASK-004: Simulation-First Closed-Loop Local Capsule Primitives

**Status:** Revised implementation authorized by the user on 2026-08-18 after review of the first flat failure.

**Planning branch:** `workflow/TASK-004-local-dynamics-primitives`

**Linux continuation branch:** `feature/TASK-004-local-dynamics-primitives`

**Required continuation head:** `2bce0d2` or its exact descendant containing the existing TASK-004 report

**Superseding design:** `docs/superpowers/specs/2026-08-18-local-dynamics-primitives-simulation-first-revision.md`

**Superseding plan:** `docs/superpowers/plans/2026-08-18-local-dynamics-primitives-simulation-first-revision.md`

**Report to update:** `handoffs/reports/TASK-004-local-dynamics-primitives-report.md`

## Revision Decision

The user requires the four target motions in simulation and does not require realistic force, torque, magnetic-field, magnetic-moment, or hardware limits. The first TASK-004 report is an accepted failed attempt, not the end of the task.

Linux shall preserve the implementation and evidence and continue the existing feature branch. Do not recreate the branch from the planning branch and do not delete the original `partial` report content.

## Required Motion Mechanism

The capsule remains a non-kinematic dynamic rigid body. Outside normal reset, every motion must result from force and torque applied to the capsule and integrated by PhysX. Direct pose, orientation, transform, linear-velocity, and angular-velocity writes remain forbidden.

Linux may use any finite force and torque needed to complete the actions. Former `mg`-based limits and the former `3e-5 N m` torque ceiling are superseded. Magnitudes are judged only against the revised numerical envelope and continuity checks, not physical realism.

## Revised Controller

Replace the former center-of-mass-only anchor assumption with a virtual non-camera endpoint force converted to an equivalent center-of-mass wrench. Combine endpoint force, equivalent endpoint torque, and closed-loop pose torque, then apply the total through the existing COM wrench API.

The controller shall keep the non-camera endpoint anchored in world XY and apply a configurable world-down endpoint pin force. This explicitly biases side-to-upright toward non-camera support without a kinematic constraint or state write.

## Numerical Envelope

Total force may be any finite value up to `5.0 N`. Total torque may be any finite value up to `0.02 N m`. Force slew may be up to `50.0 N/s`; torque slew may be up to `0.2 N m/s`. These are solver/continuity guardrails, not physical limits.

Do not stop or request a decision because a passing candidate is physically unrealistic. Describe the controller as simulation-only.

## Calibration and Flat Gate

Run the deterministic side-to-upright authority grids in the superseding design. Automatically expand to the second grid if necessary, select the lowest-authority passing candidate, and store the shared profile in `configs/local_primitives/simulation_profile.json` with SHA-256.

Rerun all four flat sequences with one unchanged profile. Each action must finish in strictly less than 10 seconds. Rise must avoid load-bearing camera-hemisphere contact; cone actual unwrapped coverage must pass. Contact magnitude and physical implausibility are not failures.

## Stomach Migration

After the flat gate passes, create the stomach task and continuous launcher. Inherit the TASK-003 scene, approximately horizontal reset, dynamic settings, contact, CCD, timing, and camera while loading the exact flat profile digest.

No stomach-specific gain, geometry query, local normal, clearance check, collision avoidance, or pose recovery is allowed. Stomach contact is allowed. The user judges the visual result.

## Unchanged Prohibitions

Do not modify USD/USDZ assets, capsule geometry, camera, mass, inertia, gravity, materials, friction, restitution, solver/CCD settings, TASK-003 placement, robot/magnetic control, VLM/RL code, rewards, coverage logic, or prior evidence.

Do not add a pose setter, velocity setter, teleport, kinematic switch, surface projection, space-margin check, automatic recovery, or hidden correction.

## Execution Environment

The superseding plan is self-contained. Linux shall execute its written tasks directly. No external Codex skill, plugin, subagent, or orchestration command is required.

## Delivery

Update the existing TASK-004 report with the revision, every calibration attempt, selected profile/digest, flat results, stomach rendered evidence, regressions, deviations, unverified visual claims, and external artifacts with sizes and SHA-256 hashes.

Push the existing `feature/TASK-004-local-dynamics-primitives` branch without merging.
