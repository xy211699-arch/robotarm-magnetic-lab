# TASK-004 Simulation-First Local Dynamics Primitives Revision

## Authority

This revision supersedes `docs/superpowers/specs/2026-08-18-local-dynamics-primitives-design.md` for unfinished TASK-004 work. The original document and first failed report remain historical evidence.

Linux shall continue `feature/TASK-004-local-dynamics-primitives` from report head `2bce0d2`; it shall preserve the implemented controller, tests, preflight, validator, calibration history, and report.

No external Codex skill, plugin, subagent facility, or orchestration package is required. Linux shall execute the written plan directly.

## Revised Objective

The objective is to make the simulated dynamic capsule reliably perform side-lying to upright, upright to side-lying, upright to a 30-degree tilt, and one complete conical revolution at a 30-degree tilt. Every action must complete in strictly less than 10 seconds of simulated time.

Applied force and torque do not need to match a real capsule, magnetic field, magnetic moment, actuator, or hardware limit. “Dynamic” means only that the capsule remains non-kinematic, commands are forces and torques, PhysX integrates state, and PhysX resolves contact.

Outside normal reset, no code may directly write capsule position, orientation, linear velocity, angular velocity, or transforms.

## Removed Constraints

The former `3e-5 N m` torque ceiling, `1 mg` horizontal-force ceiling, and every limit derived from capsule weight, inertia, gravitational tipping moment, magnetic capability, or physical plausibility are superseded.

Linux may use any finite force and torque needed to complete the actions. Remaining limits exist only to prevent nonfinite state, single-step discontinuity, solver explosion, or visually unusable motion. The report must call the result simulation-only and make no hardware-feasibility claim.

## Constraints That Remain

Gravity, passive authored physics, CCD, contact, 240 Hz physics, 60 Hz environment/render cadence, and 30 Hz capsule-camera updates remain active. The controller may command only the capsule wrench; no robot, ASM, external magnet, magnetic action, ideal-surface action, or legacy bridge is allowed.

All primitives remain closed-loop and may read simulated pose, position, velocity, contact state, and contact points. Pose setters, velocity setters, kinematic switching, teleport, projection, and hidden recovery remain forbidden.

No primitive performs clearance, free-space, swept-volume, obstacle-margin, boundary-margin, nearest-triangle, ray-cast, or collision-avoidance checks. Contact alone is not a failure.

World-coordinate semantics remain frozen. With the camera at local `-Z`, define `u = R(q)[0,0,-1]`, pointing from non-camera end to camera end. Upright aligns `u` with world `+Z`; side-lying is 90 degrees; intermediate/cone tilt is 30 degrees; default azimuth is world `+X`.

## Root Cause Addressed

The first controller applied its anchor force at the center of mass and assumed friction would convert translation into a non-camera-end pivot. The result showed millimetre translation and about one degree of rotation. The direct torque was also capped far below the observed simulated tipping requirement.

This revision treats both limits as controller-design errors. Linux shall not preserve them for physical realism.

## Virtual Non-Camera Endpoint Wrench

The revised controller shall construct a virtual force at the non-camera endpoint and convert it to an equivalent center-of-mass wrench. This is a dynamic wrench, not a kinematic constraint.

For center `p`, velocities `v` and `omega`, half total length `h = 0.0125 m`, and directed axis `u`, compute

```text
r_nc = -h u
p_nc = p + r_nc
v_nc = v + cross(omega, r_nc)
anchor_xy = p_nc_xy at primitive start
F_nc_xy = Kp_anchor (anchor_xy - p_nc_xy) - Kd_anchor v_nc_xy
F_nc_z = -F_pin
F_nc = [F_nc_x, F_nc_y, F_nc_z]
F_total = F_nc + F_com
tau_total = tau_pose + cross(r_nc, F_nc)
```

`F_com` may contain bounded translational damping but no height target or surface projection. `tau_pose` remains directed-axis feedback. The composed total wrench is applied through the existing COM API with `positions=None` and `is_global=True`.

For side-to-upright, endpoint pin force and its equivalent moment explicitly bias the non-camera end toward support while pose torque raises the camera end.

## Shared Simulation Profile

Shared parameters shall be stored in `configs/local_primitives/simulation_profile.json`. Flat and stomach tasks shall embed identical content and SHA-256 through `make_local_primitive_action_cfg()`.

The profile contains pose gains, roll damping, pose-torque limit, endpoint anchor gains, endpoint pin force, total force/torque limits, force/torque slew limits, durations, timeouts, tolerances, and stable-speed limits.

The absolute numerical envelope is

```text
0 < total_force_limit_n <= 5.0 N
0 < total_torque_limit_nm <= 0.02 N m
0 < force_slew_limit_n_per_s <= 50.0 N/s
0 < torque_slew_limit_nm_per_s <= 0.2 N m/s
```

These are numerical guardrails, not physical limits.

## Deterministic Authority Calibration

Linux shall calibrate side-to-upright on the flat task before rerunning all primitives. The first candidate grid is

```text
pose_torque_limit_nm = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
endpoint_pin_force_n = [0.05, 0.10, 0.20, 0.50]
axis_kp_nm_per_rad = pose_torque_limit_nm
axis_kd_nms_per_rad = 0.08 * pose_torque_limit_nm
```

Every candidate starts from the same normal side-lying reset and runs for at most 8 seconds. It passes when upright error is at most 3 degrees, stable hold lasts 0.4 seconds, state stays finite, no 240 Hz center displacement exceeds 5 mm, and no load-bearing camera-hemisphere contact occurs.

Select the passing candidate with lowest torque, then lowest pin force, then lowest completion time. If none passes, automatically try torque `[0.01, 0.02] N m` and pin force `[1.0, 2.0] N`. No user decision is required for this expansion.

The calibration script shall write the selected tracked profile and preserve every attempt under `logs/local_primitives_sim_authority/<timestamp>/`. Once all flat primitives pass, freeze the profile before stomach migration.

## Flat Acceptance

The existing flat task remains the quantitative gate. It shall execute the original four sequences without direct reset to upright or 30 degrees:

```text
SIDE_TO_UPRIGHT
SIDE_TO_UPRIGHT -> UPRIGHT_TO_SIDE
SIDE_TO_UPRIGHT -> UPRIGHT_TO_30_DEG
SIDE_TO_UPRIGHT -> UPRIGHT_TO_30_DEG -> CONE_30_DEG_ONE_REVOLUTION
```

Every primitive must finish before its own timeout and before 10 seconds. Rise must avoid load-bearing camera-hemisphere contact and establish late non-camera support. Cone actual unwrapped coverage must reach at least `2 pi - 10 deg` with tilt RMSE at most 5 degrees.

Unrealistic wrench magnitude, plane contact, and collision are not failures. Nonfinite state, more than 5 mm center displacement in one 240 Hz step, forbidden state writes, wrong support end, target failure, and timeout are failures.

## Stomach Migration

After flat acceptance, Linux shall create `Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0`. It inherits the TASK-003 stomach scene, flip, approximately horizontal reset, dynamic capsule, CCD, contact sensor, timing, and camera.

The stomach task must load the exact flat profile digest. No stomach-only gain, geometry query, local normal, clearance check, collision avoidance, or pose recovery is allowed.

Stomach contact and fold impact are allowed. The user judges visual usability from continuous rendering. A blocked action may time out and must be reported without correction.

## Evidence and Reporting

Linux shall update rather than replace the existing TASK-004 report. The first failed attempt remains. Append revision authority, expanded calibration attempts, selected profile/digest, flat metrics, stomach rendered results, regressions, deviations, and external artifact hashes.

## Forbidden Changes

Do not modify USD/USDZ assets, capsule geometry, camera, mass, inertia, gravity, materials, friction, restitution, solver/CCD settings, TASK-003 placement, robot/magnetic control, previous evidence, VLM/RL code, rewards, or coverage logic.

Do not add state writes, kinematic switching, teleport, projection, clearance logic, collision avoidance, automatic recovery, or stomach-specific adaptation.

## Completion Rule

TASK-004 may be `complete` when all four flat primitives pass, regressions pass, the stomach task uses the identical profile digest, and continuous stomach rendered evidence exists. Subjective stomach usefulness remains the user’s visual decision.
