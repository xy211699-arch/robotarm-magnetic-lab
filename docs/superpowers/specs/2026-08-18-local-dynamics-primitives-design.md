# Local Dynamic Capsule Primitives Design

## Purpose

TASK-004 adds four closed-loop capsule posture primitives on top of the real rigid-body force/contact environment accepted after TASK-003. It is deliberately narrower than magnetic control, navigation, coverage planning, or VLM control. The immediate objective is to determine whether direct bounded force and torque feedback can produce repeatable local capsule motions without any direct state assignment.

The four primitives are side-lying to upright, upright to side-lying, upright to a 30-degree tilt, and one complete conical revolution at a 30-degree tilt. Every primitive must complete in less than 10 seconds of simulated time.

## User-Approved Test Strategy

The controller is developed and quantitatively accepted in the existing flat-table scene first. Only after the flat gate passes is the same controller configuration instantiated in the TASK-003 stomach scene. The stomach stage is a rendered visual migration test, not a second geometry-aware controller-development stage.

The stomach reset already lies in the user-selected approximately horizontal region. TASK-004 shall not search for another patch, estimate a local surface normal, align a local frame, or change the initial capsule pose beyond the existing reset event.

No primitive performs swept-volume, free-space, boundary-margin, nearest-triangle, ray-cast, or collision-clearance checks. Physical contact is resolved by PhysX. Contact with the stomach or table is not itself a failure and never causes pose correction. A collision may nevertheless prevent the requested posture from being reached; in that case the primitive reports its ordinary target-tracking timeout.

## Confirmed Base Facts

TASK-004 is based on `06b15caf9a69bc9c20f85522ce4abbb32c8b9245`, the current head of `origin/feature/TASK-003-dynamic-capsule-force-teleop`. TASK-003 established a non-kinematic capsule, gravity, CPU PhysX with scene and body CCD, 240 Hz physics, 60 Hz environment/render cadence, 30 Hz capsule-camera updates, a contact sensor, and a center-of-mass wrench API.

The live capsule measured by TASK-003 has mass `0.0057349997 kg`, radius `0.0065 m`, cylindrical height `0.012 m`, total length `0.025 m`, and principal inertia on the order of `10^-7 kg m^2`. Its local `Z` axis is the long axis. The camera optical center is attached to the local `-Z` end and looks outward through that cap; local `+Z` is therefore the non-camera end.

TASK-003's report was `partial` because its frozen stomach collision validation measured penetration and a small continuity-bound exceedance. The user separately accepted TASK-003's force/contact behavior as sufficient to start local primitive work. TASK-004 must not rewrite that report or claim that its prior penetration evidence disappeared.

## Coordinate and Endpoint Contract

Let `R(q)` rotate a local vector into the world frame. Define the directed capsule axis

```text
u = R(q) [0, 0, -1]^T
```

so `u` points from the non-camera end toward the camera end. World vertical is

```text
ez = [0, 0, 1]^T.
```

The posture angle is

```text
theta = acos(clamp(dot(u, ez), -1, 1)).
```

Upright means `theta = 0 deg`, with the camera end above the non-camera end. Side-lying means `theta = 90 deg`. The tilted posture means `theta = 30 deg`. All meanings remain fixed in the world frame in both scenes; TASK-004 never substitutes a stomach-local normal for `ez`.

For capsule center `p` and half total length `h = 0.0125 m`, the endpoint poles are

```text
p_non_camera = p - h u
p_camera     = p + h u.
```

The side-to-upright primitive must not pivot on the camera hemisphere. In flat acceptance this is verified from read-only PhysX contact points and the analytical endpoint locations. It is not implemented as a controller-side collision rule. In the stomach visual stage the same torque and non-camera anchor behavior is retained, but the endpoint-contact claim is visually reviewed rather than guaranteed by a mesh-aware controller.

## Primitive Interface

The only new action term is `LocalPrimitiveAction`. Its four-float command is

```text
[start_pulse, primitive_code, direction_x, direction_y]
```

`start_pulse > 0.5` on a rising edge requests one primitive. `primitive_code` is rounded to one of `0`, `1`, `2`, or `3`. The direction is normalized in the world XY plane; a near-zero direction deterministically defaults to world `+X`. The action term latches a request and continues feedback at every 240 Hz physics substep while later environment steps send `start_pulse = 0`.

The primitive codes are frozen as follows:

```text
0 = SIDE_TO_UPRIGHT
1 = UPRIGHT_TO_SIDE
2 = UPRIGHT_TO_30_DEG
3 = CONE_30_DEG_ONE_REVOLUTION
```

A command may start when the controller is idle or holding the previous successful target. A command received while another primitive is running is rejected as busy and does not restart the timer. Invalid primitive codes and invalid start postures are reported without changing the capsule state.

## Closed-Loop Wrench Controller

The action term reads the simulated center-of-mass pose and velocity as feedback. It may use all simulator ground truth because TASK-004 is a dynamics-controller bring-up task, not a deployable visual estimator. Simulator state is read-only outside the existing reset event.

For desired directed axis `ud` and desired minimal angular velocity `omega_d`, the torque command is

```text
e_axis = cross(u, ud)
omega_perp = omega - dot(omega, u) u
tau = Kp_axis e_axis
    + Kd_axis (omega_d - omega_perp)
    - Kd_roll dot(omega, u) u.
```

The torque norm is saturated at `2.0e-5 N m`. The initial gains are `Kp_axis = 1.2e-5 N m/rad`, `Kd_axis = 2.0e-6 N m s/rad`, and `Kd_roll = 1.0e-6 N m s/rad`.

At primitive start the controller records the world-XY position of the non-camera pole:

```text
anchor_xy = p_xy - h u_xy.
```

It then generates the desired center position and horizontal velocity

```text
p_xy_d = anchor_xy + h ud_xy
v_xy_d = h d(ud_xy)/dt
```

and applies

```text
F_xy = Kp_xy (p_xy_d - p_xy) + Kd_xy (v_xy_d - v_xy).
```

The initial gains are `Kp_xy = 0.8 N/m` and `Kd_xy = 0.03 N s/m`. The horizontal force norm is limited to `0.5 m g`. The vertical command is a constant world-down preload `Fz = -0.15 m g`; it is not a height controller. Gravity and PhysX contact determine vertical position.

Force and torque are applied at the live center of mass with `positions=None` and `is_global=True`. No robot command, magnetic action, magnetic collision bridge, ideal-surface action, pose target, velocity target, surface projection, or direct capsule state write is present.

## Trajectories and Time Limits

Every posture transition uses a unit-vector trajectory with a quintic time-scaling function

```text
s(r) = 10 r^3 - 15 r^4 + 6 r^5,  r in [0, 1].
```

Side-to-upright uses shortest-arc spherical interpolation from the measured starting axis to world `+Z`, with `5.5 s` motion time and an `8.0 s` hard timeout. Upright-to-side uses the requested azimuth at `90 deg`, with `4.5 s` motion time and a `7.0 s` hard timeout. Upright-to-30-degree uses the requested azimuth at `30 deg`, with `3.5 s` motion time and a `6.0 s` hard timeout.

The cone primitive starts at the measured 30-degree azimuth and commands

```text
ud(t) = [sin(30 deg) cos(phi(t)),
         sin(30 deg) sin(phi(t)),
         cos(30 deg)]
phi(t) = phi_start + 2 pi s(t / 8.0 s).
```

Its motion time is `8.0 s`, its final stabilization allowance is `1.0 s`, and its hard timeout is `9.5 s`. Both target phase and actual unwrapped capsule azimuth are recorded so returning to the starting orientation cannot be misclassified as a full revolution.

The start-posture gates are `75 deg <= theta <= 105 deg` for side-to-upright, `theta <= 5 deg` for both upright-origin primitives, and `abs(theta - 30 deg) <= 3 deg` for the cone. These are command-validity gates, not spatial-clearance gates.

## Completion and Holding

A transition succeeds only after its motion profile has finished and the target error remains within tolerance for `0.4 s`. Upright and side tolerances are `3 deg`; the 30-degree tolerance is `2 deg`. During the stable window, center-of-mass linear speed must remain below `0.02 m/s` and angular speed below `0.15 rad/s`.

The cone additionally requires actual unwrapped azimuth coverage of at least `2 pi - 10 deg` and tilt root-mean-square error no greater than `5 deg` over the revolution. A completed primitive enters a holding state that keeps the final closed-loop target active until the next primitive starts. Completion time stops when the stable window is satisfied. A timed-out or nonfinite primitive clears active force and torque and reports failure; it never repairs the pose.

## Flat Quantitative Gate

The flat task is `Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0`. It reuses the delivered table scene and dynamic capsule but replaces all actions with the single local-primitive action. The robot and magnetic assembly may remain visually present in the inherited scene, but TASK-004 sends no joint or magnetic command and instantiates no magnetic executor.

The flat validator executes four independent sequences from the normal side-lying reset:

```text
SIDE_TO_UPRIGHT
SIDE_TO_UPRIGHT -> UPRIGHT_TO_SIDE
SIDE_TO_UPRIGHT -> UPRIGHT_TO_30_DEG
SIDE_TO_UPRIGHT -> UPRIGHT_TO_30_DEG -> CONE_30_DEG_ONE_REVOLUTION
```

This sequencing avoids direct resets into upright or tilted states. Direct pose and velocity writes remain authorized only inside the standard episode reset before the first physics step.

The flat gate requires every primitive to succeed before its own hard timeout, finite states and wrenches, bounded wrench norms, no forbidden state setter, correct final posture, stable holding, and a complete cone azimuth. For side-to-upright, no load-bearing contact may occur on the camera hemisphere, and the dominant late-transition support contact must lie on the non-camera hemisphere. Contact-point classification uses axial coordinate `sigma = dot(contact_point - p, u)`: the camera hemisphere begins at `sigma > +0.006 m`; the non-camera hemisphere begins at `sigma < -0.006 m`.

No surface clearance or obstacle margin is measured. Contact with the flat plane is expected and required, not avoided.

## Stomach Unchanged-Controller Migration

The stomach task is `Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0`. It inherits the TASK-003 stomach scene, flip, approximately horizontal reset region, dynamic-body settings, CCD, timing, contact sensor, camera cadence, and CPU PhysX selection.

Both task configurations must obtain their `LocalPrimitiveActionTermCfg` from the same `make_local_primitive_action_cfg()` factory. A test compares every dataclass/config field and fails if the stomach task overrides gains, limits, durations, tolerances, coordinate semantics, force composition, or controller type.

The stomach implementation shall contain no branch based on task ID, stomach asset, contact normal, mesh query, wall distance, or local surface frame. It performs the same four sequences with the same controller and parameters. The user judges rendered behavior. If it is visually satisfactory, TASK-004 is considered usable at that initial approximately horizontal region without a stomach-specific adaptation task. That result does not establish robustness on arbitrary slopes, folds, curvatures, or initial locations.

## Rendering and Evidence

Both tasks retain 240 Hz physics, 60 Hz environment and render interval, and 30 Hz capsule-camera updates. The launcher advances `env.step()` continuously even while no new primitive is requested. It provides an external Kit view, the existing capsule-camera view, keyboard commands `1` through `4`, reset, snapshot, and exit, plus deterministic scripted sequences.

Flat validation writes JSON and JSONL evidence outside Git under `logs/local_primitives_flat/<timestamp>/`. The visual launcher writes under `logs/local_primitives_visual/<timestamp>/`. Each sample records simulated time, primitive state, target and actual axis, angle error, unwrapped phase, center pose and velocity, requested and applied force/torque, contact force, saturation flags, and completion reason. Rendered video or screenshots remain outside Git and are reported with absolute path, byte size, and SHA-256.

## Authorized Calibration

Linux may tune only the controller gains, wrench limits, preload ratio, and motion durations in the shared flat-and-stomach configuration after flat measurements. Gain changes must remain within the ranges `Kp_axis in [2e-6, 3e-5] N m/rad`, `Kd_axis in [2e-7, 8e-6] N m s/rad`, `Kd_roll in [1e-7, 5e-6] N m s/rad`, `Kp_xy in [0.1, 3.0] N/m`, `Kd_xy in [0.005, 0.15] N s/m`, horizontal limit in `[0.1, 1.0] m g`, downward preload in `[0.0, 0.5] m g`, and every hard timeout below `10.0 s`.

Every calibration attempt and final value must be reported. Once the flat gate passes, the shared controller configuration is frozen before the stomach run. No stomach-only tuning is authorized in TASK-004.

## Forbidden Changes

TASK-004 shall not modify USD assets, capsule geometry, camera calibration, mass, inertia, gravity, friction, restitution, damping, CCD settings, solver parameters, stomach pose, TASK-003 reset placement, robot/magnet control, previous tasks, previous reports, VLM code, RL code, rewards, or coverage logic.

TASK-004 shall not add a surface normal estimator, local tangent frame, mesh projection, clearance probe, swept-volume test, collision avoidance, automatic recovery, penetration repair, teleport, or runtime root-state setter.

## Decision and Failure Rules

Linux returns `needs_decision` before behavior work if the live camera/non-camera endpoint convention cannot be confirmed, if direct torque cannot be applied through the installed center-of-mass wrench API, if read-only flat contact points cannot be obtained for the endpoint acceptance, or if the new task cannot be isolated from magnetic and joint actions without changing shared assets.

Linux returns `partial` if the controller and flat task run but one or more primitives miss their quantitative target, exceed a timeout, use a camera-hemisphere support contact during rise, or become nonfinite. It shall report the observed trajectory and saturation rather than changing physics or adding hidden pose correction.

Linux may report implementation `complete` when the flat quantitative gate and regressions pass and the unchanged stomach task produces the required continuous rendered evidence. Subjective stomach usability remains a user acceptance decision and may be recorded as unverified until the user reviews the visualization.
