# Verified project state

This file records only verified state already present in merged repository history. Plans and unverified claims are not implementation status.

## Atomic action layer

Atomic action layer stage 1 is implemented and accepted. The frozen interface contains 11 action IDs, cumulative field and external-magnet commands, action masking, whole-trajectory joint/workspace/XRDF preflight checks, and a non-preemptible state machine. Terminal results are restricted to `DONE` and `HARD_FAILURE`, and the deployable execution interface excludes capsule ground truth.

The independent task `Template-Robotarm-Magnetic-Atomic-Table-Lab-v0` exposes a scalar action at 20 Hz with magnetic force updates at 240 Hz. Its deployable observation contains nine joint positions, nine joint velocities, and the seven-value external-magnet pose.

Stage 1 protocol tests passed 10/10. All 11 action IDs reached `DONE` in the Isaac Lab table validation, with reported durations of 0.95--1.05 s and minimum ASM clearance of about 9.94--10.00 mm. The legacy nine-dimensional table task also passed its smoke regression. Runtime acceleration remains telemetry-only until its threshold and filter are calibrated; planned trajectory acceleration remains hard-limited.

Atomic action layer stage 2 has not started. Its current scope is the 1 Hz SMDP boundary, Actor transition and privileged Critic/evaluation channels, short-action effect measurement, and failure-containment validation. It does not yet authorize Actor training or changes to the stage 1 safety contract.

## Coordination readiness

`TASK-000-git-handoff` is complete. Windows published the contract at commit `3137234ed47d7c15b3d6380b1222acd956849f36`; Linux returned the accepted report at commit `24a1469d3b797d75b4d7e8e3079a69c26e85fc83`. The round trip verified the shared Fork, isolated branches, Linux Deploy Key push access, read-only upstream configuration, and independent Windows verification.

There is currently no active implementation task. Atomic action layer stage 2 remains planned but is not authorized to start until Windows publishes a new approved task contract in `handoffs/active/`.
