# Verified project state

This file records only verified state already present in merged repository history. Plans and unverified claims are not implementation status.

## Atomic action layer

Atomic action layer stage 1 is implemented and accepted. The frozen interface contains 11 action IDs, cumulative field and external-magnet commands, action masking, whole-trajectory joint/workspace/XRDF preflight checks, and a non-preemptible state machine. Terminal results are restricted to `DONE` and `HARD_FAILURE`, and the deployable execution interface excludes capsule ground truth.

The independent task `Template-Robotarm-Magnetic-Atomic-Table-Lab-v0` exposes a scalar action at 20 Hz with magnetic force updates at 240 Hz. Its deployable observation contains nine joint positions, nine joint velocities, and the seven-value external-magnet pose.

Stage 1 protocol tests passed 10/10. All 11 action IDs reached `DONE` in the Isaac Lab table validation, with reported durations of 0.95--1.05 s and minimum ASM clearance of about 9.94--10.00 mm. The legacy nine-dimensional table task also passed its smoke regression. Runtime acceleration remains telemetry-only until its threshold and filter are calibrated; planned trajectory acceleration remains hard-limited.

Atomic action layer stage 2 has not started. Its current scope is the 1 Hz SMDP boundary, Actor transition and privileged Critic/evaluation channels, short-action effect measurement, and failure-containment validation. It does not yet authorize Actor training or changes to the stage 1 safety contract.

## Active coordination task

The active coordination task is `TASK-000-git-handoff`. It validates only the Windows-to-Linux GitHub handoff loop and must not change simulation source, assets, training configuration, or runtime behavior.
