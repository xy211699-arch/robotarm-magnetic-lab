# Stomach contact-physics acceptance report

Date: 2026-07-29  
Task: `Template-Robotarm-Magnetic-Stomach-Lab-v0`  
Physics/control rate: 240 Hz / 20 Hz  
Capsule mass read from PhysX: 5.735 g

## Outcome

All seven automated scenarios passed. The result establishes that the current
rigid stomach mesh produces consistent finite contact responses for the tested
local region. It does **not** establish biological fidelity; wet-tissue
friction, compliance, damping and deformation remain provisional parameters
that require physical calibration.

| Test | Result | Key measured result |
|---|---:|---|
| No-magnet resting | PASS | 100% contact in final 3 s; support 0.05468 N; drift 0.259 mm |
| Drops from 5/15/30 mm | PASS | all contact and settle; substep peaks 0.323–0.341 N; no tunneling |
| Three inclined regions | PASS | 6.1–24.8 mm planar slide; all finish in bounded contact |
| Magnetic wall attraction | PASS | 5.0 mN magnetic force; contact after 0.15 s; support rises to 0.06020 N |
| Wall rolling and turning | PASS | 12.18 mm travel; 107.6° axis change; contact ratio 100% |
| Ball pose while contacting | PASS | Ball delta 1.151 rad; capsule peak 0.780 rad/s; contact ratio 100% |
| Five initial stomach points | PASS | all points contact and retain contact; all states finite |

## What is recorded

Every 20 Hz row includes capsule pose, linear/angular velocity, local axis,
Ball position/action, normal contact force, 240 Hz history peak, contact/air
time, estimated tangential force, magnetic force/torque, and termination flags.
Raw rows are in `telemetry.jsonl`; objective checks and metrics are in
`summary.json`.

Latest accepted summaries:

- `logs/contact_physics/resting/20260729_102839/summary.json`
- `logs/contact_physics/drop/20260729_104239/summary.json`
- `logs/contact_physics/incline_slide/20260729_103154/summary.json`
- `logs/contact_physics/magnetic_attraction/20260729_104114/summary.json`
- `logs/contact_physics/wall_roll_turn/20260729_103543/summary.json`
- `logs/contact_physics/ball_pose_contact/20260729_103732/summary.json`
- `logs/contact_physics/multi_start/20260729_103921/summary.json`

## Visual acceptance still required

Run each script without `--visualizer none`. Confirm that contact markers occur
at the stomach wall, no frame shows tunneling or an explosive launch, rolling
looks continuous rather than teleporting, and the capsule stays passive while
only the Ball joints are commanded. The camera window is optional via
`--capsule_camera_view`.

## Remaining limitations

- The stomach is a static exact triangle mesh, not deformable tissue.
- Static/dynamic friction (0.20/0.15) and compliant-contact parameters are
  engineering estimates, not measured wet-tissue values.
- The tested initial points are near the validated lower-wall operating region,
  not a whole-volume sampling of the stomach.
- The force-history peak captures all 12 substeps per policy step, but it is
  still simulated force; a drop-test or force-transducer experiment is needed
  before interpreting it as physiological load.
- `friction_estimate_N` is reconstructed from dynamics at 20 Hz and is intended
  for diagnostics, not as a calibrated friction sensor.
