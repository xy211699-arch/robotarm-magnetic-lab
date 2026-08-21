# TASK-008 Six-Action Dynamic-Force Controller

## Execution authority

Windows planning branch: `workflow/TASK-008-six-action-dynamic-force`

Required Linux implementation branch: `feature/TASK-008-six-action-dynamic-force-controller`

Exact source baseline: `06b15caf9a69bc9c20f85522ce4abbb32c8b9245` from `origin/feature/TASK-003-dynamic-capsule-force-teleop`

Authoritative design: `docs/design/2026-08-21-task008-six-action-dynamic-force-design.md`

Authoritative implementation plan: `docs/design/2026-08-21-task008-six-action-dynamic-force-implementation-plan.md`

Required return report: `handoffs/reports/TASK-008-six-action-dynamic-force-controller-report.md`

## Linux start instruction

Fetch `origin/workflow/TASK-008-six-action-dynamic-force`, record the fetched planning HEAD, and verify that its source lineage contains the exact baseline above. Create `feature/TASK-008-six-action-dynamic-force-controller` from the fetched planning HEAD, read both authoritative documents completely, and execute their checklist manually.

The Linux session is not required to install or invoke `superpowers:subagent-driven-development`, `superpowers:executing-plans`, or any other skill. Missing optional skills are not a blocker and must not change the contract.

## Objective

Replace the abandoned eleven-action, latch, virtual-magnet, and closed-loop pose-control routes with six one-second force macros on the verified TASK-003 dynamic rigid body.

The six IDs are `HOLD`, `MOVE_POS`, `MOVE_NEG`, `VIEW_POS`, `VIEW_NEG`, and `UP`. MOVE applies total `0.9mg` initially, split equally between both hemisphere centers. VIEW and UP apply `0.9mg` initially at the camera-side hemisphere center. MOVE and VIEW use `0.2 s` wait, `0.6 s` force, `0.2 s` wait. UP applies force for the full second. HOLD applies no active force.

Physics is `240 Hz`, environment/render control is `60 Hz`, camera and coverage are `30 Hz`, and the Actor boundary is `1 Hz`. Isaac Lab training uses synchronous stepping. At the UP boundary, capture RGB before clearing the force and do not advance physics during inference.

## Mandatory gates

Flat calibration uses 20 samples per non-HOLD action per candidate and searches from `0.9mg` to at most `3.0mg`. Final acceptance uses a separate held-out set of 20 samples per non-HOLD action and requires at least `16/20` for every action with no selected-candidate FAULT.

MOVE requires at least `5 mm` displacement projected along the force-onset direction. VIEW requires at least `15 deg` signed camera-side tilt in the requested direction. UP requires at least `45 deg` camera-side elevation at `t=1 s`, no crossing beyond world vertical, and RGB capture before release. HOLD verifies zero active force but not zero motion.

After flat acceptance, load the identical selected profile in the stomach. Do not run stomach calibration or claim a stomach success rate. Provide one-key/one-action inspection with simultaneously visible external, capsule-RGB, and cumulative coverage views. Coverage updates from every unique 30 Hz frame using the approved FOV, `50 mm` range, first-hit occlusion, and camera-facing normal gate.

## Forbidden changes

Do not modify USD assets, capsule mass or inertia, gravity, friction, damping, collision geometry, camera calibration, action duration, acceptance thresholds, VLM/RL code, rewards, prior reports, magnetic control, robot-arm control, ideal-surface control, latch control, virtual-magnet control, or pose/velocity outside reset.

Do not stop after one failed direction or one failed force candidate. Complete the authorized search and preserve evidence. Do not hide penetration, instability, or rendering limitations.

## Delivery

Push the Linux feature branch without merging. The report must distinguish verified facts, interpretations, deviations, and unverified visual judgments and must include external evidence paths with byte counts and SHA-256 hashes.
