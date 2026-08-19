# TASK-006 Hybrid Latched Eleven-Action Controller

## Execution authority

Windows design branch: `workflow/TASK-006-hybrid-latched-v1`

Linux implementation branch: `feature/TASK-006-hybrid-latched-v1`

Exact implementation base: `67b7bf44747f08422add0cee7e6b94280bbeff6d`

Authoritative design: `docs/superpowers/specs/2026-08-19-hybrid-latched-eleven-action-controller-design.md`

Authoritative implementation plan: `docs/superpowers/plans/2026-08-19-hybrid-latched-eleven-action-controller.md`

Required return report: `handoffs/reports/TASK-006-hybrid-latched-eleven-action-controller-report.md`

## Linux execution instruction

The Linux session must read the design and implementation plan completely before editing. The Linux environment is not required to install or invoke `superpowers:subagent-driven-development`, `superpowers:executing-plans`, or any other superpowers skill. Execute the repository checklist manually, task by task, with the specified tests and evidence gates.

Do not reuse the old TASK-005 `needs_decision` conclusion as a TASK-006 result. TASK-005 remains the baseline evidence showing that continuous dynamic settling failed. TASK-006 must produce new latch-backend, paired-release, flat randomized, RGB-boundary, stress, stomach, and regression evidence.

## Frozen decisions

Keep GPU dynamics and record that sweep CCD is disabled; do not switch to CPU and do not change CCD configuration.

Keep the existing eleven action IDs, nine-grid camera-relative VIEW mapping, relative 15 degree target, strict one-second/240-substep boundary, `0.9mg` shared MOVE force, keyboard mapping, and only three public results: `COMPLETED`, `REJECTED`, and `FAULT`.

Use `dynamic rigid body + all six rigid-dynamic lock flags` as the preferred and initially selected latch. Run the mandatory live CUDA feasibility and paired-release gate before integrating it. Change the tracked `selected_backend` to the approved kinematic fallback only after preserving evidence that dynamic lock flags failed. If both fail, return `needs_decision`.

Every 1 Hz policy RGB frame must be captured only after six-DOF latch, zero wrench, zero linear velocity, and zero angular velocity are confirmed. Frame zero follows the same rule. Continuous rendering remains active independently of the policy-frame rate.

VIEW latches the actual pose on the first substep where full target-axis error is at most 3 degrees and support tangent drift is at most 2 mm. A real camera-hemisphere contact has priority and immediately latches the actual pose. No angular-speed dwell is required. If neither occurs, latch the actual pose at substep 240 and return `COMPLETED` without adding a `target_reached` field.

HOLD remains latched for all 240 substeps. Rejected MOVE remains latched for all 240 substeps and returns `REJECTED`. Accepted MOVE unlocks, executes the unchanged 60/120/60 free/force/free schedule, computes direction at 0.25 seconds, and latches at substep 240 without early 5 mm stopping.

MOVE uses the immutable contact snapshot captured at the preceding latch, not nonzero ContactSensor force during the locked wait. A planned-valid validation sample that fails the actual latched predicate must be reclassified and replaced.

Never snap or project pose, repeatedly write pose, repair penetration, add a fourth result, expose latch truth to the future Actor, change assets/physics, or adapt parameters for stomach.

## Blocking gates

The preferred/fallback latch backend must pass live flag/mode readback, one-second zero-drift hold, no API-time pose jump, and paired release comparison. During the first 0.05 seconds after unlock, the latched-versus-direct maximum position difference must be at most 0.5 mm and maximum target-axis difference at most 1 degree.

Flat randomized acceptance is blocking. Each action needs at least ten actual-valid randomized starts; VIEW needs ten unblocked target-latch samples, each MOVE needs at least ten eligible plus ten ineligible samples, and valid MOVE signed-displacement success must be at least 90 percent for both signs. All policy frames must be latched and zero-velocity. Then run the fixed no-reset 100-action sequence.

Only after every flat and release gate passes may the identical controller, backend, profiles, digests, and 100-ID sequence run in the stomach scene. Stomach receives no adaptation and no second 5 mm gate. Provide keyboard visualization evidence for user review.

## Stop and return rules

Return `needs_decision` and skip stomach if both latch backends fail or any flat/release/stress gate fails. Return `partial` if flat passes but unchanged stomach initialization produces a true system FAULT. Return `complete` only after flat, paired release, stress, same-digest stomach run, visualization evidence, and regressions finish.

Push `feature/TASK-006-hybrid-latched-v1` without merging. The return message must contain final HEAD, disposition, report path, selected latch backend, both profile digests, key flat/release/stomach metrics, regression counts, and external evidence paths with byte sizes and SHA-256 values.
