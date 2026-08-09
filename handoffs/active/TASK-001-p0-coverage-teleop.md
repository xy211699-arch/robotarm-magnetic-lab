# TASK-001: P0 Manual Teleoperation and Coverage Evaluation

**Status:** Approved for Linux implementation by the user on 2026-08-09.

**Planning branch:** `workflow/TASK-001-p0-coverage-teleop`

**Required Linux branch:** `feature/TASK-001-p0-coverage-teleop`

**Design authority:** `docs/superpowers/specs/2026-08-09-p0-coverage-teleop-design.md`

**Execution plan:** `docs/superpowers/plans/2026-08-09-p0-coverage-teleop.md`

**Required report:** `handoffs/reports/TASK-001-p0-coverage-teleop-report.md`

## Objective

Implement a prerequisite Isaac Lab platform that lets an operator submit the existing eleven frozen atomic actions from a fixed keyboard mapping in a dedicated stomach environment and evaluate cumulative complete-inner-surface coverage using a 120-degree circular FOV, 50 mm maximum observation distance, and GPU-batched first-hit ray occlusion.

This task validates the coverage evaluator, display, records, and action-boundary protocol. It does not authorize reinforcement-learning training, VLM integration, reward design, action redesign, or capsule-motion feedback control.

## Base and Branch Rules

Linux shall fetch the planning branch and record its exact commit as `base_commit` before editing. Linux shall create `feature/TASK-001-p0-coverage-teleop` from that exact commit and shall not implement directly on `main` or on the Windows planning branch.

Linux shall preserve unrelated local work. If the checkout is dirty or the required base cannot be established without overwriting work, report `blocked` and stop.

## Mandatory Preflight Gate

Before feature implementation, Linux shall inspect the actual target scene and dependency stack. The preflight shall enumerate every stomach `UsdGeom.Mesh`, its topology, transforms, bounds, and plausible surface role; identify the configured capsule camera and its optical transform; confirm the existing atomic task and action IDs; and identify an available GPU-batched first-hit ray-query API.

Linux shall report `needs_decision` and stop before later implementation when the complete inner luminal surface cannot be selected unambiguously, when the camera optical transform cannot be confirmed, or when no suitable GPU-batched first-hit ray API exists. Linux may not guess the surface, substitute a depth buffer, or silently use scalar production rays.

## Authorized Changes

Linux may add isolated coverage, teleoperation, record, and visualization modules; a dedicated stomach teleoperation task; task registration; validation scripts; tests; user documentation; and the required report. Linux may make the smallest package-export changes needed to expose these additions.

Linux may update `.gitignore` only to exclude newly generated P0 logs, masks, images, timing arrays, and simulator artifacts. The report must identify any path that differs from the implementation plan because of the actual package layout.

## Forbidden Changes

Linux shall not edit the existing atomic executor semantics, action IDs, action templates, action controller trajectories, safety thresholds, device result vocabulary, camera calibration, robot or stomach assets, magnetic-force model, existing task behavior, training configuration, or prior acceptance results.

Linux shall not add preemption, queuing, pause, automatic recovery, recentering, action-effect error as a hard safety condition, or a final coverage-success threshold. If an authorized addition cannot be made without one of these changes, report `needs_decision`.

## Frozen Interaction Contract

The action IDs remain `HOLD=0`, `TILT_POS=1`, `TILT_NEG=2`, `AZIMUTH_POS=3`, `AZIMUTH_NEG=4`, `ROLL_POS=5`, `ROLL_NEG=6`, `TURN_POS=7`, `TURN_NEG=8`, `APPROACH=9`, and `RETREAT=10`.

The keys remain `W/S`, `D/A`, `E/Q`, `C/Z`, and `R/F` for the corresponding positive and negative pairs; `Space` requests `HOLD`; `Backspace` requests a boundary reset; `F12` saves a snapshot; and `Esc` exits.

Only key-down edges submit actions. Busy requests are discarded as `IGNORED_WHILE_BUSY` and never queued. Disabled requests return `MASKED_ACTION`. Busy reset returns `RESET_WHILE_BUSY`. Post-termination requests return `EPISODE_TERMINATED`. These session outcomes do not alter the device result set `DONE/HARD_FAILURE`.

## Frozen Coverage Contract

The denominator is the complete stomach inner luminal surface and excludes outer wall, thickness side walls, artificial caps, collision proxies, and helpers. Selected vertices are transformed to world space, welded at `1e-6 m`, and retain incident-triangle membership. The selected prims and a deterministic geometry hash are recorded.

Coverage updates occur only once for each new recorded 1 Hz frame ID. A vertex must be within 50 mm Euclidean distance, within a circular cone of 60-degree half-angle, and visible under the incident-triangle first-hit rule with `1e-4 m` distance tolerance. The episode mask is cumulative and monotonic until a valid reset.

The primary display is an isolated 3D point-cloud view with uncovered red, covered green, a dark capsule marker, and a black trajectory. A deterministic 2D projection with coverage, elapsed time, legend, position, and trajectory is exported on snapshot, reset, hard failure, and exit. Neither display participates in coverage calculation.

## Information Isolation

Capsule ground-truth pose is authorized only inside the privileged evaluator, visualization, telemetry, and offline evidence path. Coverage, rays, visualization, and capsule truth must not enter deployable observations, the action request, action mask, executor state transition, completion condition, or hard-failure condition.

The dedicated task remains one environment with scalar action shape `(1,)`, 240 Hz physics and magnetic-force updates, 20 Hz atomic control, 1 Hz recorded RGB and coverage boundaries, and a 30 Hz engineering display that reuses the latest completed mask.

## Required Acceptance

Linux shall pass the keyboard/session protocol suite, reference-mesh tests, geometric boundary and occlusion tests, GPU-versus-scalar fixture comparison, monotonic/reset tests, task-registration and observation-isolation tests, all-eleven-action integration, one-update-per-unique-frame checks, and exact mask/UI/log/export consistency checks.

Linux shall collect at least 100 stomach-scene coverage-update timing samples and show that every completed update fits within the 1 second recorded-frame interval. Linux shall report median, p95, maximum, candidate counts, ray counts, and synchronization assumptions.

Linux shall also rerun the stage-one pure protocol suite, the existing eleven-action table acceptance, and the legacy 9D table smoke regression. P0 has no minimum final coverage percentage.

## Delivery Contract

The final Linux report shall state one of `complete`, `partial`, `needs_decision`, or `blocked`. It shall include base commit, head commit, branch, every validation command, exact results, deviations from this contract, unverified items, and every external evidence artifact with path, byte size, and SHA-256 digest.

Linux shall push the feature branch but shall not merge it. Windows remains responsible for reviewing the diff and evidence and for authorizing any subsequent implementation or integration.
