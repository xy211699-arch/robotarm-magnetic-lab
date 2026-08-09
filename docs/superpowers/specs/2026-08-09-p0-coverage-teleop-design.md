# P0 Manual Teleoperation and Coverage Evaluation Design

**Status:** Approved by the user on 2026-08-09.

**Position in the project:** This is a prerequisite validation module placed before the formal section "Coverage Module Architecture and Data Flow." It validates the coverage definition, geometry implementation, visualization, and action-boundary protocol before reinforcement-learning integration.

## Purpose and Scope

The platform lets an operator drive the capsule in the Isaac Lab stomach scene with the existing eleven frozen atomic action IDs while viewing cumulative stomach-surface coverage in real time. Covered vertices are green, uncovered vertices are red, the capsule location is shown with a dark marker, and the traversed trajectory is black.

This module is not reinforcement learning. It contains no VLM, Actor, Critic, PPO, reward optimization, or policy update. Keyboard input temporarily replaces the future Actor at the same atomic-action request boundary. The resulting coverage implementation and records will later be reused by training and evaluation.

The module shall not redesign the eleven actions, change their templates, add preemption, add an action queue, or use capsule ground truth to alter action execution. Capsule pose may be used only by the privileged coverage evaluator, visualization, telemetry, and offline evidence.

## Frozen Existing Interfaces

The implementation shall reuse the existing non-preemptive atomic executor and these action IDs without renumbering:

| ID | Action |
|---:|---|
| 0 | `HOLD` |
| 1 | `TILT_POS` |
| 2 | `TILT_NEG` |
| 3 | `AZIMUTH_POS` |
| 4 | `AZIMUTH_NEG` |
| 5 | `ROLL_POS` |
| 6 | `ROLL_NEG` |
| 7 | `TURN_POS` |
| 8 | `TURN_NEG` |
| 9 | `APPROACH` |
| 10 | `RETREAT` |

An accepted action must run its complete predefined trajectory. The device-level terminal result remains strictly `DONE` or `HARD_FAILURE`. A busy executor shall reject new requests rather than queue them. A hard failure shall hold the last safe target and terminate the episode according to the existing containment behavior.

## Dedicated Environment

The implementation shall register a dedicated environment named `Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0`. It shall run one environment with a scalar discrete action interface of shape `(1,)` and reuse the existing stomach scene, capsule, robot, external magnet, camera configuration, magnetic bridge, atomic executor, safety checks, and action mask.

Physics and magnetic-force updates remain at 240 Hz. Atomic action control remains at 20 Hz. Only newly recorded 1 Hz RGB frame identifiers trigger a coverage update. The engineering visualization may refresh at 30 Hz, but it shall display the most recently completed coverage mask and shall not create additional observations or coverage samples.

## Keyboard and Session Protocol

The fixed keyboard mapping is:

| Key | Request |
|---|---|
| `W` / `S` | `TILT_POS` / `TILT_NEG` |
| `D` / `A` | `AZIMUTH_POS` / `AZIMUTH_NEG` |
| `E` / `Q` | `ROLL_POS` / `ROLL_NEG` |
| `C` / `Z` | `TURN_POS` / `TURN_NEG` |
| `R` / `F` | `APPROACH` / `RETREAT` |
| `Space` | `HOLD` |
| `Backspace` | reset at an action boundary |
| `F12` | save a snapshot |
| `Esc` | exit |

Only key-down edges may submit requests. Operating-system key repeat must not resubmit an action. Requests received while the executor is busy return `IGNORED_WHILE_BUSY`; requests disabled by the current action mask return `MASKED_ACTION`; reset while busy returns `RESET_WHILE_BUSY`; and requests after episode termination return `EPISODE_TERMINATED`. These are session-level request outcomes and do not expand the device result vocabulary.

After `DONE`, the controller shall log the completed request and acknowledge the boundary exactly once before accepting the next request. After `HARD_FAILURE`, the controller shall accept only reset or exit. No pause, cancellation, queue, automatic recovery, or recenter operation is authorized.

## Coverage Reference Surface

The denominator is the complete inner luminal surface of the stomach mesh. The outer wall, thickness side walls, artificial opening caps, collision proxies, and helper geometry are excluded. Complete-surface coverage is the experimental objective even though the intended capsule route uses the comparatively flat, lower-climb region.

At startup, a prerequisite inspector shall enumerate every `UsdGeom.Mesh` below the configured stomach root and record prim paths, vertex and face counts, world-space bounds, transforms, purpose, visibility, and available names or material cues. The implementation must not infer the inner surface from a convenient mesh name alone. If the inner luminal surface cannot be selected unambiguously from the asset structure, execution stops with `needs_decision` and reports the candidates.

Selected reference vertices are transformed to world coordinates and duplicate coincident vertices are welded with tolerance `1e-6 m`. The preprocessing output shall preserve, for every welded reference vertex, the set of incident stomach triangles needed by the first-hit visibility test. The mesh identity, selected prim paths, preprocessing parameters, and a deterministic asset or geometry hash shall be recorded.

## Visibility and Coverage Definition

A reference vertex is a candidate for recorded frame `t` only when its Euclidean distance from the optical center is at most `0.05 m` and its camera-frame direction lies within a circular field-of-view cone with half-angle `60 degrees` around the optical axis. The initial implementation therefore uses a 120-degree circular FOV and a maximum effective observation distance of 50 mm.

Every candidate is then tested with a GPU-batched first-hit ray from the optical center toward that vertex. A candidate is visible only when the first stomach-surface hit belongs to one of that vertex's incident triangles and the absolute difference between hit distance and vertex distance is at most `1e-4 m`. Rays must be evaluated against the selected stomach reference surface, not debug geometry or the capsule.

The scalar per-ray implementation is allowed only as a deterministic test oracle on small synthetic fixtures. A depth-buffer approximation is not an authorized fallback. If the installed Isaac Sim or Isaac Lab stack cannot provide a suitable GPU-batched ray-query path, the Linux executor shall stop with `needs_decision` and report the available APIs and measured limitation.

The persistent coverage mask is cumulative and monotonic within an episode:

\[
M_t = M_{t-1} \lor V_t,
\qquad
C_t = \frac{\sum_i M_t(i)}{N}.
\]

Only a new recorded 1 Hz frame ID may produce `V_t` and update `M_t`. Preview frames, physics steps, UI refreshes, and repeated reads of the same frame ID must not change coverage. Reset clears the accumulated mask, trajectory, counters, and elapsed-session state after the current action boundary.

## Visualization

The primary engineering display is an independent Kit 3D point-cloud view of the reference vertices. Uncovered vertices are red and covered vertices are green. It also shows the capsule position with a dark marker and the capsule trajectory as a black line. Debug rendering must be non-colliding, non-magnetic, excluded from ray queries, and isolated from the capsule RGB sensor so it cannot change physics or deployable observations.

The overlay shall show cumulative coverage, latest coverage gain, elapsed time, current action ID and name, request identifier, executor state, latest result, action mask, recorded-frame count, candidate count, newly covered count, and ray-query/update timing.

A deterministic 2D projection shall be exported on `F12`, reset, hard failure, and normal exit. It shall include the red/green coverage view, coverage percentage, elapsed time, legend, capsule position, and trajectory. Projection orientation, transform, scale, and image size shall be stored in metadata. This projection is a visualization artifact only and never participates in coverage calculation.

## Records and Artifacts

Each run shall write to `logs/p0_coverage_teleop/<session-id>/`. The small text records are `metadata.json`, `actions.jsonl`, and `frames.jsonl`. Large masks, trajectories, images, and timing arrays stay outside Git. The Linux report shall list every produced artifact with path, byte size, and SHA-256 digest.

`metadata.json` records commit, branch, task ID, environment ID, simulator and dependency versions, random seed, camera parameters, stomach prim selection, mesh hash, transforms, weld tolerance, visibility thresholds, action-table version, key mapping, projection metadata, and clock definitions. `actions.jsonl` records request and completion boundaries, key, action ID/name, mask decision, state transitions, session outcome, device result when applicable, and timestamps. `frames.jsonl` records each unique 1 Hz frame ID, pose used only by the evaluator, candidate/visible/new/cumulative counts, percentage, and timing breakdown.

## Safety and Information Isolation

Existing hard safety checks for robot collision, workspace and joint limits, XRDF constraints, planned acceleration, mechanical clearance, and magnetic-force chain remain authoritative. Coverage, camera rays, visualization state, capsule truth, and observed action-effect error shall not be added to the executor's action mask, completion condition, or hard-failure condition.

The capsule pose used by coverage calculation is privileged evaluator information. It shall not enter the deployable observation path, action request, atomic state machine, action mask, action completion, or hard-failure logic. The implementation shall include a field-whitelist or equivalent dependency check demonstrating this separation.

## Acceptance Contract

Protocol tests shall cover every key, key-down de-duplication, operating-system repeat suppression, busy rejection without queuing, mask rejection, boundary-only reset, post-termination rejection, and one acknowledgement per completed request.

Geometry tests shall cover the exact 50 mm distance boundary, 60-degree cone boundary, occlusion by a nearer surface, incident-triangle acceptance, non-incident first-hit rejection, cumulative monotonicity, reset, and agreement between GPU batch results and the scalar oracle on deterministic fixtures.

Integration tests shall confirm registration of the dedicated task, scalar action shape, submission of all eleven IDs, exactly one coverage update for each unique recorded 1 Hz frame, absence of duplicate updates, and exact agreement among stored masks, red/green vertex counts, overlay percentage, JSON records, and exported images.

Performance evidence shall contain at least 100 coverage-update samples in the target stomach scene. Every update must finish before the next 1 Hz recorded frame deadline. The report must provide median, p95, maximum, candidate counts, ray counts, and GPU/CPU synchronization assumptions. This gate validates operability, not a final optimization target.

Regression shall rerun the stage-one pure protocol suite, the existing table atomic-action acceptance for all eleven actions, and the legacy 9D table smoke test. No minimum final coverage percentage is imposed in P0 because the purpose is to validate measurement and interaction, not to claim autonomous complete coverage.

## Explicit Non-Goals

P0 does not train an Actor-Critic policy, use VLM features, define a reward function, perform domain randomization, establish robust stomach navigation, add capsule-motion closed-loop control, redesign atomic actions, or prove that complete stomach coverage is dynamically achievable. Those decisions remain downstream of an accepted coverage evaluator and measured manual-operability evidence.
