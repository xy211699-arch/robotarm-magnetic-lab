# P0 stomach teleoperation and coverage

This launcher exposes the frozen scalar atomic-action boundary in a dedicated
one-environment stomach task. It adds a privileged evaluator; it does not add
capsule feedback to the controller or change any action trajectory.

## Start

From the repository root:

```bash
./run_isaaclab.sh -p scripts/action_layer/teleop_atomic_stomach_coverage.py \
  --task Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0 \
  --num_envs 1 --viz kit
```

Headless all-action validation:

```bash
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py \
  --num_envs 1 --coverage_samples 100
```

## Fixed keyboard contract

| Key | Action |
|---|---|
| Space | HOLD (0) |
| W / S | TILT_POS (1) / TILT_NEG (2) |
| D / A | AZIMUTH_POS (3) / AZIMUTH_NEG (4) |
| E / Q | ROLL_POS (5) / ROLL_NEG (6) |
| C / Z | TURN_POS (7) / TURN_NEG (8) |
| R / F | APPROACH (9) / RETREAT (10) |
| Backspace | Reset at an action boundary |
| F12 | Save a coverage snapshot |
| Esc | Save and exit |

Only a key-down edge submits. A request made while an action is active is
reported as `IGNORED_WHILE_BUSY` and discarded; it is never queued. A disabled
action is `MASKED_ACTION`. Busy reset is `RESET_WHILE_BUSY`. After a device
`HARD_FAILURE`, only reset or exit is accepted.

## Coverage and clocks

The denominator is the preflight-approved rendered luminal surface. Each new
recorded 1 Hz camera frame evaluates welded vertices within 50 mm and a circular
120-degree full FOV. CUDA batched first-hit rays must hit a triangle incident to
the candidate vertex within 0.1 mm. The cumulative mask is monotonic until a
valid reset. The guide-purpose 3D point view refreshes from the latest mask at
up to 30 Hz and never participates in rays, collision, magnetic force, the
capsule RGB stream, policy observations, or executor decisions.

Physics/magnetic force run at 240 Hz, atomic decisions at 20 Hz, recorded RGB
and coverage at 1 Hz, and the engineering view at 30 Hz.

## Evidence layout

Each run writes atomically under
`logs/p0_coverage_teleop/<UTC-session-id>/`: `metadata.json`, append-only
`actions.jsonl` and `frames.jsonl`, final mask/trajectory/timing arrays,
deterministic projection PNG/JSON snapshots, `summary.json`, and an artifact
inventory. The directory is renamed from a hidden partial directory only at
clean finalization. Generated evidence is not committed.

P0 deliberately defines no minimum coverage percentage and provides no RL,
VLM, autonomous navigation, queueing, preemption, automatic recovery, or
capsule-motion feedback control.
