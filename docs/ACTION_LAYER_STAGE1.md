# Atomic action layer — implementation stage 1

This stage implements the revised 2026-08-08 short-action contract without
changing the validated legacy 9-D environments.

## Frozen interface

| ID | Action |
|---:|---|
| 0 | `HOLD` |
| 1/2 | `TILT_POS` / `TILT_NEG` |
| 3/4 | `AZIMUTH_POS` / `AZIMUTH_NEG` |
| 5/6 | `ROLL_POS` / `ROLL_NEG` |
| 7/8 | `TURN_POS` / `TURN_NEG` |
| 9/10 | `APPROACH` / `RETREAT` |

The independent task is
`Template-Robotarm-Magnetic-Atomic-Table-Lab-v0`. Its policy action shape is
`(1,)`; the value is one integer ID. The existing zero-dimensional magnetic
physics action continues to refresh the analytical wrench at 240 Hz.

The action executor runs at the environment's 20 Hz control rate. An accepted
action cannot be preempted. The SMDP wrapper must observe a terminal result and
call `action_manager.get_term("atomic").acknowledge_result()` before submitting
the next request. This prevents a held action tensor from accidentally
repeating an action at the first frame after `DONE`.

## Safety and privilege boundary

The planner pre-samples the complete joint trajectory and validates joint
limits, velocity, acceleration, external-magnet workspace and XRDF
ASM-to-robot sphere clearance. Runtime monitoring uses only encoder/FK/device
signals. Capsule pose, contact, magnetic wrench, depth and stomach truth are
not fields in the deployable snapshot or action result.

An action has exactly two terminal outcomes: `DONE` or `HARD_FAILURE`.
Deviation of capsule motion from the intended effect is an offline evaluation
signal and cannot produce `HARD_FAILURE`. Initial failure containment holds the
last validated target and lets the environment terminate; autonomous recovery
motion remains intentionally unimplemented until separately validated.

## Validation

Dependency-light contracts:

```bash
python3 scripts/action_layer/validate_pure.py
```

Isaac Lab table validation (all 11 IDs, one reset per action):

```bash
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_table.py \
  --task Template-Robotarm-Magnetic-Atomic-Table-Lab-v0 \
  --visualizer none
```

Results are written to `logs/action_layer/stage1_atomic_results.jsonl`.

## 2026-08-08 acceptance result

- Pure contract/safety tests: 10/10 passed.
- Gym task registration: passed; action space is one scalar and deployment
  observation space is 25 values (9 joint positions, 9 velocities, 7 magnet
  pose values).
- Isaac Lab table run: all 11 action IDs reached `DONE`; action duration was
  0.95--1.05 s and minimum observed/planned ASM clearance was about
  9.94--10.00 mm.
- A first run exposed false hard stops from an uncalibrated raw acceleration
  estimate. Planned trajectory acceleration remains hard-limited; runtime
  acceleration is logged but will not become a hard stop until its physical
  threshold and filter are calibrated.
