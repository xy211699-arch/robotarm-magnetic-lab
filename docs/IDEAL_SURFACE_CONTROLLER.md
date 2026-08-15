# Ideal Surface Controller

## Purpose and boundary

`Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0` is a
single-environment Isaac Lab task for evaluating high-level capsule exploration.
Its `ideal_surface_v2` controller is a privileged simulation oracle: it reads the
approved stomach mesh and capsule truth, then writes a continuous kinematic target
at 240 Hz. It is not a magnetic controller and does not demonstrate that the
robot, external magnet, or a physical capsule can reproduce these motions.

The Actor observation remains capsule RGB only. Coverage rays, surface normals,
active triangles, contact geometry, and capsule truth remain confined to the
controller, evaluator, diagnostics, and offline evidence.

## Frozen action table

| ID | Name | Keyboard | Meaning |
|---:|---|---|---|
| 0 | `HOLD` | Space | Hold the current pose for one action boundary. |
| 1 | `START_TILT_000` | T / Numpad 8 | From logical upright, tilt to 15° at local azimuth 0°. |
| 2 | `START_TILT_045` | Y / Numpad 9 | Tilt to 15° at 45°. |
| 3 | `START_TILT_090` | H / Numpad 6 | Tilt to 15° at 90°. |
| 4 | `START_TILT_135` | N / Numpad 3 | Tilt to 15° at 135°. |
| 5 | `START_TILT_180` | B / Numpad 2 | Tilt to 15° at 180°. |
| 6 | `START_TILT_225` | V / Numpad 1 | Tilt to 15° at 225°. |
| 7 | `START_TILT_270` | F / Numpad 4 | Tilt to 15° at 270°. |
| 8 | `START_TILT_315` | R / Numpad 7 | Tilt to 15° at 315°. |
| 9 | `TILT_MORE` | W | Increase tilt by 15° in the current tilt plane. |
| 10 | `RISE` | S | Pivot 15° about the non-camera (+Z) end toward the 180° upright pole, lifting the camera end. |
| 11 | `PRECESS_POS` | D | Increase local azimuth by 15° while holding tilt. |
| 12 | `PRECESS_NEG` | A | Decrease local azimuth by 15° while holding tilt. |
| 13 | `ROLL_POS` | E | Positive right-hand axial roll with 10.0 mm no-slip arc. |
| 14 | `ROLL_NEG` | Q | Negative right-hand axial roll with 10.0 mm no-slip arc. |

The letter compass is `R T Y / F _ H / V B N`; the numpad layout remains an
alias. Both are compasses in the local stomach tangent plane. Azimuth zero
is the projected camera image-up direction and positive azimuth follows the
right-hand rule about the inward surface normal.

## State and safety semantics

- One Actor action lasts exactly 1.0 simulated second. A quintic target is
  evaluated at every 1/240 s physics substep.
- Logical upright enters at 5°, exits at 8°, and requires 0.1 s stability.
- Side contact requires at least two near-contact barrel samples separated by at
  least 25% of cylinder length for 0.1 s. Normal end support alone is not side
  contact, and roll actions remain masked until stable side contact.
- A predicted contact or open boundary stops at the last safe sub-target and
  returns normal `DONE` with `contact_limited` or `boundary_limited`.
- A manually requested masked action completes as `DONE/no_effect`; it is not a
  device failure.
- `HARD_FAILURE` is reserved for nonfinite state, lost valid surface,
  nonadjacent surface jump, or actual penetration greater than 0.05 effective
  radius.

## Manual coverage session

From the repository root:

```bash
./run_isaaclab.sh -p scripts/ideal_surface/teleop_ideal_surface_stomach_coverage.py \
  --task Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0 \
  --num_envs 1 --seed 42 --viz kit
```

Special keys are Backspace to reset the episode and cumulative coverage, F12 to
write a snapshot, and Escape to exit. The launcher accepts one action only when
the previous result has been acknowledged. It displays the cumulative P0
coverage point cloud at up to 30 Hz while the policy camera and recorded coverage
frames remain 1 Hz.

For a noninteractive smoke, supply comma-separated action IDs and headless mode:

```bash
./run_isaaclab.sh -p scripts/ideal_surface/teleop_ideal_surface_stomach_coverage.py \
  --headless --scripted_actions 0,10,10,10
```

Session records are written under the primary clone's absolute
`logs/ideal_surface_coverage_teleop/<UTC-ID>/` path. This remains true when the
launcher runs from a linked Git worktree. At startup the exact destination is
printed as `IDEAL_SURFACE_OUTPUT_DIRECTORY ...`, and finalization prints
`IDEAL_SURFACE_OUTPUT ...`.
They include metadata, action events, coverage frames, mask, trajectory, timing,
snapshots, summary, and an artifact inventory. `logs/` is external evidence and
must not be committed.

## Validation

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface -q
./run_isaaclab.sh -p scripts/ideal_surface/validate_ideal_surface_geometry.py
./run_isaaclab.sh -p scripts/ideal_surface/validate_ideal_surface_stomach.py \
  --task Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0 \
  --seed 42 --random_actions 1000 --output logs/ideal_surface_validation
```

The long validator samples uniformly only from the current mask and fails on
duplicate requests or coverage frames, nonfinite poses, disconnected component
jumps, unexpected hard failure, or hard penetration. Final coverage is recorded
for information and has no TASK-002 pass threshold.
