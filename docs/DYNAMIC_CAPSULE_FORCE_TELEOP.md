# Dynamic Capsule Force Teleoperation

## Scope

`Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0` is an isolated,
single-environment Isaac Lab task for testing a real dynamic capsule against the
delivered static stomach collider. PhysX owns capsule pose and velocity during
normal operation. The only commanded actuator is a bounded world-frame force at
the capsule center of mass; commanded torque is always zero.

The task intentionally contains no magnetic actuator, ideal-surface controller,
robot command, hidden support force, pose projection, penetration recovery,
reward, or capsule root-state writer. `Backspace` invokes the normal episode reset,
which is the only permitted state reinitialization.

## Controls

| Key | Held command |
|---|---|
| `W` / `S` | world `+X` / `-X` |
| `A` / `D` | world `+Y` / `-Y` |
| `Q` / `E` | world `+Z` / `-Z` |
| `Space` | clear all held force keys |
| `Backspace` | reset episode |
| `F12` | save diagnostic snapshot |
| `Escape` | exit cleanly |

Opposite keys cancel. Simultaneous orthogonal keys are norm-limited. The default
single-axis magnitude is `0.5 * live_capsule_mass * 9.81 N`; the accepted CLI
ratio range is `(0, 2]`. With the delivered live mass of approximately
`0.005735 kg`, the default force is approximately `0.02813017 N`.

## Timing and physics

- PhysX: 240 Hz (`dt=1/240 s`)
- environment and held-key update: 60 Hz (`decimation=4`)
- Kit render interval: four physics steps, approximately 60 simulated Hz
- capsule camera: 30 simulated Hz
- one environment, CPU PhysX, GPU RTX rendering

CPU PhysX is required in this installed Isaac Lab version because its physics
manager disables scene CCD when GPU Dynamics is active. Both scene CCD and capsule
body CCD are verified by the preflight. Contact-constrained directions need not
produce equal or free-space displacement.

## Commands

Preflight:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/inspect_dynamic_force_prerequisites.py \
  --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 \
  --num_envs 1 --headless
```

Deterministic no-input and six-direction validation:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/validate_dynamic_force_stomach.py \
  --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 \
  --seed 42 --force_weight_ratio 0.5 --headless
```

Interactive teleoperation:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/teleop_dynamic_force_stomach.py \
  --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 \
  --force_weight_ratio 0.5 --viz kit
```

Scripted rendered smoke:

```bash
./run_isaaclab.sh -p scripts/dynamic_force/teleop_dynamic_force_stomach.py \
  --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 \
  --force_weight_ratio 0.5 \
  --scripted_sequence "+x:0.5,zero:0.25,-x:0.5,zero:0.25" \
  --max_steps 120 --viz kit
```

Evidence is written outside Git under
`/mnt/isaac-linux/robotarm_magnetic_lab/logs/dynamic_force_preflight`,
`dynamic_force_validation`, and `dynamic_force_teleop`.

## Current interpretation boundary

The interface, timing, exact wrench, contact sensing, dynamic motion, and rendered
execution are verified. The delivered stomach/capsule parameters currently show a
minimum measured clearance near `-1.98 mm` and one small authored-velocity-bound
overshoot in the deterministic run. These are exposed as a `partial` result; they
were not hidden by pose correction or altered by unauthorized physics tuning.
The stomach mesh also has 21 known open boundary edges. No claim is made about
tissue deformation, fluid effects, magnetic tracking, autonomous navigation,
hardware calibration, or subjective long-duration smoothness.
