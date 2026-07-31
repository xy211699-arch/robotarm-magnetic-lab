# Robotarm Magnetic Lab

Isaac Lab 3.0 single-environment task for the AUBO-style six-axis arm, the
three-axis magnetic ball assembly, and the external capsule magnet.

## Project status and execution history

- Incremental work since the 2026-07-17 handover:
  [`docs/POST_HANDOVER_WORK_SUMMARY.md`](docs/POST_HANDOVER_WORK_SUMMARY.md)
- Concise per-conversation execution log:
  [`docs/PROJECT_RUN_LOG.md`](docs/PROJECT_RUN_LOG.md)

## Current bring-up status

- Task ID: `Template-Robotarm-Magnetic-Lab-v0`
- Workflow: Manager-Based, single agent
- Physics: 240 Hz
- Policy: 20 Hz (`decimation=12`)
- Action: 9 joint-position increments in this exact order:
  `j1..j6, ballxj, ballyj, ballzj`
- State observation: 31 values: nine relative joint positions, nine joint
  velocities, 12 magnetic wrench values, and one ASM clearance value
- Vision observation: capsule-mounted `1280x720` circular RGB and aligned
  metric depth; inactive pixels outside the optical circle are zero
- Capsule camera: provisional DS01/CX93510-series model, 120-degree horizontal
  circular FOV, 30 Hz render rate, local `+Z` optical direction, equidistant
  wide-angle remap, soft optical border and nominal edge lens shading
- Illumination: four 5600 K close-range LEDs mounted around the camera axis
- Reset: directly restores the validated initialization pose; it does not
  replay the old Script Editor trajectory
- Asset source: `/home/multirobo/Desktop/sim of FF/Stage.usd`

The training asset
[`assets/robotarm_magnetic_training.usda`](assets/robotarm_magnetic_training.usda)
is a non-destructive compatibility layer. It leaves the source stage unchanged
and relocates `PhysicsArticulationRootAPI` so Isaac Lab's PhysX tensor API can
bind the robot and attached ASM as one nine-DOF articulation.

## Installation

Always use the project launcher below. It removes Conda/virtual-environment
variables before calling Isaac Lab, so the simulator cannot accidentally mix a
Conda interpreter with Isaac Sim's Python 3.12 standard library.

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab
./run_isaaclab.sh -p -m pip install -e \
  source/robotarm_magnetic_lab
```

## Validation

List the registered task:

```bash
./run_isaaclab.sh -p \
  scripts/list_envs.py --keyword Robotarm-Magnetic
```

Run the finite 100-step headless smoke test:

```bash
./run_isaaclab.sh -p \
  scripts/zero_agent.py \
  --task Template-Robotarm-Magnetic-Lab-v0 \
  --num_envs 1
```

Run interactively with the Kit viewport:

```bash
./run_isaaclab.sh -p \
  scripts/zero_agent.py \
  --task Template-Robotarm-Magnetic-Lab-v0 \
  --num_envs 1 \
  --viz kit
```

Close the Kit window to stop the visual run. The headless version terminates
automatically after 100 environment steps.

Run the permanent nine-axis interface acceptance test:

```bash
./run_isaaclab.sh -p \
  scripts/validate_interfaces.py \
  --task Template-Robotarm-Magnetic-Lab-v0 \
  --num_envs 1
```

This test excites `j1..j6` and `ballxj/ballyj/ballzj` one at a time and records
joint tracking, capsule motion, field anchoring and collision clearance in:

`logs/interface_validation.jsonl`

The zero-agent camera diagnostics save the processed policy RGB and depth
previews (not the raw rectangular RTX buffers) in:

`logs/camera/`

For a finite interactive regression, add both `--viz kit` and
`--max_steps 100` to the zero-agent command.

## Main configuration

The environment configuration is:

`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_lab_env_cfg.py`

It currently provides the stable simulation/control foundation, analytical
magnetic coupling, approximate ASM collision clearance, and the first
capsule-view RGB-D interface. The next migration stages are:

1. replace provisional camera intrinsics/extrinsics with measured endoscope
   calibration;
2. add PhysX contact sensors and stomach/capsule contact observations;
3. add task goal, progress, and success terms;
4. add deterministic episode recording for behavior cloning;
5. add lighting, material, friction, magnetic and camera randomization;
6. scale from one environment to multiple environments.

Do not start PPO training with the current neutral reward. First complete the
task reward, contact safety, demonstration recorder, and reset randomization
interfaces.

## Model fine-tuning data interface

The versioned model contract is:

`configs/interfaces/robotarm_magnetic_v1.json`

It freezes image shapes, units, state layout, joint order, action semantics and
control rates. The episode recorder, integrity validator and temporal
fine-tuning index are documented in:

[`docs/TRAINING_DATA_WORKFLOW.md`](docs/TRAINING_DATA_WORKFLOW.md)

Generated data belongs under `datasets/` and is intentionally ignored by Git.
Do not place recorded images or episode JSONL files in `logs/` or source
directories.
