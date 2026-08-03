# Robotarm Magnetic Model I/O and Fine-tuning Workflow

## Canonical timing contract

The current model-facing contract is
`configs/interfaces/robotarm_magnetic_v2.json`. Version 1 remains readable
only for the historical bring-up dataset and must not be mixed with v2 data.

The stomach system is asynchronous by design:

| Layer | Rate | Meaning |
|---|---:|---|
| PhysX and analytical magnetic wrench | 240 Hz | contact and force integration |
| low-level joint control and raw rows | 20 Hz | one state/action transition every 50 ms |
| capsule camera acquisition | 1 Hz | physical DS01-class policy observation |
| VLA/high-level inference | 1 Hz | one inference per newly acquired image |
| engineering preview | 30 Hz | GUI only; never recorded as policy input |

One newly acquired RGB-D frame at time `t` supervises a 20-step action chunk
executed at 20 Hz over `[t,t+1 s)`. Between acquisitions, raw control rows
reference the latest frame rather than writing duplicate PNG files. Because
1 Hz acquisition, 30 Hz rendering and 20 Hz control are asynchronous, the
control loop can observe consecutive camera events 19, 20 or 21 control steps
apart. The recorded camera timestamp is authoritative; this measured one-step
jitter must not be hidden by fabricating frames.

## Model interface

Inputs at each 1 Hz inference point:

- circular RGB, `720 x 1280 x 3`, uint8 PNG;
- aligned depth, `720 x 1280`, uint16 PNG, `0.1 mm/unit`;
- a history of distinct 1 Hz camera frames;
- associated 20 Hz policy state where the selected model permits it;
- episode language/task instruction.

The 31-element policy state contains nine relative joint positions, nine joint
velocities, twelve magnetic-wrench values and one ASM-clearance value.
Privileged simulator fields must not silently become deployment inputs.

Output is a 20-step action chunk. Every step has nine normalized absolute
position offsets in this fixed order:

`j1, j2, j3, j4, j5, j6, ballxj, ballyj, ballzj`

Arm scale is `0.05 rad`, Ball scale is `pi/2 rad`. Commands are offsets around
the reset pose, not deltas integrated from the previous action.

## Raw episode format

Every `steps.jsonl` row is a 20 Hz control transition and contains:

- `control_time_s`;
- `camera_frame_id`, `camera_timestamp_s`, `camera_is_new`;
- paths to the latest RGB and depth frame;
- policy state, command, applied joint target, reward and termination;
- privileged teacher/diagnostic fields.

Only rows with `camera_is_new=true` write image files. Stale rows reuse the
latest paths. An episode is first written as a hidden incomplete directory and
is atomically committed only after successful close.

## Collection

The default recorder now uses the stomach task and v2 dataset root:

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab
./run_isaaclab.sh -p scripts/collect_finetune_dataset.py \
  --episodes 1 --steps 100 --policy scripted_tilt
```

`scripted_tilt` is a deterministic recorder check, not expert training data.
The 30 Hz preview can be opened with `--capsule_camera_view`; it does not
change or enter the 1 Hz recorded observations.

Default output:

```text
datasets/robotarm_magnetic_v2_bringup/
├── dataset.json
├── episodes.jsonl
├── finetune_index.jsonl
└── episodes/<episode_id>/
    ├── episode.json
    ├── steps.jsonl
    ├── rgb/<camera_frame_id>.png
    └── depth/<camera_frame_id>.png
```

## Validation and temporal index

```bash
./run_isaaclab.sh -p scripts/validate_dataset.py \
  datasets/robotarm_magnetic_v2_bringup --check_images

./run_isaaclab.sh -p scripts/build_finetune_index.py \
  datasets/robotarm_magnetic_v2_bringup --history 4
```

The validator checks 20 Hz timestamps, nominal 1 Hz camera strides with the
declared +/-1 control-step tolerance, frame IDs, stale
frame references, vector ranges and image dimensions. The indexer anchors only
on newly acquired frames, uses four distinct 1 Hz frames by default, and takes
the 20-step/1-second horizon from the interface contract.

## Dataset stages

Keep quality levels in separate roots:

1. `robotarm_magnetic_v2_bringup`: scripted interface tests;
2. `robotarm_magnetic_v2_teacher`: reviewed controller demonstrations;
3. `robotarm_magnetic_v2_stomach_nominal`: nominal stomach episodes;
4. `robotarm_magnetic_v2_stomach_randomized`: domain-randomized episodes;
5. `robotarm_magnetic_v2_real`: synchronized physical captures.

Failure episodes remain useful for recovery and safety classification but must
carry explicit labels. Never mix v1 and v2 manifests or treat the 30 Hz preview
as a sensor-rate increase.
