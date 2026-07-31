# Robotarm Magnetic Model I/O and Fine-tuning Workflow

## Scope

This document freezes the model-facing interface before the stomach asset is
introduced. Physics remains at 240 Hz, the policy interface runs at 20 Hz, and
the camera renders at 30 Hz. A model never writes rigid-body poses or forces
directly; it emits nine normalized position-offset commands which are executed
by the existing low-level controllers.

The canonical machine-readable contract is:

`configs/interfaces/robotarm_magnetic_v1.json`

Every dataset records the contract SHA-256. Changing an order, shape, unit,
range, camera model, or action meaning requires a new schema version and a new
dataset root.

## Model interface

### Inputs at policy step `t`

- circular RGB: `720 x 1280 x 3`, uint8 PNG on disk;
- aligned metric depth: `720 x 1280`, uint16 PNG with `0.1 mm/unit`;
- policy state: 31 float32 values:
  - 9 relative joint positions;
  - 9 joint velocities;
  - 12 magnetic wrench values;
  - 1 ASM clearance value;
- language/task instruction stored once in episode metadata.

The deployment policy may use RGB only or RGB plus proprioception. Simulator
teacher fields must not accidentally become deployment inputs.

### Outputs at policy step `t`

Nine float32 values in `[-1, 1]`, ordered:

`j1, j2, j3, j4, j5, j6, ballxj, ballyj, ballzj`

These are **absolute normalized offsets around the reset pose**, not deltas
integrated from the previous action. Arm scale is `0.05 rad`; Ball scale is
`pi/2 rad`. The default fine-tuning target is an eight-step action chunk
covering 0.4 seconds.

## Repository tree

```text
robotarm_magnetic_lab/
├── configs/
│   └── interfaces/
│       └── robotarm_magnetic_v1.json       # versioned machine contract
├── docs/
│   └── TRAINING_DATA_WORKFLOW.md           # this workflow
├── scripts/
│   ├── collect_finetune_dataset.py         # Isaac Lab episode capture
│   ├── build_finetune_index.py             # history -> action chunks
│   └── validate_dataset.py                 # integrity validation
├── source/robotarm_magnetic_lab/
│   └── robotarm_magnetic_lab/
│       └── io/
│           ├── schema.py                   # runtime schema checks
│           └── episode_writer.py           # atomic recorder
└── datasets/                               # generated; ignored by Git
    └── robotarm_magnetic_v1/
        ├── dataset.json                    # dataset + complete interface
        ├── episodes.jsonl                  # committed episode catalog
        ├── finetune_index.jsonl            # temporal training samples
        └── episodes/
            └── YYYYMMDD_HHMMSS_e0000/
                ├── episode.json
                ├── steps.jsonl
                ├── rgb/000000.png
                └── depth/000000.png
```

An episode is first written as `.episode_id.incomplete` and atomically renamed
only after all files and metadata are complete. Interrupted collection cannot
silently enter training.

## Collection

Run a short deterministic recorder check:

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab
./run_isaaclab.sh -p scripts/collect_finetune_dataset.py \
  --task Template-Robotarm-Magnetic-Lab-v0 \
  --episodes 1 \
  --steps 20 \
  --policy scripted_tilt
```

`scripted_tilt` validates the pipeline; it is not an expert demonstration.
Later, replace the action source with a collision-safe magnetic teacher,
teleoperation, or a trained state policy without changing the writer.

## Validation

```bash
./run_isaaclab.sh -p scripts/validate_dataset.py \
  datasets/robotarm_magnetic_v1 \
  --check_images
```

Validation checks committed counts, continuous timestamps, vector dimensions,
finite values, action range, image existence and image dimensions.

## Build temporal fine-tuning samples

```bash
./run_isaaclab.sh -p scripts/build_finetune_index.py \
  datasets/robotarm_magnetic_v1 \
  --history 4 \
  --horizon 8 \
  --stride 1
```

Each row contains four past RGB-D frames and policy states, plus the following
eight action commands and applied joint-position targets. Data loaders can read
this JSONL directly or convert it to a framework-specific shard format later.
The raw episode data remains the source of truth.

## Synchronization convention

One `steps.jsonl` row represents:

1. observation and privileged teacher state at time `t`;
2. model/teacher command selected from that observation;
3. actual low-level joint-position target written for the transition;
4. reward and termination produced by transition `t -> t+1`.

`sim_time_s` is the policy timestamp. Camera frames are the latest available
30 Hz images when the 20 Hz policy step begins. Physics and magnetic wrenches
continue at 240 Hz.

## Dataset stages

Keep separate dataset roots rather than mixing quality levels:

1. `robotarm_magnetic_v1_bringup`: scripted interface tests;
2. `robotarm_magnetic_v1_teacher`: state-controller demonstrations;
3. `robotarm_magnetic_v1_stomach_nominal`: nominal stomach episodes;
4. `robotarm_magnetic_v1_stomach_randomized`: domain-randomized episodes;
5. `robotarm_magnetic_v1_real`: time-synchronized physical-system captures.

Only successful, reviewed teacher/teleoperation episodes should be used as
positive behavior-cloning demonstrations. Failure episodes remain useful for
recovery learning and safety classification but must carry explicit labels.

## Next stage: stomach environment

The stomach scene must conform to this interface rather than redefining it.
It may add teacher-only contact, surface-normal, goal and segmentation fields.
Any new deployment input requires schema `1.1.0` or `2.0.0`, depending on
whether the change is backward-compatible.
