# Robotarm Magnetic Lab

Isaac Lab 3.0 single-environment task for the AUBO-style six-axis arm, the
three-axis magnetic ball assembly, and the external capsule magnet.

## 当前归档主版本：A版（新胃模型联合磁控）

用户于2026-10-08确认选择A版，归档分支为
`release/new-stomach-magnetic-v1-20261008`。版本入口、文件索引、依赖与限制见
[`docs/RELEASE_A_NEW_STOMACH_MAGNETIC.md`](docs/RELEASE_A_NEW_STOMACH_MAGNETIC.md)。

- 任务：`Template-Robotarm-Magnetic-New-Stomach-Vector-Lab-v0`。
- 控制：240 Hz物理、1 Hz动作；4×9相对增量chunk，首秒执行、后三秒预览。
- 胶囊由真实计算的磁力/磁矩、重力和接触驱动，不采用MOVE/VIEW/UP直接施力。
- 新胃双管口水平；不可达区是候选掩码，尚未启用正式新胃覆盖率或训练。
- 旧开环演示与TASK-010直接施力训练代码保留，不能与A版入口或指标混用。
- 本仓库仍依赖外部Isaac Lab/Isaac Sim安装和机器人原始USD资产；不是独立安装包。

下面保留的bring-up、训练和数据接口说明属于历史入口，不代表A版控制时序。

## Project status and execution history

- Incremental work since the 2026-07-17 handover:
  [`docs/POST_HANDOVER_WORK_SUMMARY.md`](docs/POST_HANDOVER_WORK_SUMMARY.md)
- Concise per-conversation execution log:
  [`docs/PROJECT_RUN_LOG.md`](docs/PROJECT_RUN_LOG.md)

## Current bring-up status

- Task ID: `Template-Robotarm-Magnetic-Lab-v0`
- Workflow: Manager-Based, single agent
- Physics: 240 Hz
- Low-level control: 20 Hz (`decimation=12`); VLA inference: nominal 1 Hz
- Action: 9 absolute normalized offsets about the reset pose in this exact order:
  `j1..j6, ballxj, ballyj, ballzj`
- State observation: 31 values: nine relative joint positions, nine joint
  velocities, 12 magnetic wrench values, and one ASM clearance value
- Vision observation: capsule-mounted `1280x720` circular RGB and aligned
  metric depth; inactive pixels outside the optical circle are zero
- Capsule camera: provisional DS01/CX93510-series model, 120-degree horizontal
  circular FOV, 1 Hz policy acquisition plus a separate 30 Hz engineering
  preview, local `+Z` optical direction, equidistant
  wide-angle remap, soft optical border and nominal edge lens shading
- Illumination: four 5600 K close-range LEDs mounted around the camera axis
- Reset: directly restores the validated initialization pose; it does not
  replay the old Script Editor trajectory
- Asset source: `/home/multirobo/Desktop/sim of FF/Stage.usd`

## TASK-010 visual-dependence validation

The visual-dependence implementation is complete and awaits manual V3 execution. Do not start
the formal three-seed experiment from Codex.

Manual start, read-only status, watch, and error-recovery commands are documented in
[`docs/TASK010_VISUAL_DEPENDENCE_AUTOMATION.md`](docs/TASK010_VISUAL_DEPENDENCE_AUTOMATION.md).

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

`configs/interfaces/robotarm_magnetic_v2.json`

It freezes image shapes, units, state layout, joint order, action semantics and
the asynchronous 1 Hz camera / 20 Hz control / 240 Hz physics rates. The 30 Hz
camera window is preview-only. The episode recorder, integrity validator and temporal
fine-tuning index are documented in:

[`docs/TRAINING_DATA_WORKFLOW.md`](docs/TRAINING_DATA_WORKFLOW.md)

Generated data belongs under `datasets/` and is intentionally ignored by Git.
Do not place recorded images or episode JSONL files in `logs/` or source
directories.
