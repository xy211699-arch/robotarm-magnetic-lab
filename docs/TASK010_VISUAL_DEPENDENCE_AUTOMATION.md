# TASK-010 视觉依赖性验证后台运行说明

## 状态

本任务只交付实现、测试、CPU self-check 门禁和监督入口。Blind-GRU 三种子训练已经完成，
监督器会先审计并冻结既有 B0、B1 检查点，然后跳过三个 Blind-GRU 训练阶段，直接进入 B0
正常视觉、跨回合错配和首帧冻结推理。正式 V3 尚未启动，必须由用户人工执行 `start`。

## 正式启动

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_visual_dependence_supervisor.py start \
  --config configs/task010/visual_dependence_v1.json \
  --b0-run-dir /mnt/isaac-linux/robotarm_magnetic_lab_task010/artifacts/task010_cnn_gru/formal_seeds/20260830T124744.667141Z-fcf8b406 \
  --b1-run-dir /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_blind_training_v2/20260905T055703.090461Z-ccadb75b
```

`start` 只创建唯一运行目录、启动后台协调器、打印 `run_id`、绝对路径和 PID 后立即返回。
使用 `--b1-run-dir` 后，第一项实际计算阶段固定为
`validate_update750_normal_seed_991001`，不会再次训练 Blind-GRU。若 B0/B1 状态文件、完成审计、
检查点路径或哈希不一致，启动会在安全位置拒绝执行。

## 只读状态

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_visual_dependence_supervisor.py status
```

## 只读观察

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_visual_dependence_supervisor.py watch --interval 60
```

`watch` 可以随时 Ctrl-C 中断，不影响后台 worker。

## 错误后人工继续

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

TASK010V_RUN_DIR="$(cat /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/latest_run_path.txt)"

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_visual_dependence_supervisor.py continue \
  --run-dir "${TASK010V_RUN_DIR}"
```

`continue` 只允许对持久化为 `paused_on_error` 的运行目录显式执行；不自动重试、不跳过阶段、
不更换种子。
