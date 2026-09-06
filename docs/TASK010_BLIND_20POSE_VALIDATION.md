# TASK-010 Blind-GRU 三种子 20 位姿验证指南

## 用途

本入口依次验证 Blind-GRU 的三个正式种子 `991001`、`991002`、`991003` 的
`update_1000.pt`。每个种子均在冻结的 20 个 validation 位姿上执行 120 秒确定性验证，
三个验证进程严格串行使用同一张 GPU。

程序只做验证，不训练模型，也不修改检查点。启动命令创建后台协调进程后立即返回；关闭当前
终端不会停止验证。

## 启动

在项目目录执行：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_blind_validation_supervisor.py start \
  --source-run /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_blind_training_v2/20260905T055703.090461Z-ccadb75b
```

启动前必须保证已跟踪文件没有未提交修改。未跟踪实验目录不会阻止启动。程序会核对三个训练
阶段均已完成、最终检查点均为 `update_1000.pt`，并校验检查点 SHA-256。

不要重复执行 `start`。启动输出中的 `run_dir` 是本次验证的唯一运行目录，应保留用于后续查询。

## 查询状态

另开终端执行：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_blind_validation_supervisor.py status
```

默认查询 `artifacts/task010_blind_validation_v2/latest`。也可查询指定运行：

```bash
./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_blind_validation_supervisor.py status \
  --run-dir /绝对路径/到/本次/run_dir
```

主要状态为：

- `queued`：等待后台协调器开始；
- `validating`：正在验证当前种子；
- `paused_on_error`：发生错误并暂停，不会跳过当前种子；
- `completed`：三个种子均通过 20 位姿工件审计。

状态中 `current_seed` 表示当前种子，`validation_pose_progress` 表示当前种子已落盘的完整位姿数。

## 错误后继续

先阅读 `error_summary`、`coordinator.log` 和对应的
`seed_<seed>_validation_attempt_<序号>.log`，确认问题已经解决且旧进程已退出，再执行：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_blind_validation_supervisor.py continue \
  --run-dir /绝对路径/到/本次/run_dir
```

`continue` 会复核 Git、配置和检查点身份。已经通过完整工件审计的种子不会重跑；不完整种子会
重新执行，程序不会把残缺文件误判为完成。

## 输出内容

默认根目录：

```text
/mnt/isaac-linux/robotarm_magnetic_lab_task010/artifacts/task010_blind_validation_v2/
```

每个种子目录包含：

```text
seed_<seed>/
├── pose_records.jsonl
├── coverage_trajectories.jsonl
├── telemetry_10hz.jsonl
├── final_masks/<pose_id>.npz
├── mean_coverage.csv
├── summary.json
└── validation_audit.json
```

`telemetry_10hz.jsonl` 对每个位姿保存 1201 个状态/覆盖点和 1200 个动作/奖励点，包括胶囊
位置、`wxyz` 四元数、线速度、角速度、动作模式、力度和奖励。最终覆盖掩码来自环境自动 reset
前的终止快照。

三个种子全部通过后，运行根目录会生成：

```text
three_seed_mean_coverage.csv
```

其中包含三个种子的平均覆盖曲线，以及三种子总体均值和种子间标准差。程序不会自动绘图。
