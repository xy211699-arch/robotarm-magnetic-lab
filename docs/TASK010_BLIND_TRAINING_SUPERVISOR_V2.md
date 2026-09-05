# TASK-010 三正式种子训练监督器 V2 操作说明

## 1. 功能边界

本监督器只按固定顺序训练三个 Blind-GRU 正式种子：

```text
991001 → 991002 → 991003 → completed_training
```

固定训练合同为 12 个环境、64 步 rollout、1000 次 update、每 50 次 update 保存检查点、
`cuda:0` 和 `validation=disabled`。监督器不会自动启动验证、汇总或工件审计。

旧运行 `20260903T071833.170676Z-ff0184dd` 只保留作诊断证据，V2 不会读取或续跑它。

## 2. 启动前检查

进入实施工作树：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010
git status --short
nvidia-smi
```

正式启动要求 tracked worktree 为空。这样三个种子会绑定同一个 Git HEAD 和同一组冻结配置；
存在受版本控制的未提交修改时，程序会拒绝启动，不会自动清理文件。

可选的一次 update GPU 冒烟测试如下。它不属于正式三种子运行，输出只写入 `/tmp`：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p scripts/stomach_coverage/train_task010.py \
  --config configs/task010/cnn_gru_development_v1.json \
  --visual-condition blind \
  --seed 991001 \
  --output-dir /tmp/task010_blind_v2_smoke \
  --max-updates 1 \
  --save-interval 1 \
  --validation disabled \
  --dependency-audit /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate0/prerequisites.json \
  --backend isaac \
  --device cuda:0
```

## 3. 正式启动（只执行一次）

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_blind_training_supervisor_v2.py start
```

命令创建后台协调进程后立即返回。终端会输出 `run_id`、绝对 `run_dir` 和 `worker_pid`。
即使关闭当前终端，协调进程仍以独立会话顺序运行；不要再次执行 `start`。

默认工件根目录为：

```text
/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_blind_training_v2
```

当前运行可通过以下两个入口定位：

```text
.../task010_blind_training_v2/latest
.../task010_blind_training_v2/latest_run_path.txt
```

## 4. 查询状态

另开一个终端执行一次性只读查询：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_blind_training_supervisor_v2.py status
```

持续观察（默认每 60 秒刷新一次，`Ctrl+C` 只停止观察，不停止训练）：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_blind_training_supervisor_v2.py watch \
  --interval 60
```

重点字段：

- `state`：`queued`、`training`、`paused_on_error` 或 `completed_training`；
- `current_stage`：当前种子；
- `progress.observed_update`：已写入完整指标行的最新 update；
- `progress.checkpoint_update`：最新完整检查点；
- `progress.median_tps_last_10` 与 `eta_s`：最近十次 update 的中位吞吐和估算剩余时间；
- `progress.health`：`healthy`、`degraded_performance`、`suspected_stall` 或 `critical_stall`。

性能退化和停滞状态只用于告警，不会自动杀死仍存活的训练进程。

## 5. 故障诊断与人工继续

若状态为 `paused_on_error`，先执行只读诊断：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_blind_training_supervisor_v2.py diagnose
```

确认故障原因已消除后，复制 `status` 中的绝对 `run_dir`，只对该运行执行一次：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_blind_training_supervisor_v2.py continue \
  --run-dir /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_blind_training_v2/<RUN_ID>
```

`continue` 不会盲目重启。它要求旧协调器和训练子进程已经退出，Git HEAD、分支、冻结配置与
依赖审计均未变化，并实际加载最新完整检查点核对 update 和实验身份。补训次数严格为
`1000 - checkpoint_update`；已完成种子不会重跑。没有可用检查点时必须重新建立新正式运行。

## 6. 结束判定与证据位置

只有三个种子的最终指标均到达 update 1000、`all_finite=true`，并且各自
`update_1000.pt` 可加载且身份一致，状态才会变成 `completed_training`。

每个运行目录包含：

```text
manifest.json                 # Git、配置、依赖、主机和固定训练合同
status.json                   # 当前权威状态
events.jsonl                  # 追加式生命周期事件
coordinator.log               # 后台协调器输出
logs/train_blind_seed_*/...   # 各种子各次尝试的完整终端日志
training/blind/seed_*/        # 指标、事件和检查点
```

本说明生成时，Codex 只执行了纯单元测试和短时伪训练进程测试，没有启动 991001、991002、
991003 的正式 GPU 训练。
