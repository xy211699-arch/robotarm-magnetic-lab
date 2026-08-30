# TASK-010 三正式种子自动训练与验证

## 功能边界

`task010_formal_seed_supervisor.py`只负责按固定顺序协调以下流程：

```text
训练991001 → 验证991001
→ 训练991002 → 验证991002
→ 训练991003 → 验证991003
→ 汇总三个种子
```

正式合同固定为12环境、64步rollout、1000次更新、每50次更新保存检查点。三个种子的训练和验证始终顺序使用`cuda:0`，协调器不会并发启动两个GPU子进程。

## 单次启动

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_formal_seed_supervisor.py start
```

启动命令创建独立会话中的后台协调进程后立即返回。关闭终端、断开SSH或结束Codex对话不会停止后台流程。

## 只读状态查询

```bash
./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_formal_seed_supervisor.py status
```

状态包含当前种子、训练或验证阶段、训练update、验证位姿进度、最近检查点、运行时长、心跳年龄和错误摘要。`status`不修改状态文件。

## 错误后人工继续

流程遇到非零退出、非有限覆盖、陈旧状态、检查点不匹配、位姿不足20个或轨迹不足1201点时进入`paused_on_error`，并停止启动后续种子。

用户检查并处理错误后执行：

```bash
./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_formal_seed_supervisor.py continue
```

`continue`只重试当前暂停种子的失败阶段，不跳过种子、不删除原有日志、不自动选择替代检查点，也不自动恢复训练检查点。

## 输出目录

默认根目录：

```text
artifacts/task010_cnn_gru/formal_seeds/
```

每个种子保留：

- `training/checkpoints/update_1000.pt`；
- `validation/pose_records.jsonl`；
- `validation/coverage_trajectories.jsonl`；
- `validation/mean_coverage.csv`。

三个种子全部验证成功后生成：

```text
summary/formal_three_seed_mean_coverage.csv
```

CSV包含1201行10 Hz时序数据，以及三个种子均值、三种子总体均值和种子间总体标准差。
