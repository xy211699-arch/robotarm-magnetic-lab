# Codex Agent Handoff — 2026-09-05

## 1. 目的

本文件用于交接当前对话中另一个 Codex Agent 的工作。新代理应继续维护
`feature/TASK-010-visual-dependence-validation` 分支，并接续正在运行的
TASK-010 视觉依赖性验证正式流程。

## 2. Git 状态

- 工作目录：`/mnt/isaac-linux/robotarm_magnetic_lab_task010`
- 分支：`feature/TASK-010-visual-dependence-validation`
- 本地 HEAD：`37d5c22a29177f8ec0042332b1c5eef307986eca`
- 远端 HEAD：`37d5c22a29177f8ec0042332b1c5eef307986eca`
- 当前工作树：干净，无未提交 tracked 文件。
- 该目录是主仓库 `/mnt/isaac-linux/robotarm_magnetic_lab` 的 Git worktree。

## 3. 已完成工作

### 代码实现

- 新增视觉依赖性冻结配置：
  - `configs/task010/visual_dependence_v1.json`
- 新增严格配置加载器：
  - `source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_visual_dependence_config.py`
- 新增视觉干预层：
  - `source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_visual_intervention.py`
- 修改训练观测：
  - `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/task010_terms.py`
- 绑定 checkpoint 与视觉条件：
  - `source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_runner.py`
  - `scripts/stomach_coverage/train_task010.py`
- 扩展验证器和离线特征库：
  - `scripts/stomach_coverage/validate_task010_checkpoint.py`
  - `source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_feature_bank.py`
- 新增统计汇总：
  - `scripts/stomach_coverage/summarize_task010_visual_dependence.py`
- 新增后台监督器：
  - `scripts/stomach_coverage/task010_visual_dependence_supervisor.py`
- 新增门禁检查：
  - `scripts/stomach_coverage/validate_task010_visual_dependence_gate.py`
- 新增/修改测试：
  - `tests/stomach_coverage/test_task010_visual_dependence_*.py`
  - `tests/stomach_coverage/test_task010_feature_bank.py`
  - `tests/parameterized_force/test_vectorized_parameterized_force_action.py`
- 新增文档：
  - `docs/TASK010_VISUAL_DEPENDENCE_AUTOMATION.md`
  - `handoffs/reports/TASK-010-visual-dependence-implementation-report.md`

### 已验证

全仓回归结果：

```text
251 passed, 1 skipped, 49 warnings
```

命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest tests/stomach_coverage -q
```

## 4. 当前正式运行状态

运行目录：

```text
/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/20260903T071833.170676Z-ff0184dd
```

`latest` 指向：

```text
/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/latest
```

当前监督器状态：

```text
state: running
current_stage: train_blind_seed_991002
worker_pid: 762766
child_pid: 762767
```

训练进度：

```text
seed 991001: completed, update 1000/1000
seed 991002: running, 当前约 update 433/1000
seed 991003: queued
```

seed 991001 检查点：

```text
.../seed_991001/checkpoints/update_1000.pt
```

seed 991002 最近检查点：

```text
.../seed_991002/checkpoints/update_0400.pt
```

## 5. 未完成事项

- `train_blind_seed_991002` 尚未完成。
- `train_blind_seed_991003` 尚未开始。
- 所有 update 750 validation 条件尚未开始。
- 所有 update 1000 sensitivity 条件尚未开始。
- `summarize` 尚未运行。
- `audit_artifacts` 尚未运行。
- 正式 V3 尚未完成。

正式流程的最终统计结果不能由代码代理编造，必须在实际运行完成后生成。

## 6. 重要命令

查看状态：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab_task010

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_visual_dependence_supervisor.py status
```

继续训练：

```bash
RUN="$(cat /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/latest_run_path.txt)"

./run_isaaclab.sh -p \
  scripts/stomach_coverage/task010_visual_dependence_supervisor.py continue \
  --run-dir "$RUN"
```

查看 seed 991002 最新指标：

```bash
RUN="$(cat /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/latest_run_path.txt)"

tail -1 "$RUN/training/blind/seed_991002/metrics.jsonl"
```

## 7. 踩过的坑和已修复问题

### launcher 注入 `--kit_args`

`run_isaaclab.sh -p <python_script>` 会自动注入：

```text
--kit_args=--/UJITSO/enabled=false --/UJITSO/geometry=false
```

所有监督器子命令必须接收并忽略 `--kit_args`，否则 argparse 报错。

### B0 checkpoint 文件名格式

正式 B0 文件是：

```text
update_0750.pt
update_1000.pt
```

不是 `update_750.pt`。监督器路径必须使用零填充格式。

### 训练阶段不能只凭退出码标记完成

必须检查：

- `update_1000.pt` 存在；
- 最新 metric 的 `update >= 1000`。

否则用户 kill 后可能被误标成 `completed`。

### 恢复训练要计算剩余 update 数

从 `update_0950.pt` 恢复时，如果仍传 `--max-updates 1000`，会继续跑 1000 次。
监督器现在会传：

```text
--max-updates 50
```

### 垂直轴向的 MOVE/VIEW 未定义方向

原控制器遇到胶囊轴向接近世界垂直方向时会抛：

```text
ValueError: undefined lateral direction at environment rows [9]
```

已改为：

- 只把该环境本步动作降级为 HOLD；
- 其他环境继续；
- 同时更新该环境的 previous action features。

修改文件：

```text
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/vectorized_parameterized_force_action.py
```

### standalone 子脚本缺少源码路径

监督器直接用 Isaac Sim Python 启动子脚本，不经过 `run_isaaclab.sh`。
汇总脚本和门禁脚本需要自行插入：

```python
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
```

### 远程桌面导致 GPU 降速

训练吞吐会从约 38 TPS 突然掉到约 1.16 TPS。

日志中与 `awesun` 远程桌面断线/重连和 NVIDIA 显示输出重新枚举同时出现。

这是当前最大的运行风险。建议：

- 不使用 GUI 远程桌面；
- 改用 SSH + `tmux`；
- 如再发生，停止训练并从最新 checkpoint 恢复。

### 沙箱 PID 隔离

Codex 沙箱看不到宿主机训练进程，运行 `status` 时可能误显示：

```text
coordinator state is stale
```

用户在真实终端运行 `status` 不会受影响。

## 8. 当前建议的下一步

1. 优先确认用户是否仍在通过 `awesun` 远程桌面连接。
2. 如果训练仍很慢：
   - 关闭远程桌面，改用 SSH + `tmux`；
   - 在用户终端 kill 当前训练子进程；
   - 使用 `continue` 从 seed 991002 的 `update_0400.pt` 恢复。
3. 等待三个 B1 seed 全部完成。
4. 让监督器继续 validation、summary 和 audit。
5. 最后核对统计结果，不要提前声称视觉依赖性结论成立。

## 9. 远端同步状态

最后一次推送成功，本地和远端 HEAD 均为：

```text
37d5c22a29177f8ec0042332b1c5eef307986eca
```

如果之后又产生新提交，需要执行：

```bash
git push origin feature/TASK-010-visual-dependence-validation
```
