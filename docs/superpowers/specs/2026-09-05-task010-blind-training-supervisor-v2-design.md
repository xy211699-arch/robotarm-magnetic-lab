# TASK-010 Blind-GRU三种子训练监督器V2设计

## 1. 文档地位与目标

本文定义TASK-010视觉依赖性实验中Blind-GRU三个正式种子的独立训练入口和监督协议。它替代
旧视觉依赖性监督器的训练职责，但不删除旧代码、不覆盖旧工件，也不负责任何validation、统计
汇总或工件审计。

本次正式训练采用方案A：2026-09-03创建的旧运行
`20260903T071833.170676Z-ff0184dd`永久保留为试运行，不从其检查点恢复，不纳入正式B1统计。
三个种子必须在同一最终Git HEAD和同一冻结配置下从update 0开始。

## 2. 已确认故障与设计依据

旧试运行包含两类不同问题：

1. `seed_991002`第二次尝试在update 100后抛出`undefined lateral direction at environment rows
   [9]`。该动作执行缺陷已由提交`37d5c22a29177f8ec0042332b1c5eef307986eca`修复为只将受影响
   环境当步降级为HOLD。
2. 第三次尝试从update 100恢复后运行至update 437，最近完整检查点为update 400；指标保持
   有限，但子进程以退出码0提前结束。日志没有记录Python异常、CUDA OOM或非有限值，因此
   无法证明具体外部触发源。现有训练循环只会在完成固定更新数后正常返回，故该现象符合
   Kit或进程生命周期触发`SystemExit(0)`等非`Exception`退出路径。新监督器不得把零退出码
   等同于完成。

旧监督器还存在可观测性缺口：冻结配置中的5分钟停滞和15分钟严重停滞阈值未参与子进程监督；
正式运行manifest没有把统一Git HEAD及工作区身份作为每个种子的持续门禁；TPS下降只展示数值，
没有结构化性能状态。

TPS下降本身不能归因于监督器。旧运行从约38 TPS下降至约1.166 TPS，Isaac日志同时报告CPU
`powersave`，且远程桌面可能影响RTX，但现有证据不能唯一确定性能根因。V2负责识别、记录和
阻止错误完成，不承诺通过代码自动恢复主机性能。

## 3. 冻结训练合同

三个正式种子固定为：

```text
991001
991002
991003
```

每个种子固定使用：

```text
visual_condition = blind
num_envs = 12
rollout_steps = 64
target_update = 1000
checkpoint_interval = 50
device = cuda:0
validation = disabled
```

Blind语义、CNN、GRU、Critic、PPO、奖励、动作、胃部场景、覆盖率、相机、训练split和环境配置
全部沿用现有冻结实现。本任务不得修改这些内容。

## 4. 架构边界

### 4.1 纯监督逻辑

新增：

```text
source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_training_health.py
```

该模块不得导入Isaac Lab。它负责：

- 解析update编号和最近完整检查点；
- 读取最后一条完整JSONL metric；
- 计算最近十个update的TPS中位数和metric年龄；
- 校验全部训练指标有限；
- 判断单种子是否满足严格完成条件；
- 计算恢复所需的剩余update数；
- 生成结构化健康状态和错误分类。

### 4.2 三种子监督CLI

新增：

```text
scripts/stomach_coverage/task010_blind_training_supervisor_v2.py
```

该脚本只负责运行编排，不实现训练算法。它继续以子进程调用现有：

```text
scripts/stomach_coverage/train_task010.py
```

正式训练子进程必须顺序独占GPU，任意时刻最多一个。监督器以独立进程会话启动，启动命令返回
后不依赖调用终端存活，也不要求SSH或tmux。

### 4.3 测试

新增：

```text
tests/stomach_coverage/test_task010_training_health.py
tests/stomach_coverage/test_task010_blind_training_supervisor_v2.py
```

测试使用临时目录和伪训练子进程，不启动Isaac Lab或GPU正式训练。

旧文件`task010_visual_dependence_supervisor.py`保留，仅在文档中标注为legacy，不作为V2正式
训练入口。

## 5. 状态机

顶层状态固定为：

```text
queued
training
paused_on_error
completed_training
```

种子阶段固定为：

```text
queued
running
completed
paused_on_error
```

正常顺序固定为：

```text
991001 → 991002 → 991003
```

已完成种子不会重跑。第三个种子通过严格完成检查后，顶层状态写为`completed_training`，worker
退出，不产生validation命令。

## 6. 启动门禁与运行身份

`start`必须：

1. 拒绝dirty tracked worktree；
2. 记录完整Git HEAD、分支、tracked状态摘要和工作树状态SHA-256；
3. 记录基础训练配置、视觉依赖配置、依赖审计文件的绝对路径、字节数和SHA-256；
4. 验证冻结训练合同中的12环境、64 rollout、1000 updates、50检查点间隔和blind条件；
5. 记录Python、PyTorch、Isaac Lab、驱动、GPU和主机身份；
6. 检查输出目录可写且至少有1 GiB空间；
7. 拒绝同一artifact root下仍处于活动状态的V2运行；
8. 创建新的不可变运行目录，不读取旧试运行检查点；
9. 写入manifest和初始状态后才启动后台worker。

默认工件根目录为：

```text
/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_blind_training_v2
```

worker启动每个种子之前必须重新校验Git和配置身份。运行期间身份发生变化时，禁止启动下一
种子并进入`paused_on_error`。

## 7. 子进程监督与健康状态

后台worker每5秒原子更新`status.json`。训练进度必须从metric和检查点工件读取，而不是从
墙钟推断。

状态至少包含：

- 当前种子和阶段；
- 子进程PID和协调器PID；
- 当前update、目标update和剩余update；
- 最近metric时间与年龄；
- 最新检查点路径、update、字节数和SHA-256；
- 当前TPS与最近十个update的TPS中位数；
- 当前种子及全部运行墙钟时间；
- 基于最近有效TPS计算的动态ETA；
- 健康级别和最近错误摘要。

健康级别为：

```text
starting              尚未产生第一条metric
healthy               正常产生新metric
degraded_performance  仍在推进，但最近十次TPS中位数低于10
suspected_stall       超过300秒没有新metric
critical_stall        超过900秒没有新metric
```

TPS阈值只用于告警，不自动修改参数或终止仍存活的子进程。`critical_stall`也保留进程，由用户
判断是否停止；子进程消失后再依据退出码和工件进入完成或错误状态。

## 8. 严格完成条件

单种子只有同时满足以下条件才能标记`completed`：

1. 子进程退出码为0；
2. 最后一条metric的`update == 1000`；
3. `update_1000.pt`存在且非空；
4. 检查点内部`current_update == 1000`；
5. 所有metric的`all_finite == true`且数值字段不存在NaN/Inf；
6. `runner_initialized`事件中的种子与manifest一致；检查点中的基础配置哈希、视觉依赖配置
   哈希、Git提交和`visual_condition=blind`与manifest一致；
7. Git与配置身份仍与运行创建时一致。

退出码0但未满足这些条件时，错误类型为`incomplete_zero_exit`。例如update 437退出时，状态必须
报告expected 1000、observed 437和最新完整checkpoint 400，不得标记完成或启动下一种子。

其他错误类型至少包括：

```text
child_nonzero_exit
non_finite_metric
checkpoint_missing
checkpoint_corrupt
identity_mismatch
coordinator_stale
```

## 9. 恢复语义

`continue --run-dir <path>`只允许用于`paused_on_error`的V2运行。恢复前必须重新校验运行身份、
当前种子、metric和最新检查点。

若最新有效检查点为update `u`，则：

```text
remaining_updates = 1000 - u
```

恢复命令必须显式传递该检查点和剩余更新数，并保持保存间隔50。恢复过程写入同一正式种子的
追加日志，但每次尝试使用独立控制台日志和事件记录。已完成种子不恢复、不覆盖。

如果Git、训练配置、动作语义或检查点身份变化，V2拒绝继续。用户必须创建全新的正式运行并从
三个种子的update 0开始。

## 10. CLI接口

公开接口固定为：

```text
start       创建新运行并启动后台协调器
status      只读打印一次状态
watch       定期只读打印，Ctrl-C只停止观察
continue    人工检查后恢复当前失败种子
diagnose    只读输出身份、进度、健康和最近日志摘要
```

本任务不提供自动重试、跳过种子、替换种子、修改训练参数或自动validation入口。

## 11. 原子性与并发

- 状态和manifest采用临时文件、`fsync`和`os.replace`原子写入；
- events、metrics和训练日志只追加；
- 运行目录使用文件锁，禁止两个协调器同时操作；
- `start`和`continue`必须拒绝仍存活的旧协调器或训练子进程；
- 三个正式种子不得并发占用GPU；
- `status`、`watch`和`diagnose`均为只读，不加载Isaac Lab。

## 12. 测试与验收

纯测试必须覆盖：

1. 三种子固定顺序和禁止并发；
2. 完整update 1000正常完成；
3. 零退出码但update不足触发`incomplete_zero_exit`；
4. 非零退出；
5. NaN/Inf metric；
6. 检查点缺失、空文件和内部update错误；
7. Git或配置身份变化；
8. `starting`、`healthy`、`degraded_performance`、`suspected_stall`和`critical_stall`；
9. 从update 400恢复时只请求600 updates；
10. 已完成种子不会重跑；
11. 重复start和重复continue被拒绝；
12. 第三个种子完成后不产生任何validation进程。

实现后执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/stomach_coverage/test_task010_training_health.py \
  tests/stomach_coverage/test_task010_blind_training_supervisor_v2.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest tests/stomach_coverage -q

git diff --check
```

不得由Codex启动三个正式种子。代码和纯测试通过后，由用户执行一次短GPU smoke；smoke通过且
工作树重新冻结后，Codex交付正式`start`、`status`、`watch`、`continue`和`diagnose`命令，
正式训练仍由用户启动。

## 13. 完成定义

监督器代码完成的条件是纯测试和全量回归通过、命令帮助可用、未启动正式种子，并形成中文操作
文档。正式训练完成的条件是三个种子均在同一冻结身份下达到update 1000且顶层状态为
`completed_training`。这两个状态不得混写。
