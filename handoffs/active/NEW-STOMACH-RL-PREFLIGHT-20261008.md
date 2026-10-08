# 新胃连续 RL 预验证交接入口

## 交付内容与版本

本任务交接用户在 2026-10-08 提供的冻结规格和实施计划。两份文件正文保持原样；规格按计划中的引用路径登记为 design.md。

冻结规格：`docs/superpowers/specs/2026-10-08-new-stomach-rl-preflight-design.md`。

实施计划：`docs/superpowers/plans/2026-10-08-new-stomach-rl-preflight-implementation-plan.md`。

代码基准为 `release/new-stomach-magnetic-v1-20261008`，提交 `1e89c9ccb48545c4513915be94d4c17d9cdebaaa`。Windows 文档交接分支为 `workflow/new-stomach-rl-preflight-20261008`；Linux 实施分支按计划使用 `feature/new-stomach-rl-preflight-20261008`。本规划提交仅包含文档。

## Linux 接手方式

在保留现有改动和运行工件的前提下获取 origin。以本规划分支为起点，在独立工作区建立上述 Linux 实施分支，从 Gate 0 开始逐项核验并按规格执行。不得从旧 TASK-010 规划分支直接继续新胃实现，不得直接向 main 推送。

```bash
git fetch origin
# 在独立工作区，以 origin/workflow/new-stomach-rl-preflight-20261008 为起点
git switch -c feature/new-stomach-rl-preflight-20261008 origin/workflow/new-stomach-rl-preflight-20261008
```

## 执行范围

本任务仅交付新胃 RL 集成、覆盖奖励正确性、短程 PPO 可训练性及并行性能预验证。Single 为 9D×1s，Chunk 为 4×9D×0.25s 且四段全部执行；物理 240 Hz，RGB/策略 1 Hz，几何覆盖主奖励 10 Hz，有效距离 70 mm。具体约束、正确性门禁和停止条件以两份完整文档为准。

新胃位姿库必须定位用户已确认的真实文件并核对几何、split 和哈希；找不到时按文档返回所需输入。不得替换成旧胃位姿库，不启动正式四组多种子长训练。本次 Windows 上传没有执行实施计划、仿真、GPU smoke 或训练。

Linux 结果报告路径为 `handoffs/reports/NEW-STOMACH-RL-PREFLIGHT-20261008.md`，包括 base/head、分支、实际命令、逐 Gate 观察结果、偏差、未验证内容、外部工件路径与哈希。报告只能陈述实际执行证据。
