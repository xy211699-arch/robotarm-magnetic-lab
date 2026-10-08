# 活动任务合同

## 2026-10-08 新胃连续 RL 预验证

当前新胃 RL 预验证交接入口为 `NEW-STOMACH-RL-PREFLIGHT-20261008.md`。完整冻结规格及实施计划分别位于 `docs/superpowers/specs/2026-10-08-new-stomach-rl-preflight-design.md` 和 `docs/superpowers/plans/2026-10-08-new-stomach-rl-preflight-implementation-plan.md`。

规划分支为 `workflow/new-stomach-rl-preflight-20261008`，以发布提交 `1e89c9ccb48545c4513915be94d4c17d9cdebaaa` 为代码基准。Linux 按交接入口在独立 feature 分支完成预验证，不启动正式多种子长训练。本次仅上传文档，尚无新 Gate 执行结果。

## 历史任务登记（原文保留）

当前授权 Linux 执行的任务为
`TASK-009D0A-12-env-incremental-closeout.md`。

Linux 必须从远端 feature 提交
`6dbb8d5bc8c4bbd2d688eb997912d385b697df0a` 继续，获取本合同所在的最新 Windows 规划
提交并纳入现有 feature 分支。不得重建或覆盖既有 TASK-009D0 实现。

TASK-009D0A 只授权十二环境吞吐增量测试、必要的十二环境稳定性验证、两个过期力度测试
修正和 D0 文档收尾。原 Gate 3 失败事实必须保留，最终状态使用用户豁免后的验收表述。

本任务不授权 CNN、GRU、Actor、Critic、PPO、VLM、奖励塑形、扰动范围、控制器重新标定、
覆盖区域变化、位姿库重建或 USD 修改。

Linux 执行状态（2026-08-28）：TASK-009D0A `complete`。TASK-009D0 的原始 Gate 3 结果保持
为 `fail_with_manual_waiver`，D0 总状态按合同收尾为 `accepted_with_manual_waiver`。用户
进一步接受少见非正C0 reset直接中止的运行策略，最终冻结12环境；不重采样或修改位姿库。
完整证据见
`handoffs/reports/TASK-009D0-vectorized-training-infrastructure-report.md`。
