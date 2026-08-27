# TASK-009C 同步随机基线预实验交接合同

## 执行权威

Windows 规划分支为 `workflow/TASK-009C-synchronous-random-baselines`。

Linux 实现分支必须为 `feature/TASK-009C-synchronous-random-baselines`。

精确代码基线为 `64dd2ff33951cb780f938a81c91c22dde8764c93`，来源是 `origin/feature/TASK-009B-stomach-coverage-environment`。

权威实施方案为 `docs/design/2026-08-27-task009c-synchronous-random-baseline-preexperiment-plan.md`。

要求返回的报告为 `handoffs/reports/TASK-009C-synchronous-random-baseline-preexperiment-report.md`。

## Linux 启动指令

Linux 必须获取 `origin/workflow/TASK-009C-synchronous-random-baselines`，记录获取到的完整规划 HEAD，并确认该规划提交以精确 TASK-009B 基线为祖先。随后从规划 HEAD 创建 `feature/TASK-009C-synchronous-random-baselines`，完整阅读权威实施方案并按 Gate 1 至 Gate 6 的顺序手动执行。

Linux 不需要安装或调用任何 superpowers skill。缺少 `superpowers:subagent-driven-development`、`superpowers:executing-plans` 或其他可选 skill 不构成阻塞，也不能改变任务合同。

## 授权范围

TASK-009C 授权实现指定 validation 位姿先写入再执行一秒 HOLD 的 reset 接口、单环境同步回合运行器、七种随机策略、一个 HOLD 诊断策略、三十七个三百秒正式回合、严格对齐的数据记录、平均覆盖率曲线和结构化返回报告。

TASK-009C 不授权实现 VLM、CNN、GRU、Actor、Critic、PPO、奖励、多环境并行训练、控制器再标定、覆盖 ROI 修改、位姿库重建或仿真资产修改。

## 强制实验规模

七种随机策略必须共享五个 validation 位姿，每种策略每个位姿运行一个三百秒回合，共三十五个回合。HOLD 只在两个固定 validation 位姿各运行一个三百秒诊断回合，共两个回合。正式总数为三十七个回合。

每个回合必须包含正式动作前的 $C_0$ 和三千个十赫兹动作后的覆盖率，因此必须恰有三千零一个严格对齐时间点。七种随机策略各自对五个位姿逐时刻求平均，HOLD 对两个位姿逐时刻求平均，并把八条曲线绘制在同一张高区分度图中。

## 停止与返回规则

执行端必须严格通过纯策略、指定位姿 reset、同步回合、GPU 冒烟、三十七回合正式预实验以及汇总报告六个门禁。前一门禁失败时停止后续工作并返回 `partial`，不得以更换位姿、种子或参数的方式补足结果。

只有缺少外部位姿库或用户拥有但执行端无法读取的冻结配置时返回 `needs_input`。正常的低覆盖、无新增覆盖、受阻和动作效果不明显属于有效实验结果，不能报告为仿真异常。
