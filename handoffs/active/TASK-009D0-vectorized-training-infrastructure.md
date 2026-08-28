# TASK-009D0 向量化训练基础设施交接合同

## 执行权威

Windows 规划分支为 `workflow/TASK-009D0-vectorized-training-infrastructure`。

Linux 实现分支必须为 `feature/TASK-009D0-vectorized-training-infrastructure`。

精确代码基线为 `7c4c5a18780b980ad3882ce75f1d64733fc3080d`，来源是远端
`feature/TASK-009C-synchronous-random-baselines` 的最终 HEAD。Windows 设计提交
`f8eb6b825aa8e5765b3db52532b169a9d299066e` 必须是 Linux 获取到的规划 HEAD 的祖先。

权威设计为
`docs/design/2026-08-28-task009d0-vectorized-training-infrastructure-design.md`。

权威实施计划为
`docs/superpowers/plans/2026-08-28-task009d0-vectorized-training-infrastructure.md`。

版本化研究合同为 `docs/vlm_gastric_coverage_research_contract_v1.md`。

要求返回的报告为
`handoffs/reports/TASK-009D0-vectorized-training-infrastructure-report.md`。

## Linux 启动规则

Linux 必须获取远端规划分支，记录完整规划 HEAD，确认代码基线和设计提交均为该 HEAD 的
祖先，再从规划 HEAD 创建规定的 feature 分支。Linux 必须完整阅读设计、研究合同和实施
计划，并按 Task 1 至 Task 12、Gate 1 至 Gate 6 的顺序执行。

Linux 当前缺少 superpowers skill 不构成阻塞。执行端应手动遵循实施计划中的 TDD、逐任务
测试、逐任务提交和 Gate 停止规则，不得因为缺少可选 skill 改变任务语义。

## 授权范围

TASK-009D0 授权新增独立多环境任务、训练候选配置、split 安全位姿批次、批量参数化力、
逐环境 RGB 帧同步、GPU 精确批量可见性、逐环境面积覆盖状态、同步 120 秒回合、原始新增
覆盖奖励接口、Actor 与特权观测隔离、单环境等价验证、双环境串扰验证、GPU reset 验证、
1/2/4/8 环境吞吐实测、并行数冻结、两个完整 120 秒长时回合和结构化返回报告。

TASK-009D0 只允许新增任务和必要的共享纯函数。旧单环境任务 ID、旧随机基线运行器和旧
覆盖运行时必须继续通过回归。不得删除、改名或把旧入口重定向到新实现。

## 不授权范围

本任务不授权 CNN、GRU、Actor 网络、Critic 网络、PPO、RSL-RL 训练配置、Beta 动作分布、
奖励缩放、持续停滞惩罚、VLM、扰动范围、课程学习、位姿库重建、不可达区域重标定、控制器
力度重标定或任何 USD 资产修改。

胃部几何、胶囊质量惯量、碰撞、摩擦、重力、相机内参、1280×720 分辨率、120 度圆形
视场、70 mm 可见距离、面积权重、不可达区域、入口位姿库、动作方向、作用点和冻结力度
均不得为了吞吐或测试结果改变。

如果多环境复制确实要求修改 USD 结构，Linux 必须在接触资产前返回 `needs_decision`，说明
非破坏性 overlay 或派生资产方案及其物理风险。未经新的 Windows 合同不得修改资产。

## 固定回合与信息边界

每个正式回合为 120 秒、1200 个 10 Hz 动作边界和 28800 个正式物理子步。正式动作之前
必须统一写入各环境位姿、清零速度和残余力、执行十个 HOLD 边界、采集最后 RGB 并分别
计算严格大于零的 $C_0$。每环境每回合恰有 1201 个正式覆盖点。

Actor 观测只含 10 Hz RGB 和上一实际动作。上一动作固定为六维 one-hot 模式加一维力度，
HOLD 力度为零。位姿、速度、接触、法向、覆盖、位姿 ID、split、环境编号和 RNG 状态不得
进入 Actor 路径。

训练模式只能加载 1000 个 train 位姿。Gate 2 可以在显式 validation 模式下解析五个冻结
validation ID，但不得把这些 ID 接入训练采样器。测试 split 不参与 TASK-009D0 的模型选择
或性能比较。

## 门禁顺序

Gate 1 包含版本与设备核验、配置、split 位姿采样、批量动作、批量覆盖、相机同步、环境
注册和全部纯测试。Gate 1 失败时不得进入 live 仿真。

Gate 2 使用一个新环境与旧标量数学逐边界对照五个冻结 validation 位姿。当前可见与累计
布尔掩码必须逐顶点完全相同。任何布尔差异都失败，不允许用浮点容差掩盖。

Gate 3 使用两个环境执行相同动作一致性、分叉动作和逐行状态清空，证明未操作环境的局部
物理、覆盖、帧、上一动作、奖励和 RNG 账本不变。

Gate 4 使用两个不同 train 位姿完成二十次同步 reset，验证正确写入顺序、十个连续 RGB、
GPU 设备、有限图像、正 $C_0$、零正式 episode length 和零残余 Actor 力。

Gate 5 对 1、2、4、8 个环境各运行三次独立 Isaac Sim 进程。候选必须零故障且至少保留
20% GPU 显存。按总体环境转换吞吐选择；吞吐差异不超过 10% 时选择环境数更少者。只有
离线汇总程序可以写入冻结并行数配置。

Gate 6 使用冻结并行数连续运行两个完整 120 秒回合。每环境每回合必须恰有 1201 个覆盖点、
28800 个正式子步和回合间 240 个 HOLD 子步，第二回合不得继承第一回合覆盖或动作状态。

前一 Gate 失败时停止后续 Gate。不得为了继续运行降低图像分辨率、改变覆盖语义、删除
困难位姿、更换种子或接受近似掩码。

## 故障与有效结果

普通碰撞、受阻、短时静止、零新增覆盖、HOLD、低覆盖和 120 秒超时是有效结果。非有限
刚体状态、RGB 缺失或非有限、帧号错位、射线异常、累计覆盖下降、未定义侧向方向、PhysX
异常或不可恢复渲染故障属于正式故障，并终止整个同步批次。

本任务不允许通过换位姿、换种子或重跑成功样本补足失败批次。执行进程意外退出后只能用
同一配置和种子复现故障；第二次观察仍失败时停止相应 Gate 并报告 `partial`。

## 工件与报告

外部工件根目录固定为
`/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/`。逐帧日志、RGB、
掩码、吞吐原始数据和临时结果不得进入 Git。每个 Gate 的稳定清单必须记录绝对路径、字节数
和 SHA-256。

最终报告必须包含代码基线、规划 HEAD、实现分支、实现 HEAD、实际软件与设备版本、修改
文件、全部命令、自动测试计数、Gate 1 至 Gate 6 直接观测结果、四个候选的吞吐与显存、
最终并行数、偏离项、未验证项和全部关键外部工件证据。

只有全部 Gate 通过、冻结配置已提交、两个长时回合完整且远端 feature HEAD 与本地一致时，
Linux 才能返回 `complete`。缺少用户持有的外部位姿库时返回 `needs_input`。实现或环境门禁
失败返回 `partial`。需要改变架构、资产、安全边界或研究合同才能继续时返回
`needs_decision`。
