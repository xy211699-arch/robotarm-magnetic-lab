# TASK-005：十一动作动态控制器

## 授权状态

用户已于 2026-08-18 完成逐项需求确认并授权实施。

Windows 规划分支：workflow/TASK-005-eleven-action-dynamic-controller

Linux 实施分支：feature/TASK-005-eleven-action-dynamic-controller

已验收实现基线：87a80adcc367a3210fc1f8cfadea410f340e3918

权威设计：docs/superpowers/specs/2026-08-18-eleven-action-dynamic-controller-design.md

权威计划：docs/superpowers/plans/2026-08-18-eleven-action-dynamic-controller.md

回传报告：handoffs/reports/TASK-005-eleven-action-dynamic-controller-report.md

## 分支和启动要求

Linux 必须 fetch Windows 规划分支，记录远端精确 head，确认其历史包含上述 TASK-004 基线，
然后从该规划 head 新建隔离 feature 分支。不得从 main、旧 TASK-004 report head 或其他本地
分支开始，不得在 main 实施或合并。

开始编辑前必须完整阅读 AGENTS.md、本合同、权威设计、权威计划和 TASK-004 最终报告。本文
计划可直接执行，不要求 Linux 安装或调用 superpowers、executing-plans、
subagent-driven-development、插件、子代理或任何外部编排能力。缺少这些工具不构成偏差。

## 实施范围

只实现固定一秒十一动作动态控制器、Isaac Lab 平面和胃部任务、键盘连续可视化、平面随机
定量验收、100 动作压力测试、胃部无适配迁移、回归、文档和证据。

动作 ID 固定为：0 HOLD_VIEW；1 VIEW_UP；2 VIEW_UP_RIGHT；3 VIEW_RIGHT；
4 VIEW_DOWN_RIGHT；5 VIEW_DOWN；6 VIEW_DOWN_LEFT；7 VIEW_LEFT；8 VIEW_UP_LEFT；
9 MOVE_SIDE_POS；10 MOVE_SIDE_NEG。不得增加、删除、重新编号或加入 action mask。

不实现或修改 VLM、时序 Actor、Actor-Critic、奖励、覆盖率、action chunk、教师、专家轨迹、
数据集或训练。只可保留未来 action result 接口，不得把胶囊或胃壁真值加入 policy observation。

## 明确授权的真值边界

本任务明确允许仿真 controller、动作结果和 validator 持续读取胶囊真实位置、姿态、线速度、
角速度、接触点、接触力、胃壁 mesh 和局部法向。这是对仓库默认真值限制的任务内例外，仅为
仿真专用理想闭环执行器服务，不代表未来 Actor 可读取这些信息。

## 动力学硬约束

胶囊必须保持启用重力的非运动学动态刚体。正常 reset 之外，只能向胶囊施加世界系 COM force
和 torque，并由 PhysX 积分。严禁 pose、orientation、transform、linear velocity、angular
velocity 写入，严禁 kinematic switch、teleport、projection、surface snap、隐藏恢复或直接
姿态设置。

不得调用机械臂、ASM、磁体或磁控制；不得修改资产、胶囊几何、质量、惯量、重力、材料、
摩擦、恢复系数、CCD、求解器、既有 reset 或相机标定。不得做空间余量、预测碰撞、净空或
避障。

物理固定 240 Hz。除 FAULT 外，每个动作严格运行 1.000 模拟秒和 240 子步。VIEW 使用 0.8 秒
quintic swing 加 0.2 秒保持；MOVE 使用 0.25 秒自由、0.5 秒固定 COM force、0.25 秒自由。
动作边界保持闭环且不插入零 wrench，MOVE 内部自由段除外。

## VIEW 与支撑

九宫格相对当前相机 frame，外围光轴目标统一为 15 度圆锥半角，最小 swing 且不规划额外
长轴 twist。动作开始冻结相机 frame、局部 surface normal、底部支撑材料点和其世界切向
anchor，240 Hz 真值反馈通过支撑点等效 COM wrench 和姿态 torque 执行。

相机半球真实碰壁后取消继续朝墙内转向，冻结接触时光轴并保持到一秒结束。正常碰壁、有限
振荡和滑动返回 COMPLETED，不是 REJECTED 或 FAULT。持续接触方向可内部 constrained HOLD，
但不得形成 Actor mask 或 VIEW_BLOCKED 标签。

## MOVE 与状态

MOVE 只在动作开始检查并锁存：倾角至少 60 度，并且最近 0.05 秒，即 12 个物理子步中至少有
一次圆柱侧壁接触。不得要求最小接触力、多点接触、分离距离或稳定比例。失败时完整执行一秒
HOLD 并返回 REJECTED，不得自动倾倒。

accepted MOVE 不保持姿态或支撑。0.25 秒时按局部内法向和长轴切向投影计算并冻结正负方向，
施加 \(\mathbf F=kmg\mathbf d\)。平面按 0.9 至 3.0、步长 0.1 选择 POS 和 NEG 同时通过的
最小共享 \(k\)，随后冻结并原样用于胃部。实际滚动、滑动和倾角只记录。

COMPLETED 包含正常碰壁、有限振荡、低效果移动和方向退化。REJECTED 只用于 MOVE 前置条件
失败。FAULT 只用于 NaN/Inf、程序异常、刚体状态丢失、状态机损坏或不可恢复数值错误；轻微
胃壁不稳定不得误报 FAULT。

## 键盘、渲染和验收

数字小键盘按九宫格映射，5 为 HOLD，Q 为 MOVE_SIDE_NEG，E 为 MOVE_SIDE_POS。一次按下触发
一个动作；执行期间按键丢弃，不缓存、不排队、不抢占；完成后 READY_HOLD 持续仿真和渲染，
等待用户观察和下一次按键。无 HUD，结果只在终端打印。

默认连续渲染调度目标 120 FPS，CLI 可选 60、120、240；动作一秒按模拟时间判断，不保证墙钟
一秒。记录实际 wall FPS。

平面每个动作分别至少 10 个分层随机状态。每个 VIEW 至少 10 个未碰壁有效样本，末端偏转
15±3 度、支撑切向漂移不超过 2 mm；碰壁样本单列。HOLD 至少 10 个。每个 MOVE 至少 10 个
valid 和 10 个 invalid；valid 至少 90% 的请求方向 signed displacement 达到 5 mm。另运行
无 reset 的固定 100 动作压力测试，每个 ID 至少出现 5 次。

平面全部通过后，使用相同 controller、相同参数和相同 profile digest 迁移胃部。胃部不做
5 mm 二次门禁，不允许任何适配；运行同一 100 ID 序列和键盘可视化，主观效果由用户后续
判断。

## 交付与停止条件

Linux 必须提交可复现实验报告，包含 base/head/branch、命令、样本和指标、状态计数、MOVE
\(k\)、flat/stomach digest、渲染信息、回归、偏差、未验证声明及外部证据路径、字节数和
SHA-256。大日志、视频、数据和快照不得进入正常 Git 历史。

若 VIEW 授权网格无通过者，或 MOVE 在 \(k=3.0\) 仍失败，应返回 needs_decision，不得扩展
动作时长、角度、力、力矩或修改物理。若平面通过但胃部无法初始化、渲染或出现系统 FAULT，
返回 partial。若平面、同 digest 胃部运行和回归全部完成，返回 complete；正常胃壁阻挡和
低效果只作为观察结果，不触发胃部调参。

推送 feature/TASK-005-eleven-action-dynamic-controller，不得合并 main。
