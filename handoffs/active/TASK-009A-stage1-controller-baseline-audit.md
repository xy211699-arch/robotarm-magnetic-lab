# TASK-009A 第一执行阶段：当前控制器基线发布与审计

## 执行权威

Windows 规划分支为 `workflow/TASK-009A-stage1-controller-baseline-audit`。Linux 返回分支固定为 `feature/TASK-009A-stage1-controller-baseline-audit`，返回报告固定为 `handoffs/reports/TASK-009A-stage1-controller-baseline-audit-report.md`。

权威研究合同为 `docs/design/2026-08-25-vlm-gastric-coverage-research-contract-v1.md`，权威执行清单为本文件。Linux 必须先获取并记录 Windows 规划分支的完整提交哈希，再完整阅读两个文件。

Windows 规划分支只承载合同，不是高频控制器代码基线。Linux 不得从该规划分支重新实现控制器；Linux 返回分支必须从其当前已经完成且实际运行过的高频控制器提交创建，并在报告中记录该来源提交。

## 任务性质

本任务是研究合同第一执行阶段的零号门槛。任务只发布、核对和记录 Linux 端已经完成的高频参数化控制器，不修改控制规律，不开始 VLM、GRU、PPO、奖励或随机策略实验。

Linux 端不需要安装或调用任何 Codex skill、superpowers 或插件。所有步骤按本文档手工执行并保存证据。

## 任务目标

Linux 端必须把当前实际可运行的控制器整理为一个可追踪的 Git 提交，并证明它实现的是研究合同中的 $10\,\mathrm{Hz}$ 参数化动作接口，而不是旧 TASK-008 的 $1\,\mathrm{Hz}$ 一秒宏动作。

返回结果必须给出唯一的分支名和完整提交哈希。后续面积覆盖、入口区域、同步环境和随机策略基线只能从该提交继续，不能从旧 TASK-008 规划分支推测或重写控制器。

## 已确认的旧基线冲突

当前 Windows 端能够验证的 `origin/feature/TASK-008-six-action-dynamic-force-controller` 头为 `a782f475f6a8b94c98c429ee7cb22a37a95dc2bf`。该分支仍使用 Actor $1\,\mathrm{Hz}$、一秒动作宏、MOVE/VIEW 的等待—施力—等待时序、50 mm 顶点覆盖以及双端 UP 力偶，因此不符合当前研究合同。

Linux 不得把上述提交声明为当前高频控制器。若 Linux 当前工作区也只包含该实现，应返回 `needs_input`，不得继续面积覆盖或强化学习环境开发。

## 固定发布分支

Linux 应从当前已经完成且实际可运行的高频控制器提交创建以下分支：

```text
feature/TASK-009-stage1-controller-baseline
```

若当前实现尚有未提交修改，Linux 只能提交与高频参数化控制器直接相关的文件。不得顺带提交生成日志、数据集、视频、截图、缓存、USD 资产修改或其他任务的代码。

## 执行前取证

Linux 在任何整理操作之前运行以下命令，并把完整输出写入返回报告：

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline --decorate
git diff --stat
git diff --name-only
```

Linux 需要记录当前代码来自哪个分支、是否存在未提交修改，以及这些修改是否全部属于高频参数化控制器。发现来源不明或混入其他任务修改时，返回 `needs_decision`，不得通过覆盖、reset 或 checkout 丢弃现有修改。

## 动作接口审计

发布提交必须包含且只包含以下六个离散模式：

```text
HOLD
MOVE_POS
MOVE_NEG
VIEW_POS
VIEW_NEG
UP
```

环境动作命令必须同时携带离散模式 $m_t$ 和归一化力度 $\alpha_t\in[0,1]$。HOLD 忽略 $\alpha_t$ 并施加零主动力。非 HOLD 力度通过动作专属上下限映射：

$$
F_m(\alpha_t)
=
F_m^{\min}
+
\alpha_t\left(F_m^{\max}-F_m^{\min}\right).
$$

力度范围可以仍处于暂定状态，但 MOVE、VIEW 和 UP 必须分别暴露独立的最小值与最大值配置接口。审计报告必须同时给出无量纲 $mg$ 比例和按实时胶囊质量换算的牛顿值。

## 力学语义审计

MOVE 必须在相机端球心和另一端球心施加同方向、等大小的力，两个端点的合力等于本周期 MOVE 总力。施力方向垂直于胶囊长轴并平行于世界水平面，正负模式对应相反方向。

VIEW 必须只在相机端球心施加与 MOVE 相同定义的水平侧向力。UP 必须只在相机端球心施加世界坐标系竖直向上的力。当前研究合同不接受旧 TASK-008 中的双端 UP 力偶作为等价实现。

实现可以调用点力 API，也可以使用严格等价的质心合力与力矩：

$$
\mathbf{F}
=
\sum_i\mathbf{F}_i,
\qquad
\boldsymbol{\tau}_{\mathrm{COM}}
=
\sum_i
\left(\mathbf{p}_i-\mathbf{p}_{\mathrm{COM}}\right)
\times\mathbf{F}_i.
$$

Linux 必须记录实际采用的路径，并用纯函数测试证明没有重复施加点力与等价力矩。

## 频率与边界审计

物理仿真必须为 $240\,\mathrm{Hz}$，环境动作边界必须为 $10\,\mathrm{Hz}$。一条动作命令在完整 $0.1\,\mathrm{s}$ 内持续生效，对应恰好 24 个物理子步。

实现不得保留旧 TASK-008 的前后等待段，不得把一个 Actor 动作扩展成 1 秒宏，不得在 $0.1\,\mathrm{s}$ 中途自动撤力。周期结束后先取得边界 RGB 和动作结果，再由下一条命令覆盖当前力；模型推理期间仿真时间暂停，不额外推进物理。

至少需要以下自动化断言：

```python
def test_control_step_has_exactly_24_physics_substeps():
    assert physics_hz == 240
    assert actor_hz == 10
    assert physics_hz // actor_hz == 24


def test_non_hold_force_is_active_for_the_full_control_step(trace):
    assert len(trace) == 24
    assert all(sample.force_active for sample in trace)


def test_hold_force_is_zero_for_the_full_control_step(trace):
    assert len(trace) == 24
    assert all(sample.active_force_norm == 0.0 for sample in trace)
```

## Actor 观测泄漏审计

当前阶段的 Actor 边界只允许 RGB、上一动作模式、上一实际力度以及后续将加入的循环隐状态。真实位姿、速度、接触、胃壁法向、覆盖率和覆盖掩码不得进入 Actor 观测对象。

Linux 应对任务配置和观测构造代码执行静态扫描与运行时形状检查。仿真真值可以保留在独立的 privileged 字典中，但必须证明 Actor 返回值不引用该字典。

## 最小运行验证

Linux 需要在平面环境执行一个确定性短序列，以验证六模式均可被连续提交且每次只推进 $0.1\,\mathrm{s}$：

```text
HOLD@0.0
MOVE_POS@0.0
MOVE_POS@0.5
MOVE_POS@1.0
MOVE_NEG@0.5
VIEW_POS@0.5
VIEW_NEG@0.5
UP@0.0
UP@0.5
UP@1.0
HOLD@0.0
```

这里 `@` 后的数值表示归一化力度 $\alpha$，不是 $mg$ 比例。该运行只验证接口、方向、边界和数值有限性，不把单次位移或倾角作为最终力度范围验收。

每个周期必须记录动作模式、$\alpha$、映射后的 $mg$ 比例、牛顿值、24 子步计数、起止模拟时间、边界 RGB 帧号、质心位移、长轴角度变化以及是否出现非有限状态。

## 自动化验证命令

Linux 应使用仓库现有 Isaac Lab 启动脚本运行控制器纯测试、任务配置测试和一次 live smoke。具体测试文件名可以服从当前高频控制器的真实目录结构，但返回报告必须给出可复制的完整命令和逐项结果。

最低验证集合必须覆盖 Python 编译、动作枚举、力度映射、MOVE 分力、VIEW 单端力、UP 相机端世界向上力、24 子步同步、HOLD 零力、动作观测无真值和 live 有限状态。

## 返回报告

Linux 应创建并提交以下报告：

```text
handoffs/reports/TASK-009A-stage1-controller-baseline-audit-report.md
```

报告必须包含发布分支、完整提交哈希、来源提交、变更文件、六模式接口、三组暂定力度范围、力学 API 路径、全部频率、自动化测试结果、短序列逐周期摘要、已确认事实、未验证信息和偏离项。

外部日志或轨迹必须记录绝对路径、文件大小和 SHA-256，但不得提交生成数据本体。

## 完成条件

只有在控制器提交可获取、六模式定义正确、连续力度接口存在、UP 为相机端世界向上力、每个动作恰好持续 24 个物理子步、Actor 观测无仿真真值、自动化测试通过且 live 短序列状态有限时，本任务才可返回 `complete`。

若最新高频实现尚未提交或无法与旧 TASK-008 区分，返回 `needs_input`。若接口已经实现但某项契约不一致，返回 `partial` 并保留完整证据，不得在本任务内自行改变研究合同。

## 后续阶段边界

TASK-009A 不修改旧 50 mm 顶点覆盖率。该基线通过后，下一子任务将独立实现 70 mm 面积覆盖、10 Hz 覆盖更新、用户指定入口区域、同步回合环境以及随机策略时间预算预实验。
