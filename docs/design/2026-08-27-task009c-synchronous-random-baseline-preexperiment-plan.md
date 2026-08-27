# TASK-009C 同步步进与随机基线预实验实施方案

## 文档状态与任务目标

本文档是 TASK-009C 的权威实施方案。任务目标是在已经通过验收的 TASK-009B 胃部覆盖环境上，建立严格同步的单环境回合运行器，实现七种不读取 RGB、覆盖率或仿真真值的随机策略与一个 HOLD 诊断策略，并完成三百秒随机基线预实验。

TASK-009C 只验证同步回合、随机策略、可复现数据记录、面积加权覆盖率汇总和时间—覆盖率曲线。TASK-009C 不实现 CNN、GRU、VLM、Actor、Critic、PPO、奖励、零位移惩罚、课程学习、多环境并行训练、模型选择或论文正式统计实验。

Linux 执行端不需要安装或调用 `superpowers:subagent-driven-development`、`superpowers:executing-plans` 或其他 Codex skill。执行端应按照本文档的门禁顺序手动实现、测试并保存证据。

## 基线、分支与权威边界

TASK-009C 的精确代码基线是 `64dd2ff33951cb780f938a81c91c22dde8764c93`，对应 `origin/feature/TASK-009B-stomach-coverage-environment`。Windows 规划分支为 `workflow/TASK-009C-synchronous-random-baselines`。Linux 必须从该规划分支的最新完整提交创建 `feature/TASK-009C-synchronous-random-baselines`，不得从旧 TASK-008、旧 TASK-009A 或过期 TASK-009B 规划提交重新实现。

TASK-009B 的 Gate 5 已由用户人工确认。经实测确认的不可达区域、入口定位方式、位姿库、面积加权覆盖实现、GPU PhysX 路径和控制器力度范围均视为冻结输入，不在 TASK-009C 中重新标定或修改。

当前冻结控制器仍有六个模式：`HOLD`、`MOVE_POS`、`MOVE_NEG`、`VIEW_POS`、`VIEW_NEG` 和 `UP`。每条命令同时包含归一化力度参数 $\alpha\in[0,1]$。力度映射固定为 MOVE 的 $0.70mg$ 至 $1.40mg$、VIEW 的 $0.20mg$ 至 $0.50mg$、UP 的 $0.80mg$ 至 $1.05mg$。HOLD 必须使用 $\alpha=0$ 并施加零主动力。

物理仿真固定为 $240\,\mathrm{Hz}$，动作边界、Actor 未来使用的 RGB 和覆盖更新固定为 $10\,\mathrm{Hz}$。每条动作恰好保持 $0.1\,\mathrm{s}$ 并推进二十四个物理子步。正式环境的 `episode_length_s=1800.0` 已高于本任务三百秒预算，TASK-009C 不应修改该配置，只需在运行开始时断言自动时限不会早于外部回合终止。

## 单环境与随机性边界

TASK-009C 必须以 `num_envs=1` 顺序运行全部回合。当前覆盖运行时按照单一环境维护累计集合，本任务不得为了缩短运行时间擅自改成向量化环境。多环境覆盖状态隔离应在后续 PPO 训练任务中单独实现和验收。

环境随机性和策略随机性必须分离。五个验证位姿分别使用固定环境种子 `950006`、`950011`、`950015`、`950017` 和 `950019`。七种策略在同一个位姿上必须使用相同环境种子。当前环境重置事件只恢复默认场景，但执行端仍须扫描并记录实际启用的 reset event；如果发现光照、材质、摩擦、资产或其他额外随机化，必须在预实验中关闭，不能让它们随策略变化。

策略种子按下式确定：

$$
s_{\mathrm{policy}}=960000+1000k+i,
$$

其中 $k\in\{1,\ldots,7\}$ 是策略编号，$i$ 是验证位姿编号的十进制后缀。例如 R1 在 `validation-0006` 上使用 `961006`，R7 在 `validation-0019` 上使用 `967019`。HOLD 诊断使用 `960006` 和 `960019`。最终配置文件必须展开并保存每个回合的实际种子，不能只在报告中描述公式。

随机策略对象只允许接收其自身随机数生成器和自身动作历史。随机策略不得接收 RGB、当前覆盖率、累计覆盖集合、胶囊位置、姿态、速度、接触、胃壁几何或其他仿真真值。运行器和评估器可以为了记录指标读取特权状态，但不得把这些信息返回策略。

## 固定初始位姿

正式随机策略使用冻结验证集中的五个位姿：`validation-0006`、`validation-0011`、`validation-0015`、`validation-0017` 和 `validation-0019`。这些位姿是现有清单 `fixed_live_reload_pose_ids.validation` 的前五项，不得替换为训练集或测试集位姿。七种随机策略必须使用完全相同的五个位姿。

HOLD 诊断只使用上述集合中的 `validation-0006` 和 `validation-0019`。HOLD 的两个回合只用于验证无主动力条件下的覆盖变化和回合链路，不作为与七种 $n=5$ 随机策略同等样本量的性能比较，也不得用于推荐回合时长。

## 指定位姿重置合同

现有 `Task009BTrainingEnv.reset()` 在基础 reset 后立即于默认位姿执行十个 HOLD 周期，因此外部运行器不能在完整 `env.reset()` 返回后才写入验证位姿。TASK-009C 必须扩展 reset options，使位姿写入发生在基础 reset 完成之后、十个 HOLD 周期开始之前。

调用接口固定为：

```python
observation, extras = env.reset(
    seed=environment_seed,
    options={"task009c_initial_pose": pose_record},
)
```

`pose_record` 必须包含 `pose_id`、`split` 和七维 `pose_world_xyzw`。环境只接受本任务配置中列出的五个 validation 位姿，拒绝缺字段、非有限值、非 validation 分组、未知 ID 或清单哈希不匹配的记录。

环境内部顺序固定为：调用基础 reset；重置参数化动作项和永久力合成器；将胶囊根位置与四元数写入 PhysX；将六维根速度写为零；调用 `sim.forward()` 和 `scene.update(0.0)`；检查写入后位置误差不超过 $10^{-5}\,\mathrm{m}$ 且四元数绝对内积不小于 $1-10^{-5}$；执行十个新的 HOLD 控制周期；将 `episode_length_buf` 清零；返回最后一个 HOLD 边界的 RGB。

覆盖运行时必须在该回合开始前清空累计集合，且只能用最后一个 HOLD 边界的 RGB 初始化 $C_0$。基础 reset 期间默认位姿产生的图像不得进入覆盖累计集合，也不得写成该回合的 $C_0$。`extras` 必须包含 `pose_id`、请求位姿、写入误差、十个 HOLD 的帧号与时间、稳定后位姿、稳定后速度、最终 RGB 摘要和 `episode_length_buf=0`。

## 同步回合合同

每个正式回合在指定位姿的一秒 HOLD 稳定结束后开始计时。初始时刻记为 $t=0$，记录稳定后的 RGB、可达区域面积加权覆盖率 $C_0^{\mathrm{reachable}}$ 和未剔除不可达区域的原始覆盖率 $C_0^{\mathrm{raw}}$。

正式回合恰好执行三千个 $0.1\,\mathrm{s}$ 动作边界，对应三百秒模拟时间和七万二千个物理子步。每个边界必须遵循“策略生成 `[mode_id, alpha]`、环境推进二十四个物理子步、采集一个新的 Actor RGB、使用同一帧更新覆盖、写入一条记录”的顺序。推理与文件写入耗时不得推进模拟时间。

每个回合必须产生三千零一个严格对齐的覆盖点：

$$
t_j=0.1j,\qquad j=0,1,\ldots,3000.
$$

任何缺失、重复、乱序或非 $0.1\,\mathrm{s}$ 间隔的时间点都使该回合无效。汇总程序禁止用插值、补零、前值填充或平滑修复不完整数据。每个正式动作边界必须有二十四个物理子步、一个唯一递增的 Actor RGB 帧号和一个相同帧号的覆盖更新。

回合在完成第三千个动作边界后由外部运行器显式结束。任何提前的 `terminated` 或 `truncated` 都属于协议异常。策略对象的随机数生成器、上一模式、上一力度、持续计数器、随机游走状态、阶段类型和剩余阶段时长必须在每个回合开始前全部清空，不能跨回合继承。

## 七种随机策略

### R1 独立同分布随机策略

R1 在每个十赫兹边界从六个动作模式中等概率采样。采到非 HOLD 模式时，$\alpha$ 从 $U[0,1]$ 独立采样；采到 HOLD 时固定输出 $\alpha=0$。每个边界都重新采样，不维持动作块。

### R2 半秒持续随机策略

R2 在动作块开始时从六个模式中等概率采样，并将该模式连续输出五个边界。非 HOLD 模式在动作块开始时从 $U[0,1]$ 采样一次力度，块内保持不变；HOLD 块使用 $\alpha=0$。五个边界结束后重新采样完整的模式和力度，新模式允许与旧模式相同。

### R3 一秒持续随机策略

R3 与 R2 具有相同的采样规则，但每个动作块连续输出十个边界。十个边界结束后重新采样完整的模式和力度，新模式允许与旧模式相同。

### R4 马尔可夫持续随机策略

R4 的初始模式从六个模式中等概率采样；初始非 HOLD 力度从 $U[0,1]$ 采样，初始 HOLD 使用 $\alpha=0$。之后每个边界以 $0.8$ 概率保持当前模式，以 $0.2$ 概率从另外五个模式中等概率选择一个不同模式。

当前和下一模式都为非 HOLD 时，力度执行截断随机游走：

$$
\alpha_{t+1}=\operatorname{clip}\left(\alpha_t+\epsilon_t,0,1\right),
\qquad
\epsilon_t\sim\mathcal N(0,0.1^2).
$$

进入 HOLD 时力度变为零；从 HOLD 离开并进入非 HOLD 时从 $U[0,1]$ 重新采样力度；保持 HOLD 时继续输出零力度。

### R5 MOVE 偏置持续随机策略

R5 初始化和每次重新采样时使用固定模式分布：HOLD 为 $0.05$，两个 MOVE 合计为 $0.60$ 且正负各为 $0.30$，两个 VIEW 合计为 $0.25$ 且正负各为 $0.125$，UP 为 $0.10$。非 HOLD 力度从 $U[0,1]$ 采样，HOLD 使用零力度。

初始化后每个边界以 $0.9$ 概率同时保持当前模式和力度，以 $0.1$ 概率从上述完整分布重新采样。重新采样允许再次得到同一模式，因此实际连续保持同一模式的概率可能大于 $0.9$；实现不得把重新采样错误地解释为强制换成不同模式。

### R6 固定中档力度的 MOVE 偏置持续策略

R6 的模式分布、初始化、$0.9$ 保持概率和 $0.1$ 完整重采样规则与 R5 完全一致。所有非 HOLD 动作固定输出 $\alpha=0.5$，HOLD 固定输出 $\alpha=0$。R6 用于区分覆盖变化主要来自模式持续性，还是来自随机力度造成的偶发强弱运动。

### R7 MOVE—观察交替结构化随机策略

R7 每个回合固定从 MOVE 阶段开始，随后在 MOVE 阶段和观察阶段之间严格交替。进入 MOVE 阶段时，从 `MOVE_POS` 和 `MOVE_NEG` 中等概率采样；进入观察阶段时，从 `VIEW_POS`、`VIEW_NEG` 和 `UP` 中等概率采样。

每次进入新阶段时，从离散均匀分布 $U\{5,6,\ldots,20\}$ 采样持续周期，从 $U[0,1]$ 采样一次力度，并在整个阶段保持模式和力度不变。阶段持续时间因此为 $0.5$ 至 $2.0\,\mathrm{s}$。R7 不生成控制器内部宏动作，只是在每个十赫兹周期重复输出当前阶段命令；R7 不使用 HOLD。

## HOLD 诊断策略

HOLD 诊断在全部三千个正式边界输出 `[HOLD, 0]`。它仍正常推进重力、碰撞、摩擦、相机和覆盖更新，不锁定胶囊状态。HOLD 诊断用于发现无主动力运动、覆盖累计错误或 reset 残留力，不能被解释为应当保持位置完全不变。

## 实验规模与执行次序

正式随机预实验包含七种随机策略、每种策略五个验证位姿、每个“策略—位姿”组合一个三百秒回合，共三十五个随机回合。HOLD 诊断包含两个验证位姿、每个位姿一个三百秒回合，共两个回合。正式总计三十七个回合。

执行次序固定为先遍历位姿、再遍历策略，或先遍历策略、再遍历位姿中的一种，并把顺序写入配置和运行清单。为了降低温度或长时间运行漂移对单一策略的系统性影响，本方案固定采用以位姿为外层、策略为内层的交错顺序；每个位姿内策略顺序由固定调度种子 `970009` 生成一次排列并写入配置。HOLD 两个诊断回合在全部随机回合完成后执行。运行时不得依据中间覆盖结果调整剩余顺序、策略参数或种子。

正式三十七回合开始前必须完成一个独立冒烟批次。冒烟批次在 `validation-0006` 上依次运行 R1 至 R7 和 HOLD，每个策略三秒，共八个回合；每个回合必须产生三十一个覆盖点。冒烟批次只验收协议和数据结构，不参与正式平均值、回合时长推荐或性能结论。

## 数据记录合同

每个边界记录必须至少包含任务版本、运行 ID、正式或冒烟标志、策略 ID、位姿 ID、环境种子、策略种子、边界索引、模拟时间、模式 ID、模式名称、$\alpha$、映射后的 `force_ratio_mg`、物理子步数、Actor RGB 帧号、覆盖 RGB 帧号、RGB 摘要、可达当前可见面积、可达累计覆盖面积、可达面积覆盖率、原始累计覆盖面积、原始面积覆盖率、胶囊质心位置、线速度、角速度以及有限状态标志。

胶囊位姿和速度仅作为离线诊断记录，不属于随机策略输入。运行器必须通过接口边界保证策略对象无法访问这些字段。

每个回合摘要必须至少包含 $C_0$、$C_{300}$、$\Delta C=C_{300}-C_0$、归一化覆盖曲线下面积、六种模式的实际比例、非 HOLD 力度均值与标准差、没有新增覆盖的边界比例、总质心位移、最大速度、最大角速度、RGB 帧唯一性、覆盖单调性、运行状态和异常原因。

归一化覆盖曲线下面积按梯形法计算：

$$
\operatorname{AUC}_{300}
=
\frac{1}{300}
\int_0^{300}C(t)\,\mathrm dt.
$$

所有逐边界日志、图像、临时结果和汇总工件必须保存在 Linux 外部工件目录，不进入普通 Git 历史。Git 只提交代码、测试、版本化配置、运行清单摘要、文档和返回报告。报告必须记录每个外部工件的绝对路径、字节数与 SHA-256。运行器还应在外部工件根目录维护 `latest_smoke_manifest.json` 和 `latest_formal_manifest.json` 两个稳定指针；指针记录真实时间戳目录、运行清单路径及其 SHA-256，汇总程序读取指针后必须再次验证目标哈希，不能仅按目录修改时间猜测“最新”运行。

## 汇总与绘图合同

七种随机策略的主曲线在每个对齐时刻对五个位姿的可达面积覆盖率逐点求算术平均。HOLD 曲线对两个固定诊断位姿逐点求平均。时间对齐检查必须先于平均值计算，任何策略缺少完整的五个有效回合时都不得绘制成完整正式结果。

主图包含八条平均曲线，横轴为模拟时间秒，纵轴为可达区域面积加权累计覆盖率百分比。统计使用全部十赫兹点，绘图可以只显示精确的一赫兹抽样点以降低拥挤，但不得平滑。图中不绘制置信带；每种策略的逐时刻标准差、最小值和最大值单独写入 CSV。

绘图必须同时依靠颜色、线型和标记区分曲线，不能只依靠颜色。七种随机策略使用七种高对比度、色觉友好的颜色，避免在白色背景上使用低对比度纯黄色；HOLD 使用黑色点线。标记每十秒显示一次，图例放在绘图区外，并在标签中明确 R1 至 R7 为 `n=5`、HOLD 为 `n=2`。输出至少包括三百 DPI PNG 和可编辑 SVG。

主图使用绝对可达覆盖率。汇总程序还必须输出 $C(t)-C_0$ 的逐时刻均值 CSV 和对应的可选审计图，以识别不同初始视野造成的 $C_0$ 差异；该审计图不取代主图。原始未剔除不可达区域的覆盖率保留在 CSV 中，不作为主排序指标。

候选回合时刻固定为 $30$、$60$、$120$、$180$、$240$ 和 $300\,\mathrm{s}$。汇总表必须报告每种策略在这些时刻的平均值、标准差、最小值和最大值。Linux 只能根据曲线和候选时刻数据给出描述性建议，不得自动冻结后续训练回合长度；最终时长由用户在审阅报告后决定。

## 配置与文件边界

实现应创建以下职责清晰的文件，除直接必需的包导出外不得扩展范围：

```text
configs/task009c/random_baseline_preexperiment_v1.json
source/robotarm_magnetic_lab/robotarm_magnetic_lab/baselines/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/baselines/random_policies.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task009c_episode_runner.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/task009b_training_env.py
scripts/stomach_coverage/run_random_baseline_preexperiment.py
scripts/stomach_coverage/summarize_random_baselines.py
tests/stomach_coverage/test_task009c_random_policies.py
tests/stomach_coverage/test_task009c_reset_pose.py
tests/stomach_coverage/test_task009c_episode_protocol.py
tests/stomach_coverage/test_task009c_summary.py
docs/TASK009C_RANDOM_BASELINE_PREEXPERIMENT.md
handoffs/reports/TASK-009C-synchronous-random-baseline-preexperiment-report.md
```

版本化 JSON 配置必须包含本文档冻结的策略参数、五个位姿 ID、两个 HOLD 位姿 ID、全部环境和策略种子、三百秒正式时长、三秒冒烟时长、十赫兹频率、候选时刻、交错执行顺序、位姿库数据 SHA-256、位姿库清单 SHA-256、不可达区域配置 SHA-256、覆盖配置 SHA-256 以及八条曲线的样式映射。程序不得在 Python 文件中维护另一套不一致的策略参数。

## 实施门禁

### Gate 1 纯策略与配置合同

执行端首先实现七种随机策略、HOLD 诊断和配置加载器。纯单元测试必须证明模式概率、持续周期、力度规则、种子复现、重新采样允许得到同模式、R4 切换语义、R7 阶段交替以及回合 reset 后状态清零。概率测试使用足够长的固定种子序列并采用预先固定的容差，不能依赖一次短随机序列恰好接近期望值。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009c_random_policies.py
```

Gate 1 失败时停止，不得进入 Isaac Lab 长时运行。

### Gate 2 指定位姿 reset

Gate 1 通过后实现 `task009c_initial_pose` reset option。纯接口测试和 GPU live 测试必须依次加载五个固定 validation 位姿，证明每个位姿在十个 HOLD 之前写入、残余力与速度已清空、位姿误差满足门限、十个 RGB 帧连续递增、最后一帧初始化非零 $C_0$、默认位姿图像没有进入该回合覆盖累计，并且稳定后 `episode_length_buf=0`。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009c_reset_pose.py \
  tests/stomach_coverage/test_environment_contract.py

./run_isaaclab.sh -p scripts/stomach_coverage/run_random_baseline_preexperiment.py \
  --device cuda:0 --reset_only
```

Gate 2 失败时停止，不得用外部 reset 后写位姿的旧方式绕过环境接口。

### Gate 3 同步回合与汇总单元测试

Gate 2 通过后实现回合运行器、逐边界 schema 校验和汇总程序。测试替身必须证明一个三秒回合恰有三十个动作、七百二十个物理子步和三十一个覆盖点；一个三百秒配置恰有三千个动作、七万二千个物理子步和三千零一个覆盖点。测试还必须主动构造缺点、重复时间、帧号错位、覆盖下降和提前终止，证明汇总程序会拒绝这些数据而不是修复它们。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/stomach_coverage/test_task009c_episode_protocol.py \
  tests/stomach_coverage/test_task009c_summary.py \
  tests/stomach_coverage/test_task009c_random_policies.py \
  tests/stomach_coverage/test_task009c_reset_pose.py
```

Gate 3 失败时停止，不得进入冒烟或正式预实验。

### Gate 4 GPU 冒烟批次

Gate 3 通过后，在 `validation-0006` 上分别运行 R1 至 R7 和 HOLD 三秒。八个回合都必须得到三十一个严格对齐覆盖点，所有物理与 RGB 状态有限，每个动作二十四个物理子步，Actor RGB 与覆盖 RGB 帧号一致，覆盖率处于 $[0,1]$ 且单调不减。

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/run_random_baseline_preexperiment.py \
  --device cuda:0 --smoke

./run_isaaclab.sh -p scripts/stomach_coverage/summarize_random_baselines.py \
  --latest_smoke --validate_only
```

`--latest_smoke` 必须通过前述稳定指针取得真实运行清单并验证哈希。Gate 4 工件不得混入正式汇总。

### Gate 5 三十七回合正式预实验

Gate 4 通过后才能运行正式预实验。运行器必须从版本化配置读取三十七个固定回合，禁止命令行临时覆盖策略参数、位姿、时长或种子。每完成一个回合立即落盘并更新只追加运行清单；程序重启时可以跳过已经通过完整性校验且哈希一致的回合，但不得覆盖或静默重跑已有有效结果。

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/run_random_baseline_preexperiment.py \
  --device cuda:0 --formal
```

正常物理结果包括覆盖率低、长时间无新增覆盖、胶囊受胃壁阻挡、速度较小或动作效果不明显，这些结果不得触发重试。只有非有限状态、求解器或渲染器异常、RGB 无法生成、帧同步失败、数据写入失败或提前 termination 才属于可重试的仿真异常。

发生可重试异常时，执行端必须关闭并完整重启仿真器，然后以完全相同的策略、位姿、环境种子和策略种子重试一次。第二次仍异常时，将同一组合标记为失败并停止正式门禁，不得换用其他位姿、种子或策略参数补足数量。

### Gate 6 汇总、绘图与报告

只有三十五个随机回合和两个 HOLD 回合全部通过完整性校验后，才能生成正式平均曲线和候选时刻表。汇总程序必须再次校验运行清单、外部文件哈希、回合数量、位姿配对、时间对齐、RGB 帧同步、覆盖范围与单调性。

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/summarize_random_baselines.py \
  --latest_formal --write_figures
```

返回报告必须按 Gate 1 至 Gate 6 的顺序记录状态、命令、直接观测结果、偏离项、未验证项和停止位置。报告必须给出规划基线、实现分支、完整 HEAD、修改文件、自动化测试计数、三十七个回合清单、每种策略的五个位姿、HOLD 的两个位姿、候选时刻表、主图路径、SVG 路径、CSV 路径，以及所有外部工件的字节数与 SHA-256。

## 禁止事项

TASK-009C 不得修改胃部或胶囊 USD、质量、惯量、碰撞、摩擦、重力、相机内参、七十毫米可见距离、面积权重、不可达区域、入口区域、位姿库内容、控制器方向、作用点或力度范围。

TASK-009C 不得让随机策略读取 RGB、覆盖率、位姿、接触或胃壁信息，不得加入覆盖启发式、受阻恢复、零位移负奖励、碰撞惩罚、动作掩码或提前终止逻辑。TASK-009C 不得基于预实验中间结果调整策略概率、力度或持续时间。

TASK-009C 不得把本次每策略五个位姿、单策略单种子的结果描述为论文正式统计证据。本任务只用于验证数据链路、建立无学习基线并帮助用户选择后续训练回合长度。

## 完成条件

只有 Gate 1 至 Gate 6 按顺序全部通过，三十五个随机回合与两个 HOLD 回合均完整，八条平均曲线和候选时刻表成功生成，返回报告包含可复现证据时，Linux 才能返回 `complete`。

如果缺少冻结外部位姿库或配置文件，返回 `needs_input`。如果 reset 顺序、同步步进、数据完整性、GPU live、正式回合或汇总校验失败，返回 `partial`。执行端不得用截图代替结构化日志，也不得把尚未执行的门禁标记为通过。
