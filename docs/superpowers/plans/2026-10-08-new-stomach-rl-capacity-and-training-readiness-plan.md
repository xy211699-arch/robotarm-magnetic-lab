# 新胃 RL 容量与长程训练就绪实施方案

**状态：用户已确认，容量阶梯修订为 1、4、8、12、16、20，其余方案不变。日期：2026-10-08。** 本文依据最新 Linux 报告制定下一阶段，不授权直接运行正式四组多种子训练。执行者确认后逐任务实施与验收；使用 `superpowers:executing-plans` 的原生逐任务方式，不默认派生子代理。本文遵循用户排版要求，以任务段落记录进度，不使用复选列表。

**目标：先解决可信采样吞吐和多环境隔离，再交付可以跨越回合边界的训练入口。** 本阶段最终产物是经过实测的容量上限、同预算耗时报告和经过短程验收的长程训练器；不是一组可用于论文的训练结果。正式样本量、更新批量和正式训练时长须在容量结果返回后单独冻结。

**架构：保持已通过预验证的单环境实现作为参照，新增 RL 专用的多环境适配层。** 同一进程的多个环境共享物理时钟，但各自持有动作、碰撞、磁场滤波、覆盖、相机缓存、位姿采样和 GRU 状态；部分 reset 不能额外推进其他回合。先证明两环境正确，再测更大环境数，不将删除旧断言当作实现方案。

**技术栈：沿用 Linux 已记录的 Isaac Sim、Isaac Lab、PyTorch、Warp、NumPy/SciPy 与原有限尺寸磁体模型。** 不升级 SDK，不替换磁场近似模型，不安装新的训练框架。规格来源为 `docs/superpowers/specs/2026-10-08-new-stomach-rl-preflight-design.md`，既有实施计划为同目录对应的 preflight implementation plan；最新证据为 `handoffs/reports/NEW-STOMACH-RL-PREFLIGHT-20261008.md`。本文提出的多环境生命周期和正式训练器扩展只有确认后才生效。

## 起点与冻结约束

**代码起点为 `3ff631e08ecffd97a3bba78cfbbc923906d5dc01`。** 该提交位于 `feature/new-stomach-rl-preflight-20261008`；已审计实现为 `69f393cb95268eb17986f5ee0a2049386d6745aa`。保留 release `1e89c9ccb48545c4513915be94d4c17d9cdebaaa` 和所有旧 TASK-010 行为，不从旧 Windows TASK-010 分支继续。本方案确认后建议新建 `feature/new-stomach-rl-capacity-20261008`，不得直接推 main，不在原证据工作区覆盖工件。

**研究定义完全沿用冻结规格。** A=Blind+Single、B=Vision+Single、C=Blind+Chunk、D=Vision+Chunk；Single 为 9D×1 秒，Chunk 为 4×9D×0.25 秒且全部执行，同一每秒位移预算和 C2 安全约束。物理 240 Hz、RGB/决策 1 Hz、C10 10 Hz 奖励、C1 真实图像边界辅助评测；70 mm、圆形 120° FOV、第一交点可见性、面积权重及完整右管加 Tube B 排除掩码不变。

**安全和数据边界不得因性能改变。** 保留 5 mm 余量、关节限位、速度/加速度、原停止路径、磁力模型及保护、碰撞精度和 USD/贴图。新胃库保持 train/validation/test=1000/100/100；本阶段只采样 train，validation 仅用于固定端到端复核，test 不参与性能选择、开发训练或调参。Actor 保持 548D，仅冻结视觉或零视觉512D及上一秒已下发历史36D；特权 Critic 28D 分离，GT 不得进入设备完成判定或硬故障判定。

**四项奖励保持原式和旧 5 秒状态机。** 每 0.1 秒计算 `100*ΔC10 + 0.1*ΔEscape - 0.002*I_no_progress + 0.2*I_coverage_resumed`，十项加总，不奖励 C0 或 reset HOLD，不继承中性 bring-up 奖励。状态机阈值是否适合新动力学仍属待研究项；本阶段只检查一致性和可观察诊断，不擅自调权或制造覆盖收益。

## 审阅重点

**最高风险是一个环境 reset 导致其他回合多走一秒。** 当前单环境 `_settle_and_initialize()` 使用 `super().step()` 进行 HOLD，不能原样嵌入部分 reset。必须用共享时钟上的每行 WARMUP/ACTIVE 状态，且 WARMUP 行不产生策略样本、奖励或 GRU 推进；其他 ACTIVE 行正常产生一次一秒样本。

**第二项风险是坐标或缓存误用使每行都在读取 env_0。** 位姿库是既有世界坐标记录；多环境需显式转换为相对原始场景原点的坐标，再加各环境原点，保持朝向和局部几何一致。碰撞网格、相机视图、磁体、胶囊、停止路径和窄相缓存必须按行绑定；磁场计算限于同一独立仿真副本，不人为引入不同环境间的磁耦合。

**第三项风险是吞吐数字不对应有效 PPO 数据。** 初始化、WARMUP、padding、评测和作废样本均不计训练 transitions；所有报告同时给出有效样本数、全程墙钟、reset 时间和采样时间。冻结每次更新的有效样本语义之前，不把 N×T 或 1000 updates 当成同预算证明。

**第四项风险是 TIMEOUT bootstrap 与 GRU reset 混淆。** timeout 使用终止前 `final_obs` 的 Critic 值，真正终止不 bootstrap；两者都截断跨回合优势递推并清空对应 GRU。新回合 C0 不能用作旧回合末状态，批次结束但未终止的行必须保留连续 hidden 和正确 bootstrap。

**第五项风险是伪恢复和伪加速。** 保存网络并重新 reset 不等于恢复完整物理轨迹；恢复报告必须说明其级别。优化不得减少磁场更新次数、跳过停止检查、降低覆盖采样或隐藏 GPU 同步成本，所有等价性测试先于性能结论。

## 文件与接口边界

**目录约定如下，避免执行者混用历史入口。** 下文 PKG 指 `source/robotarm_magnetic_lab/robotarm_magnetic_lab`，TASK 指 `PKG/tasks/manager_based/robotarm_magnetic_lab`；所有文件名均相对于仓库根。已有 `learning/new_stomach_rl_runner.py` 和 `scripts/new_stomach_rl/train_smoke.py` 保留烟雾定位，不直接改造成无上限长训脚本。

**容量子项目新增独立模块。** `TASK/mdp/new_stomach_rl_vector_action.py` 负责每行轨迹、伺服及停止证书；`TASK/mdp/new_stomach_rl_vector_terms.py` 负责每行终止/观察绑定；`TASK/mdp/new_stomach_rl_vector_magnetic.py` 负责独立滤波及等价磁场批处理；`runtime/new_stomach_rl_vector_runtime.py` 负责双时钟、视觉缓存及每行生命周期；`TASK/new_stomach_rl_vector_env.py` 与 `TASK/robotarm_magnetic_new_stomach_rl_vector_env_cfg.py` 负责新任务注册和配置。优先复用现有纯函数；如果必须修改旧共享模块，先返回具体最小差异和 `needs_decision`，不绕开原断言或单行索引。

**计时和证据新增独立入口。** `scripts/new_stomach_rl/export_review_bundle.py` 导出可核验小工件；`scripts/new_stomach_rl/profile_capacity.py` 区分互斥与嵌套热点；`scripts/new_stomach_rl/validate_vector_isolation.py` 验证 reset/坐标/物理隔离；`scripts/new_stomach_rl/benchmark_capacity.py` 执行固定容量阶梯。对应测试放在 `tests/new_stomach_rl/test_review_bundle.py`、`test_capacity_profile.py`、`test_vector_action.py`、`test_vector_runtime.py`、`test_vector_isolation.py` 和 `test_magnetic_equivalence.py`。

**长程训练子项目使用新的明确接口。** `learning/new_stomach_rl_rollout.py` 保存 terminated、truncated、final_critic、episode_start、valid_transition、policy_latent 和旧联合 log_prob；`learning/new_stomach_rl_ppo.py` 处理递归序列及 GAE；`learning/new_stomach_rl_checkpoint.py` 定义恢复级别和哈希；`scripts/new_stomach_rl/train.py` 执行有限预算训练；`scripts/new_stomach_rl/supervise_training.py` 由用户启动并检测运行状态。对应新增 `test_long_rollout.py`、`test_timeout_gae.py`、`test_training_resume.py`、`test_training_supervisor.py`。本方案不增加 VLM、病灶识别或新的奖励模型。

## P0：接收侧可核验证据与基线冻结

**本任务先补证据可读性，不重跑 224 MB 的全部动力学工件。** 导出原 `final_artifact_inventory.json`、`final_executed_commands.json`、最终各 Gate 摘要、envs_1/envs_2 摘要及计时 CSV、129 项回归日志和外部依赖清单；完整物理轨迹及 checkpoint 保留 Git 外。小工件通过交接分支附件目录交付，超出仓库常规约束的部分采用双方确认的传输位置，不能只再交一个 Linux 绝对路径。

**测试循环固定为先失败、后实现、再复核。** `test_review_bundle.py` 首先要求缺失文件、字节数不符、SHA 不符、错误实现提交都拒绝通过；然后实现 `export_bundle(source_inventory, output_dir)`，输出保持原身份的工件和 `bundle_manifest.json`，再由 Windows 重新计算 SHA。Gate 通过条件是报告中的关键数值可从摘要和 CSV 重算，且输入工件有可追溯的原路径与哈希，不把新导出时间当成原实验时间。

**同时冻结一份单环境对照。** 固定 train 位姿 ID、原始9D/36D动作序列、资产/掩码/权重哈希及版本，保存逐步参考命令、磁力/磁矩、覆盖集合、奖励和相机帧号。既有日志不足以构造配对参照时仅补一次有限时长运行，不重新训练策略。P0 结束提交证据工具及报告，不修改动力学实现。

## P1：热点归因与保持语义的单环境提速

**本任务只优化经过测量且能证明等价的开销。** 将当前每秒约 3.682917 秒拆为磁场模型计算、CPU/GPU 拷贝、同步、物理推进、规划、停止/碰撞、10 Hz 覆盖、相机和日志开销；现有嵌套时间不得相加。`profile_capacity.py` 提供外层 CUDA 完成计时和受控 profiler 开关，计时开关默认关闭，不影响正式样本语义。

**新增测试锁定安全计算频率和模型输出。** `test_capacity_profile.py` 验证嵌套计时标签、缺失 worker 摘要判失败及有效样本计数；`test_magnetic_equivalence.py` 在相同输入 pose/速度/滤波状态下比较原桥与新适配模块的原始和滤波 wrench、反作用力矩、单位、限幅与 ramp/release 逻辑。容差必须在看到优化结果前从原实现数值重复性建立并登记，不能观察到偏差后放宽；无法证明等价的近似不进入本任务。

**允许的候选优化不改变物理任务。** 缓存只读几何和固定配置，减少重复 `.cpu()`/`.numpy()` 同步，采用原有限磁体模型支持的等价批处理，缓冲只读日志及关闭不进入 RGB 的调试展示；只有证明原相机观测和物理结果不变才保留。磁场桥代码已有逐环境循环，是否可批处理及收益必须实测，不能预先承诺 N 倍加速。

**验收按数值和语义先行。** 固定动作回放中每一秒仍有 240 次磁场更新、240 次物理推进和十次精确覆盖；相同边界可见性和奖励一致，碰撞/停止/HOLD 拒绝语义不变。完整物理轨迹受浮点或 GPU 重复性影响时报告逐步误差和原参照波动，不以最终覆盖接近代替模型等价。无收益也可以提交真实结果；P1 不以“必须加速成功”为门禁。

## P2：两环境隔离和部分 reset

**本任务先完成 N=2，不承诺 N=20。** 新任务建议注册为 `Template-Robotarm-Magnetic-New-Stomach-RL-Vector-v0`，已验收单环境 ID 保留。向量动作接口为 `process_actions(actions: Tensor[N,D])`、`apply_actions()`、`reset(env_ids)` 和 `executed_history: Tensor[N,4,9]`；每行持有轨迹、目标、tick、碰撞证书、停止缓存和 telemetry，机器人测量及关节写入必须使用对应行。

**磁控和几何隔离必须覆盖有状态对象。** 独立维护每行 coupling elapsed、滤波 wrench、模型可变缓存、碰撞 checker、mesh 原点和相机变换；可以共享只读三角网格、常量权重和网络权重，但不能共享覆盖 mask、episode clock 或位姿 RNG。实现 `reset_rows(env_ids)` 后，未重置行的全部有状态字段应保持不变。库采样器按环境独立 RNG，reset 不重复重设全局 seed，世界坐标转换有 round-trip 测试。

**部分 reset 采用共享物理时钟下的行状态，不调用额外全局 step。** 每行状态为 WARMUP 或 ACTIVE；reset 后在下一次正常的一秒批次执行原 HOLD，完成后初始化 C0。WARMUP 行输出 `valid_transition=False`，不生成 PPO 奖励、不占用训练样本预算、不推进 Actor hidden；同一批次其他 ACTIVE 行正常推进 240 步。初始化 HOLD 不算该行 120 秒任务预算，但必须计入墙钟成本；禁止冻结其他动态物体或对其他行偷偷多走 240 步。此生命周期是本方案待确认的接口扩展，不冒称原 preflight 已实现。

**边界接口必须返回可区分的旧终点与新起点。** `step()` 的 extras 含每行 `valid_transition`、`episode_start`、`final_obs` 及 final mask；历史只来自已完整下发的安全投影命令，不从胶囊实测状态反推。C10/C1、奖励和 episode length 按每行有效任务时间记录，C0 单独标识。某行安全证书失败立即使整项验收失败，不静默 reset 掩盖故障；不在本阶段重新定义硬故障为正常训练终止。

**零覆盖初始化需要显式边界测试。** 当前 `runtime/new_stomach_rl_reward.py` 要求 C0>0；新适配必须测试合法 C0=0 时仅建立奖励基线、不产生初始化奖励，负值和大于1仍拒绝。优先在新向量运行时中实现兼容初始化；若需要修改该新增预验证奖励模块，先登记精确变更并保留原正C0回归，不改旧TASK-010状态机或位姿库。该项属于代码边界风险，不声称已在真实库中观察到零C0失败。

**隔离测试以环境行置换和局部扰动为核心。** 先写失败测试，验证不同动作只改变对应行指令、不同初始位姿正确平移、交换行顺序结果随之交换；再实现并通过测试。GPU 运行将行0反复 reset，行1持续执行固定序列，确认行1不丢步、不多步、不继承覆盖或相机缓存，且 hidden 和奖励状态未被清空。相同动作/局部位姿的 N=1 与 N=2 回放比较物理误差和覆盖掩码，容差按 P1 预登记；不能用纯函数测试替代 GPU 隔离。

## P3：容量阶梯及完整预算报告

**本任务严格逐级测 1、4、8、12、16、20 环境。** 每级在隔离验收通过后做两次八步 warmup 加32步计时，再做两段64步 rollout；包含实际120秒 timeout/部分 reset 的追加有限测试，不能只测试无终止的短区间。更高级别失败就停止，记录 unsupported、OOM、数值失败或安全失败的具体原因，后续记 not_run，不自动扩大到20以外。

**吞吐指标以有效样本为主，同时报告名义步数。** 计算 `q_valid = 有效训练样本数 / 全程采样墙钟秒数`，纳入 reset/WARMUP；并另报不含初始化的稳态吞吐、每组 A/B/C/D 的代表性耗时、驱动显存采样与 torch 分配峰值、CPU RSS、PPO 更新时间和安全投影/HOLD比例。显存采样不能称硬件瞬时峰值，时间嵌套不能相加，B/C 实测前不假定与 A/D 同速。

**环境数按正确性和完整训练预算选，不按最大数选。** 最大可运行 N 与最高有效吞吐 N 分开报告；并行增加而磁场 CPU 循环线性增长时应如实保留结果。以 768000 个有效样本作为可比预算参照，按各组实测吞吐分别外推，附评测、PPO 和保存开销，不把参照量直接升级为正式冻结量。

**用户训练时间预算是 P3 后必须决定的项目约束。** 纯采样若希望每个种子三天内完成，参考768000个样本至少需要约2.963有效样本/秒；一天内至少约8.889，真实要求还要加额外开销。上述仅为决策刻度，不是性能承诺或已接受门槛。若仍不可承受，提交等价批处理、更多硬件或明确减少总样本量的成本选项；不能悄悄改为单环境1000更新并声称与原参考实验同预算。

## R0：正式训练器的纯函数与回合边界验收

**本任务可提前做 CPU 数学测试，但 GPU 接入依赖 P2/P3 的接口冻结。** 新 Rollout 保存联合9D/36D tanh变换密度及 latent、真实动作与执行历史分离、有效掩码、回合起点、独立 terminated/truncated 和旧终点 Critic 输入。不得把投影后的命令当作从原策略分布直接采样的动作来计算 PPO ratio；执行历史作为下一时刻观察保留。

**GAE 使用两个不同的边界掩码。** 对有效样本定义 bootstrap 值：普通步使用同回合下一观察，TIMEOUT 使用终止前 final Critic，真正终止为零；优势递推只在同回合下一有效样本上延续。公式为 \(\delta_t=r_t+\gamma b_t-V(s_t)\)，\(A_t=\delta_t+\gamma\lambda m_t A_{t+1}\)，其中 \(m_t\) 在 terminated 或 truncated 时为零，WARMUP/padding 不进入损失或递推。timeout 属120秒有限任务端点时是否应作真正终止是建模选择；本方案推荐沿用固定时长截断并 bootstrap 的训练约定，但须确认后冻结，不能依据短测效果来回切换。

**测试必须能手算并覆盖多个交界。** `test_timeout_gae.py` 使用可手算的 reward/value 序列分别验证普通步、TIMEOUT、真正终止、rollout 尾部和 reset padding；`test_long_rollout.py` 验证两环境不同时间重置、只清空对应 GRU、序列 minibatch 不打乱时序、padding 无梯度。收集时和更新时的 log_prob 在不更新参数的同序列回放中一致，并检查9D/36D联合密度不除维数，KL/entropy可另报每维诊断。

**训练超参只能作为明确的开发配置，不直接称正式冻结。** 起点沿用烟雾候选 gamma=0.999^10、lambda=0.95、Adam3e-4、clip0.2、value系数0.5及梯度范数1；GRU256、冻结ResNet权重及Actor输入边界不变。每次更新有效样本目标、sequence length、minibatch、epochs、总样本量、checkpoint规则和有限搜索次数由 P3 后的开发合同确定；调参只看train/validation，不能读取test来选择。

## R1：恢复合同及人工启动监督

**checkpoint 必须区分参数恢复与物理连续恢复。** 保存 Actor/Critic/优化器、update及有效样本计数、配置和代码/资产/权重哈希、Python/NumPy/torch CPU/CUDA RNG、各位姿采样器状态及适用的统计状态。暂不承诺 PhysX 内部状态可精确恢复；默认只允许在回合边界保存用于训练续接的 checkpoint，恢复时新 reset、清空GRU并登记轨迹不连续。若在批次中途保存而没有完整模拟器恢复能力，只能标记 weights-only/非严格续训，不能宣称同轨迹复现。

**人工启动监督不采用代理长期等待。** 用户通过 `supervise_training.py` 启动有限预算的 `train.py`，监督器读取结构化 heartbeat、子进程退出码、更新数、有效样本数、loss/梯度有限性、GPU内存、最近checkpoint和停止原因；启动前核对门禁及哈希，不接受尚未冻结的正式运行。短暂没有update不等于卡死，heartbeat应在采样中推进，超时阈值依据 P3 最慢健康步设置。

**监督器只控制自己启动的进程。** 无授权不反复重启、不自动改变参数或接着启动下一种子；异常时先保存可用诊断和非覆盖式摘要，再安全停止其进程组，不删除工件。`test_training_supervisor.py` 使用快速假worker分别模拟健康慢采样、进程非零退出、无摘要退出、NaN、OOM、心跳停滞及用户停止，确保不会把返回码0但无完整结果当作通过。Linux交付时必须提供实际可运行的启动命令、预计时间、日志和停止方法。

## R2：有限长程 smoke 与开发阶段准入

**本阶段 GPU 训练仍限定为验收，不评判策略性能。** A/B/C/D 各做三次更新、每次收集64个全局策略边界，记录每组实际有效样本数，保证跨越原120秒TIMEOUT；向量运行还要覆盖不同步的reset行。用不触碰安全约束的测试夹具另验真正终止路径，禁止为验收而制造真实危险碰撞。所有 loss/梯度有限、GRU/GAE/有效样本计数正确，并完成一次边界checkpoint恢复后的新回合更新。

**同一组需要完成正常结束和异常停止两种生命周期。** 正常结束持久化 summary、哈希和退出码；用户停止保存 interrupted 状态而非 pass。固定 train/validation 位姿各120秒检查原物理/覆盖/RGB/四奖励合同，终止后的 WARMUP 不混入旧回合曲线。当前单环境日志中的覆盖率只作回归参照，不设置要求“必须提高覆盖率”的正确性门槛。

**开发种子阶段单独形成下一份冻结合同。** 从已验证配置出发，先用一个开发训练种子及小规模固定validation子集检查学习曲线、动作投影、覆盖新增率、恢复阶段驻留和模式坍缩；明确训练样本上限与验证频率后由用户手动启动。短程结果只回答能否稳定学习及是否需有限调参，不能冒充三个正式种子的总体性能。

**正式研究仍须回答视觉利用问题。** 开发阶段通过后才设计配对 A/B、C/D 的多种子评价及视觉干预；训练集1000、验证集100、测试集100是位姿库划分，原计划最终固定50个test位姿的具体ID清单还需冻结。统一各组有效训练预算、评测初态与推理方式，先登记checkpoint选择规则和Mean/Median/P10/CVaR20统计方式，再开放test。VLM比较和随机策略研究不属于本实施阶段。

## 命令约定与人工执行

**本段第一条是当前已有的 Linux 回归命令，其余是交付时必须实现的目标接口。** 新脚本尚未存在，不能现在复制运行。实现者使用真实项目工作树和已确认的位姿manifest/掩码路径，不依赖新工作树的兄弟目录恰好同名；默认路径找不到就报 needs_input，不生成替代库。Isaac CLI 使用本机已验证的 `--viz none`，不是无效的 `--headless`。

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl tests/stomach_migration tests/magnetic_following \
  tests/stomach_coverage/test_task010_recovery.py -q --disable-warnings
```

```bash
# 以下为待实现 CLI，确认方案且脚本交付后才可运行。
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/new_stomach_rl/validate_vector_isolation.py \
  --num_envs 2 --seconds 120 --pose_manifest "$POSE_MANIFEST" \
  --mask "$TARGET_MASK" --device cuda:0 --viz none

ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/new_stomach_rl/benchmark_capacity.py \
  --num_envs 1,4,8,12,16,20 --pose_manifest "$POSE_MANIFEST" \
  --mask "$TARGET_MASK" --device cuda:0 --viz none

# 由用户启动；config须先由R2准入门禁冻结，不能现在创建正式训练配置。
/mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh \
  scripts/new_stomach_rl/supervise_training.py --config "$APPROVED_CONFIG"
```

## 交付、耗时与停止条件

**每个任务按红灯测试、最小实现、绿灯回归、GPU证据和独立提交的顺序交付。** P0→P1→P2→P3为容量子项目；R0纯函数可在容量阶段准备，但其GPU接入依赖P2/P3，R1→R2为训练就绪子项目。Linux报告新增 `handoffs/reports/NEW-STOMACH-RL-CAPACITY-20261008.md` 和 `NEW-STOMACH-RL-TRAINING-READINESS-20261008.md`，逐项记录 base/head、命令、实际结果、偏差、未运行内容、外部路径及SHA，不覆盖旧报告。

**当前只能给出验证计算量，不能可靠预估架构开发工期。** 现有D单环境下，一个120秒回合纯env.step约7.37分钟；旧容量标准208个策略步约12.77分钟，不含启动/reset/PPO。R2四组各3×64边界若均按D单环境耗时估算，纯步进约47分钟，另需启动、回载、跨回合和恢复；这只是调度参考，A/B/C和新并行路径尚未实测。沿用用户既定方式：Linux代理负责实现、非仿真单测及启动/检测脚本，GPU实验和训练由用户人工启动；代理读取返回摘要，不自行发起长实验或保持长时间等待。

**停止条件不得以低精度或换定义绕过。** 发现共享核心必须改动、SDK内部接口变化、reset污染、错误坐标、相机丢帧、非有限状态、安全证书失败或无法保持磁场等价，返回 needs_decision/blocked及最小证据；缺真实外部资产返回needs_input。容量不足时先交实测预算，等待用户选择计算资源或研究样本预算，不启动正式训练。全部正确性门禁通过也仅表示训练就绪，不表示模型已经优于Blind或证明视觉依赖。

**本方案提交确认只授权计划中列出的工程范围，不自动授权后续正式实验。** 用户确认后先实施P0—P3；容量报告返回并冻结批量/预算后再接入R0—R2。保留用户既定的Windows规划验收、Linux实施回报及人工启动实验方式；旧规格保留，新阶段执行入口为 `handoffs/active/NEW-STOMACH-RL-CAPACITY-20261008.md`。
