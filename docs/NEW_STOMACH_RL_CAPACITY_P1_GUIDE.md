# 新胃RL容量：P1原链路热点分析入口

## 当前执行：两小时总预算短测（用户最新要求）

此前24组长容量流程已停止，4环境D组8边界短测结果保留。新任务`capacity_jobs/20261009T124829.090487Z`已启动：D组测1、4、8、12、16、20环境，随后A/B/C各测4环境；共9项，每项2步预热+6步计时，N>1在第5个批次局部reset行0并检查其他行不变。固定动作同时覆盖平移、Ball及混合指令。物理/磁场240Hz、覆盖10Hz、RGB1Hz、安全距离及旧参数不变。

**总墙钟预算7200秒，包含初始化和退出；不是每档两小时。** 截止北京时间2026-10-09 22:48:29，提前30秒停止启动/中断当前子任务，SIGINT等待最多20秒，未退出则终止该独立进程组。已完成结果和已写边界日志保留，未测部分not_run；不会自动延长、重复或转入训练。单项通过为`screened`，全部短测通过为`screening_completed`；到时为`budget_exhausted`，均不能称完整P3/120秒回合门禁通过。四组全阶梯长测已取消。

已执行启动参数为此前`supervise_capacity.py start`命令另加`--quick --budget_seconds 7200`。只需查询，不要重复启动：

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008
./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_capacity.py status
```

下方为旧过程快照，旧“运行中”不代表当前任务。

## 当前进度与多环境入口（2026-10-09最新授权）

Single完整双环境验收已`pass`：两段121个批次、每段120个有效策略边界、240步/批、19项比较和7次局部reset另一行不变检查通过。Chunk三次20秒新向量回放动力学NPZ完全一致，但与旧P0单环境的姿态差13.67°超过冻结10°上限；保留`fail`原结果。用户批准非严重项暂缓，因此本轮不再提高阈值，记Chunk参照等价性为`deferred`，不据此宣布正式训练就绪。

已启动后台**多环境容量测试，而非训练**：先4环境D组8个一秒边界短筛查，随后逐级1、4、8、12、16、20环境，每级顺序A/B/C/D。短筛查只可得到`screened`，不能称完整容量通过。完整每组两次8+32计时、连续两段64步rollout并追加16步，以覆盖row0第17秒局部reset后的120秒TIMEOUT。初始化/部分reset HOLD成本计入全程有效吞吐，单独报告稳态吞吐；不执行PPO、不推断其更新时间。发现安全、非有限、OOM、相机/步数或reset污染即暂停后续。当前实现仍逐行复用CPU有限磁体计算，不保证增加N会提速。

只读查询（无需再次启动）：

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008
./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_capacity.py status
```

当前任务目录为`artifacts/new_stomach_rl_capacity/capacity_jobs/20261009T123005.785406Z/`，每阶段独立日志，status列出当前N/组/短筛查标志、心跳、结果与停止原因。真实每行10Hz覆盖/奖励、1Hz RGB哈希、磁力调用/步数、有效掩码、超时及投影信息存各子运行`boundaries.jsonl`。未开始正式对照训练；容量结果出来后才能冻结并行数/预算。

## 以下为此前验收过程与历史命令

## 2026-10-09代理接管与用户授权的克隆重复性验收

用户已授权Linux代理执行GPU验收并适度放宽数值标准；不再要求用户自行启动本轮验收。运行期间不要重复启动下面旧命令，以免并发抢占GPU。原P0登记与失败记录保留，默认不提供新参数时仍是旧严格门禁。

新入口先单独使用`--calibrate_repeatability`采集三次20秒回放，生成独立`acceptance_manifest.json`，标定不等于完整隔离通过。随后用`--acceptance_manifest /绝对路径/acceptance_manifest.json`执行原120秒对照和部分reset隔离。Single与Chunk分别标定、分别验收，不互用清单。所有物理配置不变。

初始试设8°姿态上限的标定真实失败（8.639°），没有生成清单；依据用户允许放宽的授权，已公开将标定工程上限修订为10°、版本v2，重新独立采集，原失败不删除。这是标定阶段修订，并非宣称先前零容差验收通过；后续正式验收不得再按失败结果抬高阈值。其余上限：位置分量3mm、线速度0.05m/s、角速度8rad/s、实际关节1e-4rad、下发关节1e-6rad、力分量1mN、力矩0.3mNm、外磁体位置50μm、方向1e-3rad；可见/累计集合用冻结面积权重计算对称差，上限4%总可达面积。实际清单阈值取独立标定最大差的3倍与预设下限的较大者，且不得超过工程上限；超限则不生成清单。

240步/240次磁力、10Hz与1Hz时钟、TIMEOUT/WARMUP、安全终止仍严格验证。每次局部reset之前/之后直接比较另一行的状态、目标、滤波、覆盖缓存、视觉特征、奖励缓存与时钟，必须完全不变；跨独立运行的完整动力学允许独立清单范围内的误差。RGB像素等价与GRU/PPO接入仍非本轮已验证结论。

代理已启动有限后台串行验收：Single完整验收→Chunk标定→Chunk完整验收；失败立即停止后续，不训练、不自动调参。首轮在Single局部reset后发现惰性相机部分推进，已修复新向量边界的单次全行采集、保留原失败数据和冻结清单，并启动新任务`acceptance_jobs/20261009T092758.806211Z`。修复后230项回归通过，GPU全程仍待实际结果；不要并发重跑下方历史命令。

用户当前只需查询状态：

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008
./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_isolation_acceptance.py status
```

`running`表示有限验收在执行；`paused_on_error`表示后续阶段未启动，需读取该任务status中的错误与日志；`completed`仅表示Single/Chunk隔离通过，不代表正式对照训练已启动或容量冻结。

## 当前下一步：先验收P2双环境（2026-10-09用户批准）

执行顺序已获用户确认，改为先做两环境正确性，再做P1性能优化和P3容量阶梯。**暂不执行下方历史P1采集命令，不重复P0参照。** 新向量任务与旧单环境任务分开；没有删除旧单环境保护，没有降低240 Hz物理/磁力、10 Hz覆盖、1 Hz RGB或5 mm安全余量。

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008

export ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor
POSE_MANIFEST=/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-coverage-regenerate/configs/new_stomach_v1/entry/pose_library_manifest_v1.json
TARGET_MASK=/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/new_stomach_coverage/tube_candidate_full_right_appendage_v1/candidate_face_mask.npz

./run_isaaclab.sh -p scripts/new_stomach_rl/validate_vector_isolation.py \
  --num_envs 2 --seconds 120 --mode single \
  --pose_manifest "$POSE_MANIFEST" --mask "$TARGET_MASK" \
  --device cuda:0 --viz none
```

这是真实双环境验收，不是训练，也不是单环境计时。一个进程顺序执行两段：不干扰的对照段，以及每17秒重置行0、行1继续原动作的隔离段。每段包含120秒任务及终点后的一个正常WARMUP批次；完整脚本另有两次全行初始化HOLD，合计244个共享一秒物理批次。仿真秒数不等于墙钟时间，新路径耗时尚未实测，不承诺并行提速。

执行期间只打印每10个批次的进度事件。最终查看`VECTOR_ISOLATION_RESULT`及`summary.json`，不能凭退出码0判定通过。Single为pass后，同一终端把`--mode single`改为`--mode chunk`再运行一次。fail或interrupted时先提供摘要路径，不继续容量阶梯，不调参或放宽容差。

默认结果保存到当前项目：

```text
artifacts/new_stomach_rl_capacity/isolation/<UTC运行ID>/
  summary.json
  control/boundaries.jsonl
  control/isolation_tape.npz
  reset_row_0/boundaries.jsonl
  reset_row_0/isolation_tape.npz
  failed_partial_tape.npz  # 发生异常且已有部分轨迹时
```

JSONL包含真实240子步计数、有效样本/WARMUP掩码、回合起点、TIMEOUT/旧终点Critic、已下发历史、逐10 Hz四项奖励/覆盖及1 Hz帧号。NPZ包含两行逐240 Hz动力学、模型调用前输入与调用后raw/filtered wrench、逐10 Hz/1 Hz的完整可见/累计集合及全局/局部时钟。文件较大，不加入Git。

比较包括：反复reset行0时行1是否保持原轨迹/滤波/覆盖/奖励/帧号；前20秒与已有P0固定参照是否一致。原链路前20秒配对失败时立即保存失败轨迹并停止剩余长批次，避免明知不等价仍跑完整段。沿用原预登记严格容差，不在看到结果后修改。克隆世界坐标的float32平移若引入差异，会如实报告失败及误差；不能凭最终覆盖接近忽略它。原始RGB像素等价、GRU训练器接入和P3吞吐仍未验证。

需要只读检查路径时，可以给同一命令加`--check_inputs`；它不会创建Kit或启动GPU，输出`inputs_verified`不代表双环境物理通过。

## 历史P1入口（暂缓，P2通过后再用）

P0的Single/Chunk两份人工采集均已复核通过。下面入口只分析原链路热点，不训练、不启用优化候选；所有物理、磁场、覆盖、动作和奖励参数不变。

已登记原始重复性容差：物理轨迹、磁场输入/输出绝对与相对容差均为0；覆盖集合、奖励及已下发历史严格一致。两次重放的21张RGB哈希均不同，而P0只有哈希、没有像素，因此不能把“哈希不同”解释成物理或图像同步失败，也不能推断像素误差大小。下面额外保存实际像素，为后续原图重复性审计提供材料。

## 1. 手动运行Single热点采集

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008

export ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor
POSE_MANIFEST=/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-coverage-regenerate/configs/new_stomach_v1/entry/pose_library_manifest_v1.json
TARGET_MASK=/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/new_stomach_coverage/tube_candidate_full_right_appendage_v1/candidate_face_mask.npz
TOLERANCE_MANIFEST="$PWD/artifacts/new_stomach_rl_capacity/evidence/p1_preregistration_20261009/tolerance_registration.json"

./run_isaaclab.sh -p scripts/new_stomach_rl/profile_capacity.py run \
  --mode single \
  --pose_manifest "$POSE_MANIFEST" \
  --mask "$TARGET_MASK" \
  --tolerance_manifest "$TOLERANCE_MANIFEST" \
  --timing --trace --save_rgb_pixels \
  --device cuda:0 --viz none
```

每种模式仍使用train-0003、20个固定动作、两个独立reset重放，共40秒有效仿真时间；每次reset仍有原1秒HOLD，不属于有效策略样本。墙钟需要数分钟，尚未实测此入口。不要并发GPU实例。

先确认最终`NEW_STOMACH_PREFLIGHT`摘要为`status: pass`。若fail/interrupted，先反馈，不继续下一种模式。程序退出0不代替摘要通过。

## 2. 同一终端运行Chunk

```bash
./run_isaaclab.sh -p scripts/new_stomach_rl/profile_capacity.py run \
  --mode chunk \
  --pose_manifest "$POSE_MANIFEST" \
  --mask "$TARGET_MASK" \
  --tolerance_manifest "$TOLERANCE_MANIFEST" \
  --timing --trace --save_rgb_pixels \
  --device cuda:0 --viz none
```

## 输出和判读

```text
artifacts/new_stomach_rl_capacity/profile/<UTC运行ID>/
  summary.json
  profile_boundaries.jsonl
  torch_trace.json
  torch_trace_events.json
  raw_actions.json
  repeat_0/
    reference_tape.npz
    policy_1hz.jsonl
    coverage_reward_10hz.jsonl
    rgb_1hz.jsonl
    rgb_pixels.npz
  repeat_1/
    同上
```

`profile_boundaries.jsonl`记录两次reset和40个有效策略边界。每个有效边界必须实际调用240次磁力更新、240次物理步进、十次C10覆盖/奖励和一次RGB采集；不减少、插值或补造样本。输出与此前登记的P0物理轨迹、覆盖集合、奖励及动作历史做严格配对比较，差异不自动放宽。

计时标签记录嵌套父关系及inclusive/exclusive CPU耗时。磁力内的cube/cylinder模型耗时不能再与磁力总耗时相加；actuator内的碰撞/覆盖也不能重复加总。外层CUDA同步保证策略步真正完成，内部标签只代表CPU跨度，不代表独立GPU内核耗时；GPU计算、拷贝、同步详情看torch trace。

`--timing`默认关闭，这里明确打开。`--trace`只记录首个有效动作，该步仍正常执行，但从稳态吞吐中排除；不在240Hz步与步之间加入计时用CUDA同步。原始RGB复制、逐步物理记录和诊断本身有开销：这些是原链路诊断数据，不是生产吞吐、优化收益或P3容量结果。

`rgb_pixels.npz`保存21张原始图像（含C0）及真实帧号；可能较大，只保留项目artifacts，不加入Git。此时还没有优化后图像，不宣称RGB等价验证通过。

两次结束后，把各自`summary.json`路径发给Linux执行端，随后根据实测热点选择保持语义的优化。当前P2代码及非仿真测试已交付，GPU隔离尚未验收；P3阶梯未开始。

## 离线登记入口（已完成，不需重新运行）

```bash
python scripts/new_stomach_rl/profile_capacity.py audit \
  --single_reference artifacts/new_stomach_rl_capacity/reference/20261008T152609.305712Z \
  --chunk_reference artifacts/new_stomach_rl_capacity/reference/20261008T153020.049473Z \
  --output_directory artifacts/new_stomach_rl_capacity/evidence/p1_preregistration_20261009
```

原目录已经存在，重复执行会拒绝覆盖。不能根据优化结果修改该登记文件。
