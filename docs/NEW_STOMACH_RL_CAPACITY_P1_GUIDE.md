# 新胃RL容量：P1原链路热点分析入口

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
