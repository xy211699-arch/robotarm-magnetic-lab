# 新胃8环境训练就绪：R0 / R1 / R2工程记录

## 2026-10-10续执行：R2入口交付

最新状态 `needs_input`：R0/R1及R2工程CPU验证通过，真实A/B/C/D GPU验收全部not_run。不是正式训练就绪或研究实验完成。本轮没有启动Kit、GPU验收或训练，仅生成并检查有限配置；运行日志已更新。

本轮基准HEAD `80cd768554dd99e4c5068420136d4b62436bf50e`；R2实现审计HEAD `d2dd2878b7c589b8ff8b30efabca4e4742ada0b2`。实现分支及工作目录不变，最终文档提交后的远端HEAD推送核验后单独返回。

实际文件：新增`learning/new_stomach_rl_collector.py`、`scripts/new_stomach_rl/train.py`、`prepare_r2.py`、`tests/new_stomach_rl/test_vector_collector.py`及`test_r2_entry.py`；扩展原R1独立`supervise_training.py`，新增操作指南。旧共享核心、模型、奖励、物理、磁力、覆盖、安全、资产与库均未修改。

复用向量环境真实`env.step`、240次磁力/物理调用与已有10Hz/1Hz输出，不改旧单环境guard。collector从`extras.final_obs`计算TIMEOUT bootstrap，不使用自动reset后观测；invalid WARMUP不推进GRU/不计样本；网络原始sample/latent/logp和实际下发history独立保存。每段采集后无参数更新的序列回放核对联合logp，连续分段保留行顺序。CPU夹具分别覆盖9D/36D、TIMEOUT、真正终止、WARMUP及分段拼接。

有限候选：每组N=8、3×64边界，原2 epochs/全64步序列单minibatch及烟雾超参，正式预算仍未冻结。第5边界后局部reset行0，不额外推进物理，使不同步TIMEOUT在前128边界内出现；并检查每行均超时。第二次更新后显式结束剩余回合并全行reset，确认零秒边界后保存训练状态；新建runner回载并核对参数，再次reset/清GRU后执行第三次更新。此显式reset切断物理轨迹，日志标明checkpoint_reset，不声称原PhysX连续恢复。固定train/validation位姿各120秒确定性回放，不更新网络，单独记录阶段，不计入训练样本预算。

正常/故障均在Kit关闭前写摘要；中断尽量写显式weights-only，强制KILL时不承诺保存成功。每次更新另保存可安全读取的CPU张量字典用于重算GAE/掩码/样本及旧终点值。完整summary包含真实设备、TIMEOUT、checkpoint回载、固定回放与工件哈希。监督器只运行一组，保留7200秒含退出硬上限，失败不启动下一组；status/stop默认定位latest且status只读。

**预算偏差如实披露**：当前短测D/N8最慢步18.012秒；192个训练步+240个固定回放+初始化粗估约2小时20分钟，完整R2可能被两小时上限中断。不延长上限、不缩短120秒回放、不改3×64或删掉组别来声称pass；超预算返回partial/paused_on_error，待实际记录决定后续。四组GPU未运行，没有推断吞吐优化或策略性能。

CPU命令仍为下方R1完整回归命令。最终277 passed、65 warnings、12.62秒、退出0，diff检查退出0。初次collector夹具出现2 failed：用T×N大矩阵重算Critic与采集逐N批次的BLAS舍入不同；改夹具按同批次逐时刻重算，继续使用逐位相等，没有放宽动力学或验收阈值。

实际另执行以下**只准备配置、不启动任务**命令（源身份b17b0a1）：

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh scripts/new_stomach_rl/prepare_r2.py \
  --pose_manifest /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-coverage-regenerate/configs/new_stomach_v1/entry/pose_library_manifest_v1.json \
  --mask /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/new_stomach_coverage/tube_candidate_full_right_appendage_v1/candidate_face_mask.npz \
  --weights /home/multirobo/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  --capacity_summary artifacts/new_stomach_rl_capacity/capacity/20261009T125733.819352Z/summary.json \
  --capacity_boundaries artifacts/new_stomach_rl_capacity/capacity/20261009T125733.819352Z/boundaries.jsonl \
  --output_dir artifacts/new_stomach_rl_capacity/training_readiness/r2/preparation_audit
```

返回prepared_not_started、退出0；四组随后通过真实`validate_config`的哈希/有限合同检查。它们绑定历史b17b0a1 HEAD；后续代码/文档提交后不可再用作启动配置，用户须按新指南重新prepare，避免身份不匹配。历史配置与日志保留，不替换旧证据。

外部证据均不入Git，前缀绝对路径为`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/training_readiness/r2/`：

| 文件（附上述绝对前缀） | 字节 | SHA-256 |
|---|---:|---|
| regression_final.log | 354 | d2f8df52daf5b1c7a1c37eee02b49be7b093146dc64a655ea799dc00a8a2c76d |
| preparation.log | 262 | fcd7534a1bc09d30778938e7afa3215d84bee042ebc114b7788633d75f0a43c7 |
| preparation_audit/A.json | 68599 | 8a1ca485980cd07e8b81c04de366a4220d7e6dd8774b0f64b89b038a48c70290 |
| preparation_audit/B.json | 68599 | 457365850d7dd82bf225f19564a1b614bdc424d4e790d78e9329992bff4a36f9 |
| preparation_audit/C.json | 68599 | b8011a8fcde7008d66b41987dfc4cf2a9383dcd742e8620c59b50e67d4b99eec |
| preparation_audit/D.json | 68599 | 2c5e6536b601ec0ed79a09de59385950e74120ee010fd1e5d54526c7b6b4fab6 |

未验证：GPU完整采集/真实TIMEOUT/磁力调用、三次PPO更新、CUDA RNG/Adam回载及固定回放。CPU测试不是这些GPU结论的替代。原Single隔离pass、Chunk跨P0参照deferred及容量短测限制保持原记录。下一步由用户按`docs/NEW_STOMACH_RL_R2_GUIDE.md`人工启动一组，返回实际summary后再审阅，不直接启动正式开发种子。

## 2026-10-10续执行：R1组件CPU验收

最新状态仍为 `partial`：R0数学验证与R1恢复/监督组件CPU测试通过；R1与真实GPU collector的集成、R2跨回合8环境训练及恢复尚未验收。未启动GPU训练，也没有通过更改旧单环境入口绕过保护。下方R0记录作为历史保留。

本轮基准HEAD：`dbb9824d4de7e8d493bccca2249b96777aacc861`；R1实现HEAD：`437e40783f09cbe8c1baa0ba81637037a5ee11dd`。分支及工作目录不变；后续报告提交另产生HEAD，远端最终哈希在推送后单独交付。

实际新增文件：

- `learning/new_stomach_rl_checkpoint.py`：Actor/Critic、Adam、更新计数、有效样本、Python/NumPy/Torch CPU及已初始化CUDA RNG、各行位姿库RNG、统计和配置/代码/资产/冻结权重SHA-256。拒绝非有限状态、错误哈希、位姿库变化及覆盖既有checkpoint。
- `scripts/new_stomach_rl/supervise_training.py`：start/status/stop/worker接口，独立进程组、单任务锁、有限预算及进度心跳；异常paused_on_error，不重启/调参/启动下一种子。只停止自己创建的进程组。只读status不写文件；stop请求可重复。
- `tests/new_stomach_rl/test_training_checkpoint.py`、`test_training_supervisor.py`：13项新CPU测试。微型网络验证模型/优化器及随机数回载；真实短生命周期CPU假子进程验证健康慢采样（update不推进但心跳推进）、非零退出、无摘要退出、NaN、OOM、心跳停滞、用户停止。退出0缺少完整结果不算通过。

默认仅允许回合边界training checkpoint。中途保存须显式weights-only，回载清Adam和计数，不当作断点续训。恢复均调用新reset并清GRU，**不恢复PhysX接触/求解器内部状态，不承诺原轨迹连续**。当前回合边界由调用者显式声明；R2须在真实collector检查全行生命周期，不能仅传True跳过验收。旧SmokePPO继承的save/load未修改，也不能称为此严格恢复API。

监督阈值函数读取既有N=8容量证据：初始化636.046秒、最慢健康步18.012秒，启动阈值为3倍初始化取整1909秒，进度心跳阈值为10倍最慢步且至少180秒，即181秒。这是诊断等待阈值，不改物理频率。运行预算另包含终止宽限，未冻结实际训练配置或预算；不将短容量测量冒称长训练耗时。status分开记录监督器与worker心跳，避免监督器活跃掩盖采样停滞。

实际CPU命令（完整回归）：

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  WARP_CACHE_PATH=/tmp/new-stomach-scope-warp-cache \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl tests/stomach_migration tests/magnetic_following \
  tests/stomach_coverage/test_task010_recovery.py -q --disable-warnings
/usr/bin/python3 scripts/new_stomach_rl/supervise_training.py --help
git diff --check
```

结果：264 passed、65 warnings、12.24秒、pytest退出0；diff检查退出0。外部最终日志绝对路径：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/training_readiness/r1/regression_final.log`，354字节，SHA-256 `c318311837e7a2492a3b6a118f19c534b9fd092e6c96b08f88bc74f1c09f0ddd`。日志不入Git。查询新阶段train/supervisor/benchmark进程无匹配（pgrep退出1），实际没有GPU任务。

已交付监督CLI组件，但**不能提供可运行的8环境GPU训练启动命令**：R2的`scripts/new_stomach_rl/train.py`尚未交付，start会明确拒绝，不启动后台；也尚无冻结R2训练配置。CPU假worker没有创建Kit或GPU环境，CUDA RNG实际回载、GPU内存/梯度心跳、真实边界checkpoint后的新回合更新均待R2。下一步接入真实collector及A/B/C/D有限跨回合验收，由用户人工启动；不开放正式开发种子。

## 当前状态

`partial`。2026-10-10用户明确冻结8环境并要求按实施方案继续。已实施R0纯数学/CPU回归及新向量任务默认N=8；R1恢复与监督、R2真实8环境跨回合GPU训练尚未实施/运行。没有启动正式训练或再次容量长测。环境数冻结不等于总有效样本预算、序列长度、minibatch、epochs、checkpoint间隔或正式配置已冻结。

原容量证据为D组8环境8边界快速筛查，非完整P3。原Single双环境120秒隔离pass；Chunk跨P0姿态参照仍按用户授权deferred，不改写为pass。旧GPU任务已按两小时预算退出，当前实现不能宣称8环境长程训练就绪。

## 身份与实际文件

- 规划合同：`handoffs/active/NEW-STOMACH-RL-CAPACITY-20261008.md`及`docs/superpowers/plans/2026-10-08-new-stomach-rl-capacity-and-training-readiness-plan.md`的R0段落。
- 本轮基准：`ea250d567b77237cf7afda3e8ef42d9f0d9b6a42`。
- R0实现HEAD：`d63193754be6c2c3e447ba9830c6f63bbc0cfd37`。
- 分支：`feature/new-stomach-rl-capacity-20261008`，工作目录`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008`；后续文档提交会产生新HEAD，最终远端完整HEAD推送后单独交付。
- 新增`configs/new_stomach_rl/development_environment_v1.json`：记录用户批准N=8、原频率/5mm安全距离、正式训练禁用及未冻结样本预算；这是环境记录，不是可启动的正式训练配置。
- 仅修改新向量配置`robotarm_magnetic_new_stomach_rl_vector_env_cfg.py`默认N=8；旧单环境任务及基准脚本不改，诊断工具仍可显式覆盖N。
- 新增`learning/new_stomach_rl_long_rollout.py`、`tests/new_stomach_rl/test_timeout_gae.py`、`test_long_rollout.py`。不修改旧`SmokePPO`、Actor/Critic结构、分布、磁场、物理、覆盖、奖励或资产。模型更新测试只用CPU随机张量，不是Isaac Lab训练。

## 已验证的R0边界

1. `LongRollout`独立保存网络原始动作/latent/联合log_prob、实际下发history、valid、episode_start、terminated、truncated、终点Critic输入与每转移bootstrap值；采样动作被替换为投影指令时拒绝更新。
2. GAE真正终止bootstrap=0；TIMEOUT使用调用者提供的终止前Critic值，且不把下一回合优势传回来；rollout尾部使用真实边界值，不向后虚构优势。无效WARMUP/padding不进入优势/回报或损失。TIMEOUT终点取值在此仅按方案推荐进行纯数学验证，真实环境collector尚未接入，不能宣称已自动选取extras.final_obs。
3. 复用原GRU256 Actor；重置只清对应行，WARMUP/padding不推进hidden。异步两行夹具检验采集/回放输出和9D/36D联合log_prob一致，密度不除动作维数。
4. 连续序列分片保留行及时间顺序、初始hidden和episode_start，尾部补零/valid=False；padding Actor及Critic输入梯度均为零。分片可用于后续minibatch，但当前PPO只验证完整序列、原两epoch烟雾候选；未冻结正式minibatch/epochs。
5. 新`LongRolloutPPO`沿用原Actor/Critic/Adam/clip等烟雾参数，数学测试中每次仅6个有效样本，更新/梯度有限。旧单环境runner及旧save/load行为未改；不能将继承的旧烟雾checkpoint称R1严格恢复，此项待R1单独实现。

首次新增测试出现1 failed/8 passed：手算夹具reward为float64、value默认float32，比较抛dtype错误；改为同精度夹具，并让函数拒绝混合类型，未修改奖励、折扣或容差。最终新增10项全部通过，完整回归251 passed、65 warnings、11.30s、退出0；`git diff --check`退出0。

## 执行命令与证据

未创建Kit、GPU仿真、Actor训练进程或监督训练后台。实际CPU命令：

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  WARP_CACHE_PATH=/tmp/new-stomach-scope-warp-cache \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl tests/stomach_migration tests/magnetic_following \
  tests/stomach_coverage/test_task010_recovery.py -q --disable-warnings
```

最终日志：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/training_readiness/r0/regression_final.log`，354字节，SHA-256 `3be99f391ae996c9c9d7dc3ac8c135f6aeb8fca86478779a91e225ec72e0c85d`。前一250项回归也保留在同目录`regression_initial.log`。日志不入Git。

## 后续未完成

R1：实现回合边界/weights-only区分、所有RNG/位姿库/身份哈希与优化器恢复，并交付假worker覆盖故障/用户停止的只读监督器。R2：真实8环境collector获取TIMEOUT旧终点值、保存逐10Hz/1Hz证据、跨120秒回合及GRU有效掩码，并进行A/B/C/D各三次64边界更新/边界checkpoint恢复的有限验收，由用户人工启动。正式开发种子预算及超参数合同仍需单独冻结；不提供将旧单环境train_smoke的guard改为8的绕过命令。
