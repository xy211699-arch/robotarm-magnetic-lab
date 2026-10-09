# 新胃8环境训练就绪：R0 / R1实施记录

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
