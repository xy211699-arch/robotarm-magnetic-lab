# 新胃连续RL预验证执行报告

最终状态：`partial`。Gate 0–4及Gate 6通过；Gate 5为`limited_pass`，最大已验证1环境。1秒Single/四段Chunk执行、真实C10/C1、四项奖励及A/D短PPO可训练性已验证；2环境受原保护拒绝，4/8/12未运行。仅执行PPO短程烟雾，未启动正式四组多种子训练。

| Gate | 最终判定 |
| --- | --- |
| 0 资产/库/GPU回载 | pass |
| 1 动作/安全执行 | pass |
| 2 覆盖/帧同步 | pass |
| 3 四项奖励 | pass |
| 4 独立RL/Actor/PPO smoke | pass |
| 5 并行容量 | limited_pass，max_envs=1；2拒绝，4/8/12 not_run |
| 6 train/validation 120秒与恢复 | pass，单环境 |

## 版本及边界

- 发布基准：`1e89c9ccb48545c4513915be94d4c17d9cdebaaa`。
- 规划/实施起点、Gate 0日志记录HEAD：`8a1156a9cf349fdd730cc6d39132628b3f73f641`；测试包含此时的新增未提交代码，不把规划提交误称为实现提交。
- 分支：`feature/new-stomach-rl-preflight-20261008`。
- 本报告及最终工件所审计的完整实现HEAD：`69f393cb95268eb17986f5ee0a2049386d6745aa`。本报告随后的文档提交会产生新HEAD，推送后的最终远端40位HEAD由终端交付信息单独提供，不将此实现HEAD冒称为最终文档提交HEAD。
- 工作树：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008`。
- 实际新增代码/测试完整列表见末尾“实现文件”；相对规划起点仅另修改本报告和项目日志。未改归档控制器、旧TASK-010、资产、物理参数或位姿库。
- superpowers技能当前不可用，使用逐Gate/TDD流程，未安装额外技能。

## Gate 0观察结果

| 检查 | 结果 |
| --- | --- |
| 发布语义 | 原4秒961点预览、首秒241点提交、单环境断言及5mm余量均保留 |
| 库 | 已确认位置/50mm区域的1200条，train/validation/test=1000/100/100，最终位姿唯一，原生成门槛满足 |
| 几何 | GPU Stage哈希c4028caa5da750d769a6c5e39709bd7caac975339f35eac5d36fc0e48036c72e，与位姿库一致 |
| 掩码 | 429029面重算标签差异0；排除176109、目标252920，目标面积约0.040205301m² |
| 依赖 | 七份外部磁场/配置/XRDF/URDF及原Stage哈希匹配；USD解析19层，无未解析项 |
| GPU回载 | 固定各split20条，60条全部通过；逐条完整环境及滤波重置，240步HOLD后物理/RGB有限、光心在胃腔侧、无方向长轴倾角≥45°、无终止 |
| GPU证据 | env/simulation/RGB=cuda:0；实际/physicsScene的GPU Dynamics=True、broadphase=GPU |
| 测试 | 75 passed、50 warnings，退出码0；git diff --check通过 |

HOLD允许真实磁力自然响应，不声称胶囊完全静止。源锚点本身未确认落稳，但1200个库样本各自通过原生成稳定性条件。test数据仅做资产有效性核查，未用于挑模型或调参。

版本：RTX5090、驱动595.91.07、Python3.12.13、PyTorch2.11.0+cu128；本机Isaac Sim VERSION为6.0.1-rc.7+release.42383.32955d8d.gl，IsaacLab仓库VERSION为3.0.0；运行包isaaclab.__version__实际返回12.0.0，两者分开记录，不擅自解释或重装。

## 实际命令

在上述工作树运行：

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl/test_preflight.py tests/stomach_migration tests/magnetic_following \
  -q --disable-warnings

ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/new_stomach_rl/validate_preflight.py \
  --gate assets --runtime --device cuda:0 --viz none

git diff --check
```

## 工件和工具修复

以下相对路径均以工作树绝对路径为前缀；大工件不进入Git。

- 正式摘要：`artifacts/new_stomach_rl/preflight/20261008T054338.801049Z/summary.json`；60条记录为同目录`runtime_reload.jsonl`。摘要assets字段含库/依赖的绝对路径、字节数和SHA。
- 完整工件清单：同目录`artifact_inventory.json`（交付前生成并核查），含全部绝对路径、字节数及SHA。
- 最终日志：`artifacts/new_stomach_rl/evidence/gate0_runtime_final.log`、`gate0_tests_final.log`。
- 初始诊断`gate0_runtime.log`未逐样本清空磁力滤波，且Kit关闭前没有持久化总摘要；不作为Gate通过证据。
- 第二次`gate0_runtime_reset_isolated.log`因新审计工具错误解包最近点函数返回值失败；失败摘要位于`artifacts/new_stomach_rl/preflight/20261008T054059.845404Z/summary.json`。原库和物理门槛未改，修复工具后重跑。
- TDD红灯：上述evidence目录`gate0_shutdown_red.log`、`gate0_closest_red.log`各2 failed，最终修复后通过；初始未实现检查时终端观察5 failed/1 passed。
- 固定日志地址`/mnt/isaac-linux/robotarm_magnetic_lab/logs/runtime.txt`由既有桥追加本轮记录，未改加载/日志路径。

## Gate 1观察结果

新增独立轨迹模块、action term、`validate_actions.py`及`test_action.py`；未改旧控制器。Single完整1秒执行，Chunk四个0.25秒段全部执行；持续时间相关的IK半径及解析速度/加速度极值认证，段间位置/速度/加速度连续，保留原停止路径和原5mm余量。

- 纯回归：25 passed、14 warnings，退出码0；原961点预览保持原样。
- GPU：Single与Chunk各21场景，合计42；每场景240物理记录及240次真实磁力更新，所有原跟踪容差通过、无非有限状态/终止，均有实际可运动指令。包含HOLD、9轴正负、组合和换向；未通过在真实场景中制造碰撞来测试拒绝，碰撞拒绝由纯函数低于5mm证书测试验证，不夸大此项。
- Single历史按四个实际四分之一秒指令差分记录，并非复制四次整秒增量；Chunk最后段影响末段，四段预算与解析加速度限制单测通过。
- 首次Chunk输入检查失败：SDK实测Ball软限位为`[-inf,+inf]`，其float32软限位计算对连续关节极大区间溢出。修复仅在新增模块接受原合法无界区间，仍拒绝NaN/反向区间；未替换SDK、修改关节限位或降低速度/碰撞保护。失败摘要保留。
- Chunk通过摘要：`artifacts/new_stomach_rl/action/20261008T060439.262435Z/summary.json`；Single通过摘要及全部逐步记录路径、字节数、SHA见`artifacts/new_stomach_rl/evidence/gate1_artifact_inventory.json`（均以本工作树为绝对路径前缀）。

实际GPU命令：`ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor ./run_isaaclab.sh -p scripts/new_stomach_rl/validate_actions.py --mode chunk --device cuda:0 --viz none`，Single仅将mode改为single。

## Gate 2观察结果

- 新增独立精确可见性/双时钟模块，复用原CUDA first-hit、入射面片、法向、距离及面积权重；70mm/120°及冻结管道掩码保持不变，面积实测0.04020530122973119m²。
- 120秒GPU回合通过：28800物理步；C10=1201点、C1=121点（含C0），每秒真实RGB帧号+1、内容SHA留存；共同边界可见顶点差异全部0；C1累计集合是C10子集。结束C10=26.237952%、C1=26.096835%，仅为脚本链路验证，不称为策略性能。
- 初始回载采用冻结库首个固定train样本，1秒HOLD不计入策略预算；C0不产生策略奖励（奖励适配尚待Gate3验证）。
- 解决新增适配问题：Python/渲染器约1nm光心舍入令1个边缘顶点first-hit不同；改为SDK相同Warp坐标转换，未量化外参或放宽可见性条件。PhysX step不主动同步Fabric，使用公开`physics_manager.forward()`及仅层级变换刷新，从真实物理状态获取挂载相机变换；逐样本检查与胶囊刚体及挂载关系匹配，绝不读取过期1Hz相机位姿作为C10。
- SDK相机第二个边界未自动采图：复用已验证Task009D0RgbSynchronizer，仅对遗漏边界补采，120秒中52次补采（直接统计摘要）；帧号仍每秒只增加1。新的PolicyRGBBoundary将供新正式环境观测路径使用，Gate4尚未验收，不声称已接入正式Actor。
- 兼容风险：pose-only路径读取Camera._view及SDK Warp约定；补采使用既有Camera._update_buffers_impl接口。SDK升级必须重跑门禁。
- 纯覆盖测试11 passed；Gate0–2及原回归合计104 passed、65 warnings，退出码0。

命令：`ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor ./run_isaaclab.sh -p scripts/new_stomach_rl/validate_preflight.py --gate coverage --seconds 120 --device cuda:0 --viz none`。

通过摘要：本工作树`artifacts/new_stomach_rl/preflight/20261008T063439.430899Z/summary.json`；同目录逐10Hz/1Hz JSONL。全部绝对路径、字节数、SHA及保留的初始失败工件见`artifacts/new_stomach_rl/evidence/gate2_artifact_inventory.json`。测试当时已提交HEAD为`0d8764ec88cabd04c32485a7ab60a8e5674524e9`，包含尚未提交的Gate2新增代码；不将该HEAD误称为Gate2实现提交。

## Gate 3观察结果

新增四项奖励适配、独立RewardsCfg、运行时与验收脚本；直接复用旧5秒状态机及固定权重，不改旧模块。12项奖励测试通过，包括正常/脱困/等待覆盖/锁定全过程与旧实现逐项相等、部分环境重置隔离、零面积不刷coverage奖励、严格十次加总。阶段总回归112项通过，最终完整回归已覆盖新增全过程对照测试。

实际GPU train脚本120秒通过：初始化1秒HOLD返回零策略奖，C0不奖励；28800个实际物理步，C10 1201点、C1 121点；仅four_terms一个RewardManager项，无继承alive/action_rate/joint_velocity/collision奖励。每秒四项之和、十次0.1秒采样之和与env.step返回奖励误差小于1e-5，全部有限。覆盖停滞进入原无进展/锁定状态，不扩大幅度以使结果好看。此测试证明奖励一致性，不证明新磁控下阈值最优。

通过摘要：本工作树`artifacts/new_stomach_rl/preflight/20261008T065553.947988Z/summary.json`；同目录`reward_10hz.jsonl`、`reward_1hz.jsonl`及`physics_240hz.jsonl`。完整绝对路径/字节/SHA见`artifacts/new_stomach_rl/evidence/gate3_artifact_inventory.json`。测试加载Gate3未提交代码，原执行HEAD为`aff567e32539d9c4f67a8bb916fcf45051a41303`。

命令：`ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor ./run_isaaclab.sh -p scripts/new_stomach_rl/validate_preflight.py --gate rewards --seconds 120 --pose_split train --device cuda:0 --viz none`。validation随机120秒追加验收也通过，摘要为`artifacts/new_stomach_rl/preflight/20261008T070409.039061Z/summary.json`，C10/C1均正确计数，所有奖励汇总一致。

验收脚本将TIMEOUT设置为121秒并在120秒精确停止，以保留最后一秒证据，未改碰撞/停止保护；独立正式配置仍为120秒，其Same-Step autoreset及正式Actor接口将在Gate4验证。Gate3的240Hz文件记录实际电机状态，尚不含逐步胶囊/磁力全量数据，不能冒称Gate6最终动力学记录。

## Gate 4观察结果

新增独立task ID `Template-Robotarm-Magnetic-New-Stomach-RL-Preflight-v0`及环境步进/重置适配；旧注册模块不改，新CLI显式注册。Actor仅548D（冻结视觉512D或全零 + 上一秒已下发的四个实际指令36D）；Critic特权输入28D单独声明为胶囊pose7/速度6/关节9/恢复阶段4/C10-C1两项。重置写入冻结库位姿，只用于reset；随后实际240步HOLD不计预算、不产生奖，清空动作历史后初始化C0。

初始9项纯测试通过；后续重置审阅发现C0特权Critic沿用上一回合last_step恢复阶段，新测试1 failed/4 passed复现。首轮性能运行在130步主动停止、不计验收（无worker最终摘要，父程序正确记blocked，不以进程退出码0误判成功）。仅修改新增运行时读当前已重置tracker阶段，Actor/旧状态机/奖励权重不变。另补float32 tanh饱和测试，用存储的有限latent计算严格变换密度。修正后的观察/PPO/隔离/trace合计16项通过。

实际GPU正式env.step路径：A与D各2次更新、每次8步rollout，策略/价值loss、KL、梯度、reward全部有限；checkpoints恢复后相同观察/hidden的输出完全相同。重置后的覆盖、动作历史、奖励清空；加载checkpoint再在新reset下推进一个1秒步成功，不宣称恢复了旧物理轨迹。D使用原SHA固定的frozen ResNet18，真实1Hz RGB，不更新backbone。

| 模式 | update | policy loss | value loss | KL/维 |
| --- | --- | --- | --- | --- |
| A | 1 | -0.114123 | 41.157551 | 0.004012 |
| A | 2 | -0.098494 | 0.094735 | 0.004432 |
| D | 1 | -0.178675 | 46.820194 | 0.005096 |
| D | 2 | -0.110727 | 0.041395 | 0.002163 |

Smoke用每秒gamma=0.999^10、lambda=0.95，Adam 3e-4、clip0.2、value系数0.5、每update两个完整recurrent minibatch epochs；均为预验证参数，正式训练前待冻结。联合log_prob不除维数，KL/entropy诊断按维平均，避免9/36维总量误比。短烟雾遇终止即失败，不把未测的长训/timeout GAE行为写成已验证。

命令：`ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor ./run_isaaclab.sh -p scripts/new_stomach_rl/train_smoke.py --group A --updates 2 --num_envs 1 --rollout_steps 8 --device cuda:0 --viz none`；D仅将group改为D。修正后A/D通过摘要分别为`artifacts/new_stomach_rl/smoke/20261008T073515.409778Z/summary.json`、`20261008T073949.034329Z/summary.json`，D额外实测两次C0 Critic正常阶段。早期071255/071628短测保留，不作为Critic修正后的最终验收证据。工件完整路径/字节/SHA在最终清单中记录。修正后测试执行已提交HEAD为`ffae824142ec6b5067e3cea4209fc30dd0e8a331`，包含当时新增未提交修正代码。

## Gate 5观察结果：limited_pass，最大1环境

按固定顺序实际尝试1、2环境。1环境完成两次8步warmup+32步测量、两个64步rollout，总计208个1秒动作；两次64步后的PPO更新loss/梯度有限，优化耗时0.133516/0.082929秒。物理及相机仍原频率/原分辨率，10Hz射线仍精确实现。

2环境在原`ActuatorVectorAction.__init__`被拒绝：`ValueError: single environment, 1Hz boundary, 240Hz physics required`。实际Stage已构造双环境，但不执行非法双环境动作；不是OOM或驱动故障。保留原保护，没有删断言或修改共享核心；4/8/12按门禁顺序未运行。轨迹/测量索引、停止路径与碰撞缓存、world_mesh路径、双时钟/部分reset仍须按env隔离后才可能扩展，不能把基础Actor/Reward纯函数隔离测试当作多环境GPU通过。

| 测量（D，1环境） | 结果 |
| --- | --- |
| 64个计时步吞吐 | 0.271524 env_steps/s = transitions/s |
| 1秒控制步平均墙钟 | 3.682917s |
| 规划 | 0.226565s/步 |
| 磁力更新 | 1.319501s/步 |
| 10Hz几何覆盖 | 0.393288s/步 |
| 四项奖励 | 0.054337s/步 |
| RGB采集同步API | 0.016651s/步 |
| 冻结视觉编码 | 0.001963s/步 |
| 执行器总计（嵌套） | 0.747592s/步 |
| 碰撞查询（嵌套） | 0.292275s/步 |
| 驱动显存采样最大值 | 6421MiB，不称瞬时硬件峰值 |
| CPU进程峰值RSS | 9294956KiB |
| 提交/缩小投影/安全HOLD比例 | 100% / 4.8077% / 0% |

嵌套计时不能相加：执行器含覆盖回调和部分安全检查，碰撞查询亦包含规划过程；render API仅计调用墙钟，不能代表完整GPU渲染核耗时。外层step在CUDA完成后计时，未包含nvidia-smi采样开销。完整原始timings提供复核。

预算推算：按当前**单环境实测吞吐**，同等768000 transitions/seed约785.689小时（32.74天），不含启动/评测/额外工件；12环境尚未支持，不允许假定12倍加速。若改用1×64×1000只需64000 transitions，约65.474小时，但那将改变正式rollout采样量，不能未经冻结直接采用。当前瓶颈主要在物理/磁场和安全执行链路，CNN仅约2ms。正式A/B/C/D多种子未启动，不以本次小PPO的覆盖率判策略性能。

若四组各三种子均使用同等768000 transitions且吞吐假设与D单环境一致，顺序占用同一GPU的纯采样预算约9428小时/393天；此假设并未实测A/B/C的性能，仅说明当前容量不足以合理安排完整研究。12环境分支尚不存在已验收能力，不能在本报告中推荐直接启动正式四组实验。

命令：`ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor ./run_isaaclab.sh -p scripts/new_stomach_rl/benchmark_preflight.py --num_envs 1,2,4,8,12 --device cuda:0 --viz none`。

正式工件根路径：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/benchmark/20261008T074407.121314Z/`，包含总summary、envs_1/summary、timings CSV/JSONL、两个64步checkpoint及envs_2/summary完整错误栈。最初被主动停止的运行保存在`20261008T072356.227350Z/`，有`interruption_diagnostic.json`，不得混入正式计时汇总。129项当前回归通过、65 warnings、退出0；Gate5隔离基础单测3项通过，GPU多环境隔离未验证。

不把单环境容量验证当作12环境通过。最终正式env自动reset与全量240Hz胶囊/磁力记录见Gate6。

## Gate 6观察结果：pass

最终检查直接调用新增正式`env.step()`及原120秒TIMEOUT。使用修正后D烟雾检查点`update_0002.pt`，以确定性均值推理，不更新模型/优化器/归一化统计，train与validation分开启动独立进程；无test位姿参与。`record_physics`只读记录原物理更新，默认关闭，不改变动作、物理或奖励。

- train-0003回合通过：28800条真实240Hz胶囊pose/速度、实际磁体pose、磁力/磁矩及关节实测/下发参考；C10含C0为1201点，C1为121点，RGB每秒恰好+1；四项之和与每秒返回值匹配。结束C10=27.197598%、C1=27.039809%，仅为检查点端到端预检结果，不评价正式策略性能。
- 同一回合四项累计为`[15.3431490064, 0.1068960079, -0.8120000386, 0]`。120秒最后边界真实触发TIMEOUT；终止观察从`extras.final_obs`保存，返回的新回合观察不会被误记为旧回合终点。原终止帧已下发历史保留，重置后覆盖/history/reward/episode计数清空、Critic恢复normal，重新计算C0=0.118544502057，再执行一个真实策略步成功。
- train正式摘要：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T080942.758742Z/summary.json`。执行所审计HEAD为`7ea3dbf58084d79de949242f087b017fce94b452`，包含当时尚未提交的Gate6代码；最终代码哈希在完整工件清单内冻结。
- validation-0006独立回合也通过：同样28800条真实物理记录、C10 1201点、C1 121点，真实帧号逐秒+1、有限状态、四项汇总和自动reset/恢复均通过。结束C10=19.749334%、C1=19.730628%；累计四项`[7.6323069632, 0.0386681244, -1.6100000765, 0]`；重置C0=0.121170258484。两个回合全部120条策略边界投影比例均为1，没有额外安全HOLD或设备终止；最后仅原120秒TIMEOUT。
- validation摘要：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T082615.445313Z/summary.json`。两次独立进程均从检查点恢复后再reset；终止后显式清空推理GRU hidden并推进新回合首秒，未继承旧覆盖/历史。两份摘要均`status=pass`，进程退出码均0；不是仅依据退出码宣称通过。
- 先进行4秒缩短TIMEOUT生命周期诊断，960条记录、41/5覆盖点及自动reset/后续步均通过；此项不代替120秒验收。
- 最终回归：129 passed、65 warnings、7.78秒、退出码0；`git diff --check`退出0。此为本轮实际重跑，不引用发布时65项。

GPU命令（两次依次运行，SPLIT分别为train和validation）：

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/new_stomach_rl/validate_end_to_end.py \
  --checkpoint artifacts/new_stomach_rl/smoke/20261008T073949.034329Z/update_0002.pt \
  --seconds 120 --pose_split "$SPLIT" --device cuda:0 --viz none

env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl tests/stomach_migration tests/magnetic_following \
  tests/stomach_coverage/test_task010_recovery.py -q --disable-warnings
```

## 实现文件（相对规划起点）

下列脚本、运行模块和测试全部为新增独立新胃预验证实现；只有既有`docs/PROJECT_RUN_LOG.md`追加记录，没有修改旧任务源码。新增环境内部的后续修补仍属于这些新增文件，不改变旧发布行为。

```text
scripts/new_stomach_rl/
  validate_preflight.py
  validate_actions.py
  validate_coverage.py
  validate_rewards.py
  train_smoke.py
  benchmark_preflight.py
  validate_end_to_end.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/
  coverage/new_stomach_rl_coverage.py
  learning/new_stomach_rl_actor.py
  learning/new_stomach_rl_distribution.py
  learning/new_stomach_rl_runner.py
  runtime/new_stomach_rl_library.py
  runtime/new_stomach_rl_optics.py
  runtime/new_stomach_rl_reward.py
  runtime/new_stomach_rl_rgb.py
  runtime/new_stomach_rl_runtime.py
  runtime/new_stomach_rl_trace.py
  tasks/manager_based/robotarm_magnetic_lab/
    controllers/new_stomach_rl_trajectory.py
    mdp/new_stomach_rl_action.py
    mdp/new_stomach_rl_terms.py
    new_stomach_rl_env.py
    robotarm_magnetic_new_stomach_rl_env_cfg.py
tests/new_stomach_rl/
  test_preflight.py
  test_action.py
  test_coverage.py
  test_reward.py
  test_observation.py
  test_ppo.py
  test_multi_env.py
  test_trace.py
handoffs/reports/NEW-STOMACH-RL-PREFLIGHT-20261008.md
docs/PROJECT_RUN_LOG.md
```

## 正式训练前的边界

- A/D已做代表性GPU短PPO，B/C结构和隔离由同一实现及单测验证，未声称四组正式实验已经运行。smoke训练器不是正式长程训练入口；目前遇到终止会拒绝继续训练，timeout的GAE bootstrap、跨回合GRU、正式参数/随机数恢复必须在正式长训方案中单独冻结和验收。检查点恢复证明输出和后续新回合步进一致，不证明随机物理轨迹可原样断点续跑。
- 本预验证最大环境数为1。现有单索引磁控/碰撞链路的多环境改造需要新的架构授权与隔离验收；不能删除原单环境断言后直接启动12环境。预算表仅按D单环境实測；未量得A/B/C完整吞吐，不推断同速。
- 外部Stage及其层、URDF、vendor、位姿库、掩码和ResNet权重仍位于本机固定路径，均保留既有资产。代码分支并非脱离这些依赖即可在另一台机器完整运行的自包含资产包；外部清单提供迁移审计，不将大资产偷偷加入Git。
- 相机补采和pose-only路径含Isaac Lab内部接口；升级SDK时必须重跑帧同步、外参和全部门禁。未做sim2real、模型性能对比、视觉依赖性结论或接触参数再标定。
- 合同示例中的`--headless`不是本机CLI有效参数；实际使用`--viz none`。没有改变物理时间、镜头分辨率、FOV、覆盖距离/权重、磁参数或安全距离来提高吞吐。

## 最终工件与复核

大工件不进Git。本次全部成功、失败、中止工件保留，完整清单含148个文件、224039444字节，并逐文件提供完整绝对路径、字节数、SHA-256；另含外部依赖及执行代码的SHA快照。清单本身排除递归自哈希，其身份为：

`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/evidence/final_artifact_inventory.json`

53515字节，SHA-256：`6ce46685d8b92950ae0db7a54b741823bdb168d0534d15fb31efefd6e4f98c2f`。

交付前重新读取清单全部文件、外部依赖和代码快照，大小及SHA均匹配。执行命令原始argv及对应成功/失败摘要关联，保存于下表的`final_executed_commands.json`；命令重放应使用正文给出的本机wrapper写法，原始argv含Kit启动时附加的内部参数，不直接当Shell命令复制。

以下是核心证据；逐物理步/10Hz/1Hz原始文件、全部性能CSV和checkpoint、各Gate红灯及失败日志的完整路径/字节/哈希均在上述清单中，不只提供目录名：

| 工件完整绝对路径 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/preflight/20261008T054338.801049Z/summary.json` | 81487 | `3d814c99d919817e6a67910576025600669abe01e11213c57ce944d653d826d4` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/action/20261008T060439.262435Z/summary.json` | 139193 | `5575c6912fd6c8228e60afdf6ae5a4d80d0bede1d2711ca2e95d58dbc1ff41dc` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/action/20261008T060717.448392Z/summary.json` | 112288 | `610d1c377659279be8f5b32c7b39a608d9683c4e77b36508c98e98969868a395` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/preflight/20261008T063439.430899Z/summary.json` | 139584 | `32db76110eb8519c57224342f948db06158130daccfa15ac8f1d1e1c6d3af13b` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/preflight/20261008T065553.947988Z/summary.json` | 47601 | `f9c22b59457f12866f3840025115fb7bce2aa7b57bc26f3fcfb3f3707b5e7f28` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/preflight/20261008T070409.039061Z/summary.json` | 49026 | `761f7534f7b4fa4d007c2656974a24d9f409a0fbbd23788772dcc65d667d6379` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/smoke/20261008T073515.409778Z/summary.json` | 7407 | `6da5224550dc7d67f9bd9e12ca717185a63417a59deb9e44669b64c696c93755` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/smoke/20261008T073949.034329Z/summary.json` | 7501 | `3229da311f580a1278228f1caa77e62f79fb07bc6f52f2fb3bac77a36f2b0850` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/benchmark/20261008T074407.121314Z/summary.json` | 2397 | `0132a03f49b3b63908ddd415a9a82c946e37c5d281140b4c4755b8c8fa4a7f52` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/benchmark/20261008T074407.121314Z/envs_2/summary.json` | 9155 | `e05db185e5372e169a98fd96a026af5cdd0e1c33190b1d633503fe32678dde49` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T080942.758742Z/summary.json` | 11104 | `91f64dba5c5c019d0831e1cf50fdea080ff2a6ca4d3c50ee58926bf14e15bd52` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T082615.445313Z/summary.json` | 11186 | `bcaddec292519b3f998a09ac7526bc8971104ea39e3b1e7e72f047f3fa1bed8e` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/evidence/preflight_regression_final.log` | 193 | `35d1c084f8efcf28c01b6286eadc4e04f758dfb3d8cc68904e4aced32fd74607` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/evidence/final_executed_commands.json` | 15754 | `10df4c363e662e63498f9dd0e48ccb17e816e353a3f7c53d42b0e0b41352e004` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T080942.758742Z/physics_240hz.jsonl` | 39956851 | `b99bbbf44ecf9b8ae24f4e6dad885db11ae56f0dca6e32afeea2bb3357aa77ac` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T080942.758742Z/coverage_reward_10hz.jsonl` | 858708 | `aaa4012b5734d6e6aaa2bdc4dc48591acb3f14645d6ebdcb397e1be4c68cccc7` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T080942.758742Z/policy_rgb_1hz.jsonl` | 680448 | `a03d1b11f749246d27ccbe3582f1a9fd41c52561575630a4ac94e71c26f1994f` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T082615.445313Z/physics_240hz.jsonl` | 39885828 | `2dbeeeff2fe2436e136e282c7b499142d6bcea726b85b3dc0f0d7c91692acfbd` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T082615.445313Z/coverage_reward_10hz.jsonl` | 847637 | `dd4eb3a3654c79706ed2fd0cb890a77ab8d08f853bc5bfbbfd4f468a49bffa62` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/end_to_end/20261008T082615.445313Z/policy_rgb_1hz.jsonl` | 679445 | `2be66b18c301b56ce240367ddfc0ca1f19a9c2e4dda6d3a29373b3455a6a17fd` |
