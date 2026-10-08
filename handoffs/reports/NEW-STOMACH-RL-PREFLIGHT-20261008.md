# 新胃连续RL预验证执行报告（进行中）

当前状态：`partial`。Gate 0–4通过；Gate 5为`limited_pass`，最大已验证1环境；Gate 6未验收。仅执行PPO短程烟雾，未启动正式训练。

## 版本及边界

- 发布基准：`1e89c9ccb48545c4513915be94d4c17d9cdebaaa`。
- 规划/实施起点、Gate 0日志记录HEAD：`8a1156a9cf349fdd730cc6d39132628b3f73f641`；测试包含此时的新增未提交代码，不把规划提交误称为实现提交。
- 分支：`feature/new-stomach-rl-preflight-20261008`。
- 工作树：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008`。
- 新增：`scripts/new_stomach_rl/validate_preflight.py`、`tests/new_stomach_rl/test_preflight.py`、本报告；项目日志更新。未改归档控制器、旧TASK-010、资产、物理参数或位姿库。
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

新增四项奖励适配、独立RewardsCfg、运行时与验收脚本；直接复用旧5秒状态机及固定权重，不改旧模块。12项奖励测试通过，包括正常/脱困/等待覆盖/锁定全过程与旧实现逐项相等、部分环境重置隔离、零面积不刷coverage奖励、严格十次加总。阶段总回归112项通过（新增全过程对照后总回归尚待重跑）。

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

命令：`ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor ./run_isaaclab.sh -p scripts/new_stomach_rl/benchmark_preflight.py --num_envs 1,2,4,8,12 --device cuda:0 --viz none`。

正式工件根路径：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/benchmark/20261008T074407.121314Z/`，包含总summary、envs_1/summary、timings CSV/JSONL、两个64步checkpoint及envs_2/summary完整错误栈。最初被主动停止的运行保存在`20261008T072356.227350Z/`，有`interruption_diagnostic.json`，不得混入正式计时汇总。129项当前回归通过、65 warnings、退出0；Gate5隔离基础单测3项通过，GPU多环境隔离未验证。

未验证：120秒正式env自动reset与全量240Hz胶囊/磁力记录；不把单环境容量验证当作12环境通过。
