# TASK-009C 同步随机基线预实验执行报告

## 最终状态

`complete`：Gate 1--6 全部通过。35 个随机策略回合与 2 个 HOLD 诊断回合均完整，无失败、
无重试、无恢复跳过；正式批次共产生 111037 个严格对齐的覆盖点。

## 执行基线

- Windows 规划分支：`workflow/TASK-009C-synchronous-random-baselines`
- 规划提交：`98dbd2f951d53962f8b3f7a2ab0f857d8aa89cb5`
- 精确 TASK-009B 基线：`64dd2ff33951cb780f938a81c91c22dde8764c93`
- Linux 实现分支：`feature/TASK-009C-synchronous-random-baselines`
- 冻结配置 SHA-256：`ee734310b167cab3d89622b80c3390a4b143d2d034770a80bd9d35de1eccc1c4`

规划提交已确认以精确 TASK-009B 基线为祖先。实现未修改 USD、质量、惯量、碰撞、摩擦、
重力、相机内参、70 mm 可见距离、面积权重、不可达区域、入口区域、位姿库、控制方向、
作用点或冻结力度。

## Gate 1：纯策略与配置

状态：`pass`。

实现 R1--R7 与 HOLD，策略只接收自身 RNG 和动作历史。配置展开保存 8 个冒烟回合与 37 个
正式回合的位姿、环境种子、策略种子和顺序。纯策略测试 20 项通过。

提交：`3c06b99faf82f8a117aa92e814d00c15985b2730`。

## Gate 2：指定位姿 reset

状态：`pass`。

五个 validation 位姿均在基础 reset 后、10 个 HOLD 前写入；根速度和动作残余清零，写入
误差、连续 RGB、最后帧非零 C0 和 `episode_length_buf=0` 均通过。GPU、PhysX、仿真和相机
设备均为 `cuda:0`。

外部证据：

- 目录：`/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009c_random_baseline_preexperiment/reset_only-20260827_091234_179170Z`
- reset JSONL：5069 字节，SHA-256 `a1ef21f3641b2c795bf7472def79e6df61cbd2783fd86dc055bc77932e8b3da3`
- summary：858 字节，SHA-256 `b7a228a857d86a990e485b48c10d12d2ae25784bfa1489f9937b9f419b4c8a25`

提交：`62a3b98`。

## Gate 3：同步回合与汇总测试

状态：`pass`。

严格校验 3 秒回合的 31 点/720 子步和 300 秒回合的 3001 点/72000 子步。主动构造缺点、
乱序索引、非 0.1 秒时间、RGB 帧错位、覆盖下降、非有限状态、提前终止和错误子步，均被
拒绝且未修复。正式汇总额外验证每策略精确样本数、3001 点精确对齐和候选时刻边界。

合同命令结果：47 项通过，49 条上游弃用警告，无测试失败。

提交：`c6743f75864bd839f332d38a72e72cf7977466ab`。

## Gate 4：GPU 冒烟

状态：`pass`。

`validation-0006` 上 R1--R7 与 HOLD 共 8 个 3 秒回合全部得到 31 个严格同步点；设备全部为
`cuda:0`，稳定指针和运行清单哈希经独立汇总命令复核。低覆盖和高覆盖都按有效结果保留。

- run ID：`smoke-20260827_092252_202602Z`
- 运行清单：8731 字节，SHA-256 `ac418288c0c5f4dfea6fab1fd48923c4deee7da3b2777348887f8e71d9a53c26`
- 稳定指针：419 字节，SHA-256 `9b9d805a414b2ca5a5edcf15640ec9f8b83c96d503d22d23661e553d1136548c`
- 批次 summary：1037 字节，SHA-256 `0ad67071ac121af3b75344977889db5bb0a8d6a5dd6fb884d70a2ab6c322bf24`

启动包装器会为所有 Python 程序追加 `--kit_args`。纯离线汇总程序最初未接收该参数，导致
校验命令在解析阶段退出；加入“接收但忽略”兼容参数后，以合同原命令复测通过。该修复不
改变实验数据或统计。

## Gate 5：37 回合正式预实验

状态：`pass`。

运行 ID：`formal-20260827_092532_645812Z`。运行器按位姿外层、冻结策略排列为内层，完成一
个回合即校验、原子落盘、追加清单并更新稳定指针。只有完整且哈希一致的回合才允许恢复
跳过。

正式进程从 2026-08-27 09:25 UTC 运行至 14:11 UTC。37/37 回合全部含 3001 点，运行清单
共 39 条记录（1 条开始、37 条通过、1 条完成），`episode_failure=0`、恢复跳过数为 0。
GPU 环境、仿真、PhysX 与相机设备均为 `cuda:0`，物理/控制频率实测为 240/10 Hz。

五个位姿、七种随机策略的 300 秒最终可达面积覆盖率如下；每个单元格代表一个完整正式
回合，因而覆盖全部 35 个随机回合：

| pose | R1 | R2 | R3 | R4 | R5 | R6 | R7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation-0006 | 62.587% | 93.898% | 99.973% | 52.783% | 99.391% | 99.836% | 99.814% |
| validation-0011 | 54.682% | 99.459% | 99.921% | 99.934% | 99.451% | 99.943% | 99.882% |
| validation-0015 | 95.393% | 99.955% | 99.975% | 99.550% | 45.340% | 99.758% | 99.989% |
| validation-0017 | 86.869% | 99.978% | 96.839% | 99.998% | 99.916% | 99.878% | 99.516% |
| validation-0019 | 91.038% | 49.834% | 99.519% | 99.944% | 96.141% | 74.626% | 91.088% |

HOLD 两个诊断回合也均为 3001 点：`validation-0006` 为 2.6500%→2.6983%，
`validation-0019` 为 7.2511%→7.4728%。HOLD 的微小变化来自零 Actor 力条件下仍然存在的
重力、接触、摩擦和自然运动，不代表残余主动控制。

正式工件：

- 运行清单：37241 字节，SHA-256 `7200a354acb9732edb22884bb73e1b9219d1ce2547bedd313afec63807b53365`
- 稳定指针：423 字节，SHA-256 `6a06430f10458454f934810acc938b10cd3da361cb46251b8c0a32d55349b7b5`
- 批次 summary：1043 字节，SHA-256 `3af66c9bd8721841a3663604b3329ad78cf627ab79e5b728eb0d3c4e59899625`
- 全量工件清单：526 个文件、344848146 字节；清单本身 206219 字节，SHA-256
  `ca5d9e145c58329363ee9e22ea5f310221f12a110895112c9d662169cd102d3c`

完整的 37 个边界日志和摘要的绝对路径、字节数与 SHA-256 均在运行清单中；覆盖快照、mask、
轨迹、各回合元数据和 Gate 6 输出的逐文件证据均在 `artifact_inventory.json` 中，报告不重复
粘贴 526 行清单。

## Gate 6：汇总、绘图与报告

状态：`pass`。

汇总程序重新验证稳定指针、运行清单与所有 37 个日志哈希、精确回合集合、策略—位姿配对、
3001 点时间对齐、逐边界 RGB 同帧、覆盖范围和单调性。未进行插值、补零、前值填充或平滑。

300 秒可达面积覆盖率的五位姿均值（HOLD 为两个位姿）如下：

| policy | n | mean | std | minimum | maximum |
|---|---:|---:|---:|---:|---:|
| R1 | 5 | 78.114% | 16.324% | 54.682% | 95.393% |
| R2 | 5 | 88.625% | 19.531% | 49.834% | 99.978% |
| R3 | 5 | 99.245% | 1.215% | 96.839% | 99.975% |
| R4 | 5 | 90.442% | 18.830% | 52.783% | 99.998% |
| R5 | 5 | 88.048% | 21.396% | 45.340% | 99.916% |
| R6 | 5 | 94.808% | 10.091% | 74.626% | 99.943% |
| R7 | 5 | 98.058% | 3.489% | 91.088% | 99.989% |
| HOLD | 2 | 5.086% | 2.387% | 2.698% | 7.473% |

描述性建议：R3 在 180 秒时均值已达 98.006%，到 300 秒仅增约 1.24 个百分点；R7 从
180 秒的 86.557% 增至 300 秒的 98.058%，对更长预算仍有明显收益。R1/R2/R4/R5/R6 的
位姿间方差较大。后续训练回合时长可优先比较 180、240 与 300 秒的算力—覆盖权衡，但本
预实验不自动冻结最终时长。

Gate 6 工件：

- 主图 PNG：346180 字节，SHA-256 `a323775e098f6a63b0525ee6398c9cde6d50fd9fbfc5834df346dc964894e11e`
- 可编辑 SVG：100558 字节，SHA-256 `d16ab1863993422423027e1d5dea310a56217ec3de0ba30dc323d3d3986f9b91`
- 增量审计图：342430 字节，SHA-256 `96d68a73c26196951f32e74000a8045cd0e941bbf05b7f643431e477ec7eeb05`
- 10 Hz 曲线 CSV：4236472 字节，SHA-256 `32ffba20fa45149d3144d1e417055979262cfc449e5431ef33830d1b76faf91e`
- 候选时刻 CSV：4246 字节，SHA-256 `b4df49895a438f26b8f26da941e2dfcfc6d95c1d04ae5815613c4002cab38ad3`
- 回合摘要 JSON：51062 字节，SHA-256 `a4e4db74f576863d94ffa1f9ec66c09295c4e92b2f3ecc06cb08c8b582e9e53b`

所有正式汇总路径均位于：

```text
/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009c_random_baseline_preexperiment/
formal-20260827_092532_645812Z/summary/
```

## 修改文件

```text
configs/task009c/random_baseline_preexperiment_v1.json
scripts/stomach_coverage/run_random_baseline_preexperiment.py
scripts/stomach_coverage/summarize_random_baselines.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/baselines/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/baselines/random_policies.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task009c_episode_runner.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/task009b_training_env.py
tests/stomach_coverage/test_task009c_episode_protocol.py
tests/stomach_coverage/test_task009c_random_policies.py
tests/stomach_coverage/test_task009c_reset_pose.py
tests/stomach_coverage/test_task009c_summary.py
docs/TASK009C_RANDOM_BASELINE_PREEXPERIMENT.md
docs/PROJECT_RUN_LOG.md
handoffs/reports/TASK-009C-synchronous-random-baseline-preexperiment-report.md
```

## 偏离项与未验证项

- 未改变合同、策略、种子、时钟、物理或覆盖定义。
- 正式清单中的 `active_reset_events` 初版扫描记录为事件模式 `reset`；同一进程的 Event
  Manager 启动表明确显示该模式唯一事件为 `reset_scene`，没有光照、材质、摩擦或资产随机
  事件。后续扫描已修正为结构化名称 `reset:reset_scene`。该字段精度问题不改变正式动力学。
- 长时运行中 RTX 曾报告一次 SyntheticData frame discarded；运行器随后继续推进，当前
  回合最终仍通过 3001 点、逐帧递增及 Actor/覆盖同帧复核，因此没有按异常回合重试。
- 本任务只建立无学习随机基线，不把每策略五个位姿结果描述为论文正式统计证据。
