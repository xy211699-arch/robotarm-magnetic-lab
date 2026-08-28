# TASK-009D0 向量化训练基础设施执行报告

## 最终状态

`partial`：Gate 1 和 Gate 2 通过；Gate 3 双环境一致性仍为失败。2026-08-28 用户明确判断
该环境差异可忽略，并授权 Linux 执行端在不改判 Gate 3、不放宽阈值的前提下继续 Gate 4--6。
后续结果属于“人工豁免后继续执行”的补充证据，不能据此声称全部合同门禁通过。未修改 USD、
胃部资产、相机参数、70 mm 可见距离、覆盖权重、不可达区域、位姿库或控制器力度。

## 执行边界

- Windows 规划分支：`workflow/TASK-009D0-vectorized-training-infrastructure`
- 规划 HEAD：`35273aa3ae5a2a33f3a443028f720f0c4308d608`
- 精确代码基线：`7c4c5a18780b980ad3882ce75f1d64733fc3080d`
- Windows 设计提交：`f8eb6b825aa8e5765b3db52532b169a9d299066e`
- Linux 实现分支：`feature/TASK-009D0-vectorized-training-infrastructure`
- 本报告审计的 Gate 3 实现 HEAD：`123995f4e8fbdcad3461e5d10bd49a45768aa480`

代码基线和设计提交均已验证为规划 HEAD 的祖先。本报告提交会产生新的最终 HEAD；推送后
由 Linux 终端交付信息单独给出本地与远端完整哈希。

## 运行环境

- Isaac Lab Git HEAD：`d23a06799c946a4581a6245ea3105dbfb387cb20`
- Isaac Lab 包版本：`12.0.0`
- Kit：`110.1.2+production.326809.f9bf0dda.gl`
- Python：`3.12.13`
- PyTorch：`2.11.0+cu128`，CUDA：`12.8`
- Warp：`1.15.0.dev20260626`
- RSL-RL：`5.4.1`
- GPU：NVIDIA GeForce RTX 5090，32607 MiB
- NVIDIA 驱动：`595.84`
- 环境、PhysX 状态、相机张量和 Warp 射线设备：`cuda:0`

运行时观察到 GPU dynamics 会禁用 CCD 的 Isaac Lab 警告；该事实未被隐藏。TASK-009D0
没有通过改用 CPU 获得门禁结果。

## 实现内容

新增独立任务、严格候选配置、split 安全位姿采样、批量参数化力 ActionTerm、逐环境面积覆盖
累积器、批量 Warp 第一命中可见性、逐环境 RGB 帧同步、Actor/特权观测隔离，以及同步
120 秒回合/reset 生命周期。旧任务 ID 和旧标量运行时未被重定向。

实现中修复了两项只会在多环境出现的坐标问题：冻结位姿应加每行完整 `env_origin`；冻结
不可达配置应在移除 env_0 克隆平移并重算几何哈希后校验。上述修复有纯测试覆盖。

本任务没有新增 CNN、GRU、Actor、Critic、PPO、RSL-RL 配置、VLM、奖励缩放或课程学习。

Git 修改范围仅包括：

- 配置：`configs/task009d0/vectorized_training_candidates_v1.json`。
- 运行脚本：先决条件检查、单环境等价、双环境隔离和 GPU reset-sync 四个 TASK-009D0
  脚本；reset-sync 因 Gate 3 失败未运行。
- 源码：`runtime/task009d0_config.py`、`runtime/task009d0_pose_batch.py`、
  `runtime/task009d0_coverage_runtime.py`，批量力控制器/ActionTerm，批量覆盖/可见性，
  TASK-009D0 MDP terms、环境配置、环境生命周期与加法注册，以及网格平移哈希纯函数。
- 测试：parameterized-force、coverage、TASK-009D0 配置/位姿/运行时/环境/等价/隔离协议测试。
- 文档：本说明、执行报告、active 索引和 `PROJECT_RUN_LOG.md`。

未创建吞吐脚本、吞吐冻结配置或长时 soak 脚本，因为这些文件属于 Gate 5/6，Gate 3
失败后继续实现会越过合同停止边界。

## 自动测试

- Gate 1 完整纯测试：168/168 通过，49 条上游弃用警告。
- Gate 3 定向回归：27/27 通过，49 条上游弃用警告。
- 单环境等价协议与相关回归：70/70 通过。
- 首次 Gate 1 合同集曾记录 151/151 通过；后续新增测试后最终完整计数为 168。

合同测试命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force tests/coverage tests/stomach_coverage tests/runtime
```

## Gate 1：配置、设备和纯接口

状态：`pass`。

先决条件检查确认外部位姿库、Git 清单、不可达配置和覆盖清单哈希匹配；RGB 为
`[1,720,1280,3]` 且有限；设备全部为 `cuda:0`。新任务策略组只含 RGB 和 7 维上一动作，
特权胶囊/覆盖状态位于独立组。

## Gate 2：单环境精确等价

状态：`pass`。

五个冻结 validation 位姿各运行 60 个边界，共 300 个边界。新批量实现与旧标量公式的当前
可见及累计布尔掩码逐顶点完全一致；最大面积误差
`3.469446951953614e-18 m²`。未通过浮点容差掩盖布尔差异。

## Gate 3：双环境隔离

状态：`fail`。

两个环境均加载 `validation-0006`，使用相同种子和相同 reset 流程。在合同要求的 10 个
HOLD 边界（240 个物理子步）后：

- 局部位置误差：`2.4250899514299817e-05 m`
- 合同最大位置误差：`1e-6 m`
- 四元数绝对内积：`0.9999984502792358`
- 合同最小绝对内积：`0.999999`
- 两行初始可达面积覆盖率：`0.026983204096286028`、`0.02698320409628603`
- 两行 RGB 帧号在全部十个 HOLD 周期中同步递增。

失败在 Phase A 初始一致性检查已经成立，因此没有执行分叉 Phase B 或逐行清空验收。曾诊断
Enhanced Determinism，并将两个重复位姿规范为完全相同的边界起点；经过一个 0.1 秒主动
边界仍出现约 60 μm 局部差异和 6 个顶点的当前可见差异。无效实验改动已撤回。

该结果说明当前 RTX 5090/GPU PhysX/4 m 克隆布局下，接触稳定过程未达到合同规定的
1 μm 跨环境确定性。不能将其解释为环境状态串扰已经通过。

## Gate 4--6

- Gate 4：`pass_after_manual_waiver`。20/20 次双环境同步 reset 通过；每次均执行十个连续
  HOLD RGB 边界，两个 train 位姿互异，RGB 有限、C0 为正、正式 episode length/上一动作/
  Actor 力清零，环境、PhysX、相机和覆盖设备均为 `cuda:0`。首次运行仅因 Isaac Lab 3.0
  底层 `RigidBodyView.device` 私有接口不存在而中止；改用公开 SimulationContext 设备接口后
  相同配置一次通过，未重试物理样本。
- Gate 5：`pending_after_manual_waiver`。
- Gate 6：`pending_after_manual_waiver`。

本报告将在后续门禁完成后补入直接观测结果；在此之前仍不授权 TASK-009D-1。

## 关键命令

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/inspect_task009d0_prerequisites.py \
  --device cuda:0 --output_directory \
  /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/prerequisites

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_single_env_parity.py \
  --headless --device cuda:0 --output_directory \
  /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate2_single_parity

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_two_env_isolation.py \
  --headless --device cuda:0 --output_directory \
  /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate3_isolation

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_gpu_reset_sync.py \
  --headless --device cuda:0 --resets 20 --output_directory \
  /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate4_reset_sync
```

Isaac Lab 3.0 当前 AppLauncher 不公开 `--headless`，验证脚本只做兼容转换并将 visualizer 设为
空列表；没有改变仿真、相机或覆盖语义。

## 外部工件

| 工件 | 绝对路径 | 字节 | SHA-256 |
|---|---|---:|---|
| 先决条件 | `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/prerequisites/task009d0_prerequisites.json` | 2309 | `e2d2fb5f6dbaea0f6c596e49c798def28da63fabaa797e9a3628c6d09a5415fa` |
| Gate 2 | `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate2_single_parity/task009d0_gate2_single_env_parity.json` | 306475 | `6dd58a33a300bcea2cec4bd7b3ca9c4f9452fee453109aaf99722a069cd514b9` |
| Gate 3 | `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate3_isolation/task009d0_gate3_two_env_isolation.json` | 2540 | `ceb3518cd23a65275f8963417110687783e823789e6062b8dc6d10d2b8f6911e` |
| Gate 4 | `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate4_reset_sync/task009d0_gate4_gpu_reset_sync.json` | 19306 | `3862a8d72f9859ae235e2ff3e5f4cc9ab5197f1fe9222467c8163b273568591b` |
| 外部位姿库 | `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009b_pose_library/20260826_040250_292641Z/pose_library_v1.jsonl` | 1935122 | `7a7a20e175dcfade0c3f07ccc2a4dca377508485f726fc6999a0453cb7cea855` |

外部工件未加入 Git。

## 偏离与后续决策

未放宽任何合同门槛。用户于 2026-08-28 明确接受当前环境差异并要求继续后续工作；Linux
端将其记录为人工豁免，而不是新合同或 Gate 3 通过。4 m 间距、资产和 1 μm 门槛保持不变。
