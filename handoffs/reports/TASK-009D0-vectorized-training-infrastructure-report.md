# TASK-009D0 向量化训练基础设施执行报告

## 最终状态

`accepted_with_manual_waiver`。Gate 1、2 通过；Gate 3 保持
`fail_with_manual_waiver`，其阈值和原始失败数据未改写。用户于 2026-08-28 明确接受该微小
跨环境差异；Gate 4、原 Gate 5、Gate 6 均在原合同参数下通过。TASK-009D0A 又完成12环境
增量决策、过期测试修正和全仓零失败回归，因此按最新合同完成D0收尾，并授权后续
TASK-009D-1使用最终冻结的12环境配置。12环境中已确认的少见非正C0 reset按用户最新决定
直接中止该次运行，不重采样、不修改位姿库，也不将其解释为持续运行故障。

## 版本边界

- Windows 规划分支：`workflow/TASK-009D0-vectorized-training-infrastructure`
- 原规划 HEAD：`35273aa3ae5a2a33f3a443028f720f0c4308d608`
- TASK-009D0A 规划 HEAD：`a082dcf6002e1a67ddb0d73f1af4abb54110ae8f`
- 设计提交：`f8eb6b825aa8e5765b3db52532b169a9d299066e`
- 精确代码基线：`7c4c5a18780b980ad3882ce75f1d64733fc3080d`
- Linux 分支：`feature/TASK-009D0-vectorized-training-infrastructure`
- Gate 5 工件实现 HEAD：`0e5a5f2fedfa2b48ae294a58b5cc52fb16cb6f32`
- Gate 6 工件实现 HEAD：`4464adaf499a85e35512dc9ce03cf298b34af013`
- TASK-009D0A 12环境工件实现 HEAD：`676e4e528c18d4f52dd067a0d88afa592821c1ed`

基线和设计提交均已验证为当前分支祖先。文档提交会产生新的最终 HEAD，推送后由终端交付
信息给出完整本地/远端哈希。

## 环境

- Isaac Lab Git：`d23a06799c946a4581a6245ea3105dbfb387cb20`
- Isaac Lab 包：12.0.0；Kit：110.1.2；Python：3.12.13
- PyTorch：2.11.0+cu128；Warp：1.15.0.dev20260626
- GPU：RTX 5090 32607 MiB；驱动：595.84
- 环境、PhysX、相机和覆盖：`cuda:0`

GPU dynamics 禁用 CCD 的上游警告保留在日志中；没有切换 CPU 或修改 USD 资产。

## 实现范围

新增独立向量任务、严格候选/冻结配置、split 安全位姿批次、批量参数化力、批量 GPU 第一
命中可见性、逐环境面积覆盖、10 Hz RGB 同步、Actor/特权观测隔离、同步 reset/1200 步
生命周期、吞吐选择和双回合长测。旧 TASK-009B/C 入口未重定向。

多环境胃网格修复仅规范化克隆坐标：相对 env_0 变换的平移分量最多枚举 ±1 ULP，仍必须
精确命中冻结几何 SHA-256。没有实现 CNN、GRU、Actor、Critic、PPO、VLM、奖励缩放、
课程学习或训练完成的模型；这些均为后续工作。

## 自动测试

- Gate 6 前定向协议：17/17 通过。
- 最终合同测试：181/181 通过，49 条上游弃用警告。
- TASK-009D0A 按合同修正两个过期测试中的旧力度期望，未改控制器实现或范围。
- 最终全仓测试：354/354 通过，0 失败，73 条上游警告。

最终命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q tests
```

## Gate 结果

- Gate 1 `pass`：输入哈希、GPU 设备、相机、批量接口和纯测试通过。
- Gate 2 `pass`：5 个 validation 位姿、300 边界与旧标量可见/累计布尔掩码逐顶点相同；
  最大面积误差 `3.469446951953614e-18 m²`。
- Gate 3 `fail`：局部位置差 `2.4250899514299817e-05 m`，四元数绝对内积
  `0.9999984502792358`，未满足 `1e-6 m` / `0.999999`。人工豁免不改判。
- Gate 4 `pass_after_manual_waiver`：20/20 双环境 GPU reset；每次十个连续 RGB/HOLD，
  C0 正、计数/上一动作/残余力清零，设备均为 `cuda:0`。
- Gate 5 `pass_after_manual_waiver`：12/12 独立进程通过、零故障、显存均大于 20%；冻结 8
  环境。
- Gate 6 `pass_after_manual_waiver`：2×8 个完整 120 秒回合通过；每环境 1201 点、28800
  正式子步、240 HOLD，状态/RGB 有限、覆盖单调、帧连续、reset 不继承掩码。
- TASK-009D0A `complete_with_user_waiver`：12环境短冒烟通过；三次独立正式进程中repeat 0/2
  通过，repeat 1因reset产生非正初始C0而严格中止。诊断确认这是`train-0419`只看到人工
  冻结不可达区导致的可达C0为0，不是GPU/帧同步/PhysX故障。用户接受该少见中止并冻结
  12环境；全仓354/354通过。12环境未追加两个120秒长回合，原Gate 6证据仍是8环境；这是
  用户在已知证据边界下作出的收尾决定，不应表述为12环境长时验证已完成。

### Gate 5 原始吞吐

| 环境数 | repeat 0 | repeat 1 | repeat 2 | 中位数 | 最低剩余显存 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.448053 | 5.509787 | 5.540883 | 5.509787 | 0.788754 |
| 2 | 10.114248 | 10.233455 | 10.231598 | 10.231598 | 0.772091 |
| 4 | 17.741569 | 18.116163 | 18.112459 | 18.112459 | 0.740890 |
| 8 | 30.310967 | 30.322024 | 30.198913 | 30.310967 | 0.676818 |

### Gate 6 摘要

- 回合 0：8 个 train 位姿，C0 为 0.0108%–12.08%，最终 60.82%–99.62%。
- 回合 1：8 个新 train 位姿，C0 为 0.4357%–11.96%，最终 88.22%–99.78%。
- 上述为确定性 R3 随机动作长测，不是学习策略性能或论文正式统计。

### TASK-009D0A 十二环境增量结果

| 进程 | 状态 | 吞吐（env-steps/s） | 最低剩余显存 | 说明 |
|---:|---|---:|---:|---|
| 冒烟 | pass | 42.438179 | 0.632678 | 2步预热、5步测量 |
| repeat 0 | pass | 44.625125 | 0.632674 | 50步预热、300步测量 |
| repeat 1 | fail | — | — | reset后非正初始C0，未进入测量 |
| repeat 2 | pass | 44.549097 | 0.632674 | 50步预热、300步测量 |

成功进程吞吐中位数为`44.587111`，比既有8环境中位数`30.310967`高47.10%，且显存满足
20%门槛。原合同的零故障规则最初会拒绝12环境；随后用户明确接受已定位的少见非正C0
reset中止，规定遇到时直接终止该次运行且不做额外处理。因此最终
`configs/task009d0/vectorized_training_frozen_v2.json`选择`selected_num_envs=12`，没有覆盖v1。
这一人工豁免只接受完全匹配的、测量开始前的`TASK-009D0 reset produced non-positive initial
C0`中止，其他故障不在豁免范围。

## 关键执行命令

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/inspect_task009d0_prerequisites.py \
  --device cuda:0 --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/prerequisites

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_single_env_parity.py \
  --device cuda:0 --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate2_single_parity

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_two_env_isolation.py \
  --device cuda:0 --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate3_isolation

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_gpu_reset_sync.py \
  --device cuda:0 --resets 20 --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate4_reset_sync

# 下条命令以 num_envs=1,2,4,8 与 repeat_index=0,1,2 分别执行，共12个独立进程：
./run_isaaclab.sh -p scripts/stomach_coverage/benchmark_task009d0_throughput.py \
  --num_envs <1|2|4|8> --repeat_index <0|1|2> --device cuda:0 \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_throughput

./run_isaaclab.sh -p scripts/stomach_coverage/summarize_task009d0_throughput.py \
  --artifact_root /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_throughput \
  --write_frozen_config configs/task009d0/vectorized_training_frozen_v1.json

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_long_soak.py \
  --device cuda:0 --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate6_long_soak

# TASK-009D0A：先加 --smoke 执行一次短冒烟，再去掉 --smoke 并分别使用 repeat 0/1/2。
./run_isaaclab.sh -p scripts/stomach_coverage/benchmark_task009d0_throughput.py \
  --num_envs 12 --repeat_index <0|1|2> --device cuda:0 \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_12env_incremental

./run_isaaclab.sh -p scripts/stomach_coverage/summarize_task009d0a_12env.py \
  --artifact_root /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_12env_incremental \
  --candidate_config configs/task009d0/vectorized_training_candidate_12env_v2.json \
  --base_frozen_config configs/task009d0/vectorized_training_frozen_v1.json \
  --write_frozen_config configs/task009d0/vectorized_training_frozen_v2.json \
  --accept_rare_nonpositive_c0_abort
```

AppLauncher 3.0 不公开 `--headless` 时，脚本移除该兼容参数并使用空 visualizer；仿真、相机与
覆盖语义不变。

## 外部证据

| 工件绝对路径 | 字节 | SHA-256 |
|---|---:|---|
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/prerequisites/task009d0_prerequisites.json` | 2309 | `e2d2fb5f6dbaea0f6c596e49c798def28da63fabaa797e9a3628c6d09a5415fa` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate2_single_parity/task009d0_gate2_single_env_parity.json` | 306475 | `6dd58a33a300bcea2cec4bd7b3ca9c4f9452fee453109aaf99722a069cd514b9` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate3_isolation/task009d0_gate3_two_env_isolation.json` | 2540 | `ceb3518cd23a65275f8963417110687783e823789e6062b8dc6d10d2b8f6911e` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate4_reset_sync/task009d0_gate4_gpu_reset_sync.json` | 19306 | `3862a8d72f9859ae235e2ff3e5f4cc9ab5197f1fe9222467c8163b273568591b` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_throughput/task009d0_throughput_summary.json` | 4801 | `9133d5dc305f810a81fd3a827006ebfc28224e4b2d92288679192f05851803a6` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate6_long_soak/task009d0_gate6_long_soak.json` | 1578983 | `3bbf68398ec3c58ff805b4d285b35250e6985f2047f5b11487f2aa10a911f023` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate6_long_soak/artifact_inventory.json` | 283 | `7f90e130241ad8ecdc4efdf1f22debac61a277f0b7fdcd34c204f396c5aa5844` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/artifact_inventory.json` | 5409 | `36b5584ea2a0aa1736680a3c1933cd39be713132ec8afdd6367c4b690528bf50` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_12env_incremental/task009d0a_smoke_env12.json` | 4952 | `f22a7961bb7c153a2bd2eecacf740bf73dbae7ae61889f589a011cff49b68531` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_12env_incremental/task009d0_throughput_env12_repeat0.json` | 250762 | `00e716ba31d28aa41da59a30ffc4af5bd4bf1035915dacd87100ce405487e088` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_12env_incremental/task009d0_throughput_env12_repeat1.json` | 730 | `7a3485c3cdc9618841d166303e583b29f0b20cc05c45bd78b6cd68307862d2a4` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_12env_incremental/task009d0_throughput_env12_repeat2.json` | 251184 | `145a12747744aeb1485d9d907e4cc2cb86a4bbb4ff7176fc651d7b67fae83382` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_12env_incremental/task009d0a_12env_throughput_summary.json` | 1967 | `5f5f772bdf0cbca87a71227e684a7c467c05e597fa8448200c99f335d6e78e70` |

Git 内冻结配置为
`configs/task009d0/vectorized_training_frozen_v1.json`，4419 字节，SHA-256
`e94993f410b2a48bcdc8d5a40ef5504b8541ff6fe68091abef540639b8eab3b1`。

TASK-009D0A 新增且不覆盖旧文件：候选配置
`configs/task009d0/vectorized_training_candidate_12env_v2.json`，1106字节，SHA-256
`2b2e12182b9b2fe0d6605cffcdb7700c305de2a792aa64d3583034eb22b3d121`；冻结配置
`configs/task009d0/vectorized_training_frozen_v2.json`，4021字节，SHA-256
`6add9b6656c400b37743269d97d5d1ab5cde61fac1cc4cac1c829043c1556f75`。

### Gate 5 十二份源清单

以下路径前缀均为绝对目录
`/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate5_throughput/`。

| 文件名 | 字节 | SHA-256 |
|---|---:|---|
| `task009d0_throughput_env1_repeat0.json` | 162807 | `6fc23c8c60996662c254dd79ff68fcfa5cc8165a9fa68b989e180f850aec1e2c` |
| `task009d0_throughput_env1_repeat1.json` | 162831 | `d3af6edd2c8e0964af8792ea6ab8c1d3895c06bc695fe2a19e7504ff05825bc1` |
| `task009d0_throughput_env1_repeat2.json` | 162808 | `e7c25055a0b2bc6759c38d289f356ba382b4ef28030d1fab974b2eb119a56c17` |
| `task009d0_throughput_env2_repeat0.json` | 170726 | `dbb39f1d120dfe761968c9f17991d7281207224b5d14da9ed8ad8fbbb65de14a` |
| `task009d0_throughput_env2_repeat1.json` | 170794 | `a694c9050eaad07927c3903cc02131a8737611f6c029c2420a9bfdb3bc6dd699` |
| `task009d0_throughput_env2_repeat2.json` | 170769 | `73c7a164e658a5aa46d48e6be9af091358f48173cee0e1852c4e8af5f7a96616` |
| `task009d0_throughput_env4_repeat0.json` | 186575 | `3755e4221b966a174bae103a30c78448cadde394d3ba8a230622d53dbeccda8a` |
| `task009d0_throughput_env4_repeat1.json` | 186761 | `cf64bd94a8a892aeda131e70740872bfa6d6aa56c48805e57bd71bf5bff99d35` |
| `task009d0_throughput_env4_repeat2.json` | 186784 | `52749d7074482f902b368df128681c96c12bd6a77617a489d9a1aac5894e2436` |
| `task009d0_throughput_env8_repeat0.json` | 218773 | `dddafdb15eb8186c122798a654df539314c7548a06648321b3c9ecfe8f9e2a74` |
| `task009d0_throughput_env8_repeat1.json` | 219017 | `ea32c381ef2d8be8d0efb3e9f344b9eb63fff21585611e086b340a68627436ef` |
| `task009d0_throughput_env8_repeat2.json` | 219075 | `7660d432857dc452b56f5ce19a47f4f4cc8d6e3f4f98dfa1456669b0c28957c9` |

外部工件未加入 Git。

## 偏离与未验证项

研究合同存在两项明确的用户豁免：其一是接受 Gate 3 的跨环境微小差异；其二是接受已定位
的少见非正C0 reset中止并冻结12环境。Gate 3结果保持
`fail_with_manual_waiver`，总状态依据TASK-009D0A合同收尾为`accepted_with_manual_waiver`。
12环境的证据覆盖短冒烟和两次完整Gate 5测量，不包含双120秒长测；后续若论文或训练准入
要求12环境长时稳定性，应另行补验。
没有放宽阈值、降低分辨率、
改变相机/70 mm/ROI/力度/位姿库、删除困难样本或修改 USD。未验证或未实现的内容包括 PPO
训练、网络吞吐、策略性能、VLM 和 TASK-009D-1 模型；本报告不作这些声明。
