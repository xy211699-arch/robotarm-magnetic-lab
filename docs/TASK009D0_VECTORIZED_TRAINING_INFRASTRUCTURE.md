# TASK-009D0 向量化训练基础设施

## 状态与边界

TASK-009D0 新增独立任务
`Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0`，不替换已验收的
TASK-009B/C 单环境入口。Gate 1、2 通过，Gate 3 未达到冻结的跨环境 1 μm 一致性门槛；
用户于 2026-08-28 明确接受该差异并授权继续。Gate 4、5、6 已在不改变门槛、资产、相机、
覆盖或力度的条件下通过，因此工程链路可用于下一阶段，但合同总状态仍为 `partial`，Gate 3
不能改写为通过。

本任务没有实现 CNN、GRU、Actor、Critic、PPO、VLM 或训练完成的策略；这些仍属于后续
TASK-009D-1 范围。

## 冻结接口

- 物理 240 Hz，控制、策略 RGB 和覆盖更新 10 Hz，每动作 24 个物理子步。
- 正式回合 120 秒、1200 个动作边界；reset 后先运行 10 个不计预算的 HOLD 边界。
- 相机 1280×720、120°圆形视场；覆盖使用 70 mm、GPU 第一命中、冻结面积权重和冻结不可达
  区域。
- Actor 只接收 `policy.rgb` 与 7 维上一实际动作；位姿、速度、接触和覆盖只在
  `privileged` 组。
- MOVE 0.70–1.40 mg，VIEW 0.20–0.50 mg，UP 0.80–1.05 mg，HOLD 为零 Actor 力。
- Gate 5 冻结并行数为 8，配置为
  `configs/task009d0/vectorized_training_frozen_v1.json`。

## 实现模块

- `runtime/task009d0_config.py`：候选/冻结配置及输入哈希校验。
- `runtime/task009d0_pose_batch.py`：train/validation/test 隔离的确定性位姿批次。
- `controllers/vectorized_parameterized_force.py` 与
  `mdp/vectorized_parameterized_force_action.py`：批量模式、力度、方向和逐子步持续施力。
- `coverage/batched_accumulator.py`：逐环境当前/累计覆盖掩码和逐行 reset。
- `coverage/batched_visibility.py`：批量候选与 Warp CUDA 第一命中。
- `runtime/task009d0_coverage_runtime.py`：10 Hz 帧同步、逐环境覆盖和克隆无关几何。
- `task009d0_vector_env.py`：批量位姿写入、十次 HOLD、正 C0、1200 步同步生命周期和终态
  审计快照。
- Gate 2–6 验证脚本位于 `scripts/stomach_coverage/validate_task009d0_*.py`，吞吐脚本为
  `benchmark_task009d0_throughput.py` 与 `summarize_task009d0_throughput.py`。

多环境下 USD 世界平移会在约 1 ULP 处改变严格几何哈希。实现先计算 Surface 相对 env_0 的
变换，只枚举平移分量 ±1 ULP，并且只有命中原冻结 SHA-256 才接受；没有采用近似哈希或
修改胃网格。

## 吞吐与长时结果

| 环境数 | 三次吞吐（env-transitions/s） | 中位数 | 最低剩余显存 |
|---:|---|---:|---:|
| 1 | 5.4481 / 5.5098 / 5.5409 | 5.5098 | 78.88% |
| 2 | 10.1142 / 10.2335 / 10.2316 | 10.2316 | 77.21% |
| 4 | 17.7416 / 18.1162 / 18.1125 | 18.1125 | 74.09% |
| 8 | 30.3110 / 30.3220 / 30.1989 | 30.3110 | 67.68% |

8 环境明显最快且不属于 10% 近似并列，故冻结为 8。Gate 6 随后完成两个 8 环境完整回合：
每环境每回合均为 1201 个覆盖点、28800 正式物理子步和 240 个 reset HOLD 子步；帧连续、
覆盖单调、状态/RGB 有限，第二回合未继承第一回合掩码。最终覆盖率范围分别为
60.82%–99.62% 和 88.22%–99.78%；这是 R3 随机动作长测结果，不是学习策略性能。

## 常用复核命令

```bash
cd /tmp/robotarm-task009d0

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force tests/coverage tests/stomach_coverage tests/runtime

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_gpu_reset_sync.py \
  --device cuda:0 --resets 20 \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate4_reset_sync

./run_isaaclab.sh -p scripts/stomach_coverage/validate_task009d0_long_soak.py \
  --device cuda:0 \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate6_long_soak
```

Gate 5 必须为 1、2、4、8 环境分别运行 3 个独立 Isaac Sim 进程；不得把一条进程内循环
当成独立证据。正式工件根目录为
`/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/`。

## 已知偏离

Gate 3 原始失败保持有效：两个相同环境在 1 秒 HOLD 后局部位置相差
`2.4250899514299817e-05 m`，四元数绝对内积为 `0.9999984502792358`，未满足
`1e-6 m` / `0.999999`。Gate 4–6 是用户人工豁免后继续执行的工程证据，不等价于修改研究
合同或证明 GPU PhysX 跨克隆达到 1 μm 确定性。
