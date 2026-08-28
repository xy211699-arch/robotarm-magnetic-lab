# TASK-009D0 向量化训练基础设施

## 当前状态

当前实现状态为 `partial`。独立任务
`Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0` 已完成配置、批量动作、批量
GPU 可见性、逐环境覆盖状态、10 Hz RGB 同步、Actor/特权观测隔离和同步回合生命周期。
Gate 1 与 Gate 2 已通过；Gate 3 在严格双环境物理一致性门槛处失败。用户于 2026-08-28
明确接受该环境差异并授权继续 Gate 4--6；该授权仅是人工豁免，不改变 Gate 3 失败结论。

本任务是新增任务，不替换 TASK-009B/C 单环境入口，也没有实现 PPO、Actor/Critic、CNN、
GRU 或 VLM。

## 固定接口

- 物理频率：240 Hz。
- 动作、策略 RGB 与覆盖更新频率：10 Hz，每个边界 24 个物理子步。
- 正式回合：120 秒、1200 个动作边界；reset 后先执行 10 个不计入回合的 HOLD 边界。
- 相机：1280×720、120 度圆形视场；覆盖使用 70 mm 第一命中和冻结面积权重。
- 动作：六模式加连续力度。MOVE 为 0.70--1.40 mg，VIEW 为 0.20--0.50 mg，UP 为
  0.80--1.05 mg。
- Actor 观测：`policy.rgb` 与 7 维上一动作（六维 one-hot 加力度）。
- 特权观测：胶囊状态与覆盖状态，保留在独立 `privileged` 组，不进入 Actor 路径。

## 已实现模块

- `runtime/task009d0_config.py`：严格配置、文件哈希和外部位姿库校验。
- `runtime/task009d0_pose_batch.py`：train/validation/test 分组安全的可复现位姿批次。
- `controllers/vectorized_parameterized_force.py`：批量模式、力度、方向和端点力数学。
- `mdp/vectorized_parameterized_force_action.py`：在每个 240 Hz 子步重算方向并持续施力。
- `coverage/batched_accumulator.py`：逐环境可见/累计掩码、面积和逐行清空。
- `coverage/batched_visibility.py`：GPU 候选、第一命中、关联面、法向和距离门控。
- `runtime/task009d0_coverage_runtime.py`：环境局部几何、RGB 帧和覆盖边界同步。
- `task009d0_vector_env.py`：位姿写入、十次 HOLD、正 C0、同步 1200 步生命周期。

向量环境会把冻结位姿库中的单环境世界坐标解释为环境局部坐标，再加上每行完整的
`env_origin`。胃网格在校验冻结不可达区域前也会移除 env_0 克隆平移并重算几何哈希，避免
批次数量改变导致同一网格被误判为不同资产。

## 已通过验收

### Gate 1

先决条件检查确认 PhysX、相机张量和 Warp 射线均位于 `cuda:0`；1280×720 RGB 有限。
完整纯测试为 168/168 通过。旧 TASK-009B/C 共享模块回归保持通过。

### Gate 2

五个冻结 validation 位姿、每个位姿 60 个边界，共 300 个边界。新批量实现与旧标量数学的
当前可见及累计布尔掩码逐顶点完全相同，最大面积误差为
`3.469446951953614e-18 m²`。

## Gate 3 阻塞

两个环境使用同一个 `validation-0006` 位姿和相同动作。在 4 m 冻结环境间距、GPU PhysX
和合同要求的 1 秒 HOLD 稳定后，两个局部质心位置相差
`2.4250899514299817e-05 m`，四元数绝对内积为 `0.9999984502792358`。合同门槛分别为
不超过 `1e-6 m` 和不低于 `0.999999`，因此 Gate 3 失败。

诊断中曾在不改变资产、相机、覆盖和力度的前提下尝试 PhysX Enhanced Determinism，并将
重复位姿在 HOLD 后规范到完全相同的局部状态；后者使边界开始时位置误差为零，但经过一个
0.1 秒动作边界后仍产生约 60 μm 的局部差异及 6 个顶点的可见差异。两项实验性改动均已
撤回，未进入交付实现。

没有放宽 1 μm 门槛、改变 4 m 间距、接受近似掩码或改动资产。后续 Gate 4--6 结果会明确
标记为“人工豁免后继续执行”；只有这些门禁完成后才能判断该基础设施的工程可用性。

## 复核命令

```bash
cd /tmp/robotarm-task009d0

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force tests/coverage tests/stomach_coverage tests/runtime

./run_isaaclab.sh -p \
  scripts/stomach_coverage/validate_task009d0_single_env_parity.py \
  --headless --device cuda:0 \
  --output_directory \
  /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate2_single_parity

./run_isaaclab.sh -p \
  scripts/stomach_coverage/validate_task009d0_two_env_isolation.py \
  --headless --device cuda:0 \
  --output_directory \
  /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009d0_vectorized_training/gate3_isolation
```

Gate 3 输出中的 `status=fail` 是当前已确认结果，不应把进程退出码代替清单状态进行判断。
