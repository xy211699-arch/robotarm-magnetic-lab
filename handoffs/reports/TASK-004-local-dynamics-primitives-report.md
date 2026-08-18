# TASK-004 Linux 执行报告

## 1. 结论

**Disposition：`partial`**。

已完成纯 NumPy 轨迹与闭环扳手控制器、四浮点脉冲协议、Isaac Lab 动态胶囊 COM
世界系力/力矩适配、隔离平面任务、实时前置检查和定量验证器。实时接口与隔离门禁通过，
但四个动作的平面定量门禁未通过；依合同没有实例化胃部任务、没有进行胃部渲染，也没有
创建会暗示可用性的交互启动器。

主要阻塞事实：允许参数范围内，`SIDE_TO_UPRIGHT` 在 8.0042 s 超时，胶囊最终仍约
88.94°倾斜；其余三个动作因前置姿态未建立而合法返回 `invalid_start`。没有非有限状态、
扳手越界、根状态写入或隐藏位姿修复。

## 2. 版本与范围

- 规划基线：`06b15caf9a69bc9c20f85522ce4abbb32c8b9245`
- 规划头：`e1dd0a74faa5d639b0ec49dab6da83ce99d947f2`
- 实施分支：`feature/TASK-004-local-dynamics-primitives`
- 代码实施头（报告提交前）：`4fccaf7da5e1ff1b1a04d48348915f2b82edf714`
- 未合并；没有修改 USD/USDZ、资产、质量、惯量、相机、物理材料、摩擦、恢复系数、
  求解器、胃部姿态、TASK-003 重置或既有实验结果。

实施提交：

1. `fff3bfc` — `feat: define local capsule primitive trajectories`
2. `bbd0ebd` — `feat: add closed-loop primitive wrench controller`
3. `12a5c15` — `feat: apply local primitive force and torque`
4. `ff1885b` — `test: validate local primitives on flat contact`
5. `eb03ef5` — `tune: record bounded primitive calibration`
6. `4fccaf7` — `fix: copy immutable wrench commands into tensors`

## 3. 实时前置门禁

实时平面预检为 PASS：

- 胶囊质量：`0.005734999664 kg`
- 主惯量：`[3.1337407e-7, 3.9471431e-7, 1.9233103e-7] kg·m²`
- 非运动学刚体；重力开启；刚体 CCD 与场景 CCD 开启
- 物理/环境/渲染/相机：`240/60/60/30 Hz`
- 相机局部偏置：`[0, 0, -0.0127] m`
- 定向局部轴：`[0, 0, -1]`
- COM 直接力与力矩 API 可调用，`positions=None`、`is_global=True`
- 只读接触回调提供 position、normal、impulse、separation
- 平面动作项仅为 `local_primitive`
- 运行源扫描未发现禁止的状态写入

实现中发现并修正了一个坐标错误：`root_com_pose_w` 的姿态属于 PhysX COM/主惯性坐标系，
与胶囊 link 几何轴约差 17°。最终实现保持 COM 位置和速度反馈，但从
`root_link_pose_w` 计算胶囊局部 `-Z` 轴。修正后正常侧躺重置正确通过 75°–105°门限。

## 4. 标定过程与最终参数

外部 `TASK004_calibration_attempts.jsonl` 含 7 条完整尝试：

| 尝试 | 单独变化的类别 | 结果 |
|---:|---|---|
| 0 | 设计初值 | 起身超时 |
| 1 | 轴向 Kp/Kd 到 `3e-5/8e-6` | 起身超时 |
| 2 | 力矩限幅到 `3e-5 N·m` | 起身超时 |
| 3 | XY Kp/Kd 到 `3.0/0.15` | 起身超时 |
| 4 | 水平力限幅到 `1.0 mg` | 起身超时；允许组中姿态变化最大 |
| 5 | 向下预载降至 `0` | 起身超时，无改善 |
| 6 | 时长改为 `(1.5,1.5,1.5,6.0) s` | 起身超时，无改善 |

最终工作树恢复并保留尝试 4 的最佳合法配置：轴向 Kp/Kd `3e-5/8e-6`、滚转阻尼
`1e-6`、力矩限幅 `3e-5 N·m`、XY Kp/Kd `3.0/0.15`、水平限幅 `1.0 mg`、向下
预载 `0.15 mg`，运动时长 `(5.5,4.5,3.5,8.0) s`，硬超时
`(8.0,7.0,6.0,9.5) s`。

## 5. 平面定量结果

最终保留参数对应汇总：

| 动作 | 状态 | 完成时间 | 最终目标误差 | 最大线/角速度 | 最大力/力矩 |
|---|---|---:|---:|---:|---:|
| 侧躺→直立 | `timed_out` | 无；8.0042 s超时 | 88.94° | 0.00675 m/s / 1.0053 rad/s | 0.03287 N / 3.0e-5 N·m |
| 直立→侧躺 | `invalid_start` | 无 | 前置直立未建立 | 不适用 | 0 / 0 |
| 直立→30° | `invalid_start` | 无 | 前置直立未建立 | 不适用 | 0 / 0 |
| 30°锥转一周 | `invalid_start` | 无 | 前置30°未建立 | 不适用 | 0 / 0 |

起身过程中力矩约 23.08% 的 60 Hz 观测样本处于限幅，力和力矩均未超过边界；
无非有限样本。起身没有进入末期非相机端支撑，因此该接触验收项为失败而不是伪造通过。
实际锥面覆盖为 0。所有依赖动作保持拒绝，未直接重置成直立或 30°姿态。

动力学解释：最大允许反馈力矩远小于把 5.735 g、半长 12.5 mm 胶囊抬升重心所需的
恢复力矩量级；XY 锚点力主要造成约毫米级平移，没有形成足够的非相机端枢轴。继续通过
改物理材料、质量或位姿写入获得 PASS 均违反合同。

## 6. 测试与回归

- TASK-004 专项：`28 passed`
- TASK-003、理想表面、覆盖与动作层纯回归：`137 passed`
- 覆盖几何 GPU 回归：`COVERAGE_GEOMETRY_PASS`
- TASK-003 动态力实时预检：`DYNAMIC_FORCE_PREFLIGHT_PASS`
- 既有原子胃部动作实时回归：11/11 `DONE`，`P0_VALIDATION ... status=PASS`
- Python `compileall`：PASS
- `git diff --check`：PASS
- 禁止项扫描：控制器/动作适配中无根状态写入或几何自适应；预检脚本只包含用于扫描的
  禁止字符串表。

## 7. 未执行项和偏差

- 平面门禁失败后依合同停止，未创建
  `Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0`。
- 未执行胃部连续渲染、胶囊相机主观检查、胃部碰撞观察或共享配置摘要比对。
- 没有任何“胃部可用性”“动作平滑性”人工验收结论。
- 设计文档列出轴向增益等显式范围，但没有单列力矩限幅范围；本次把力矩限幅保守限制在
  轴向 Kp 上界 `3e-5 N·m`，未自行扩大权限边界。

## 8. 外部证据

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `/tmp/task004-preflight-host4/20260818T040942Z_local_primitive_preflight.json` | 2268 | `cca13853cf7f34589e82f678a4c95027bad69dc99aca66a066c94233befc5176` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/local_primitives_flat/20260818_042543_538281Z/summary.json` | 6922 | `4c5d495262ec8933b8b59d2dd5583186a8a0c717f321295b538b25638e99317e` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/local_primitives_flat/20260818_042543_538281Z/samples.jsonl` | 1124681 | `b43c232ef5cd2c2097d0979c5782cf8f08ebe60065d0426d67d301d3b4f69820` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/local_primitives_flat/TASK004_calibration_attempts.jsonl` | 26622 | `7b50588483ba03894ec21eb9cf49a030f30f9ed52e26405c30c0641bd4373b7e` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/logs/dynamic_force_preflight/20260818_043335_483829Z/prerequisites.json` | 4367 | `93ed0676b11ee2980a7c8a1696b0c4d928e8e6f4306b4b751427f61dd19f957e` |

## 9. 方案端需要决定

若要继续 TASK-004，Windows 方案端应明确扩大允许的力矩/前馈权限并给出数值边界，或重新
设计不要求从平面侧躺主动抬升的验收轨迹。获得新合同前，Linux 不应迁移胃部控制器。
