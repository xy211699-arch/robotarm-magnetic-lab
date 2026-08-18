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

---

# TASK-004 仿真优先修订续报（2026-08-18）

## 10. 修订依据与最终状态

方案端规划头 `bdfb657` 明确以“仿真动作可用性”为本阶段最高优先级，取代本报告第1至9节
采用的旧物理权限边界。Linux执行端据此继续在同一功能分支实施，但仍保留以下硬约束：胶囊
必须是非运动学动态刚体，运行期不得写入根位姿/速度，不得投影、传送、查询胃壁净空或加入
胃部专用恢复；控制器只能输出力和力矩，由PhysX积分状态。

最终状态为 **partial**：修订合同要求的平面四项定量门禁全部通过，同一冻结profile及digest
也已原样迁移到TASK-003胃部场景；但连续胃部渲染中“直立到侧躺”在7 s硬超时内未达到终态。
按合同不得针对胃部结果二次调参，因此不能标记complete。

实现提交按顺序为：

- `21d98ed`：新增共享仿真权限profile；
- `25d39dc`：新增非相机端点wrench控制；
- `fd153a2`：选择仿真动作权限；
- `377406c`：平面四原语门禁通过；
- `21c9b5e`：同profile无适配迁移到胃部任务。

## 11. 修订后的控制实现

控制器读取胶囊COM状态与刚体link姿态，将胶囊局部 `-Z` 定义为从非相机端指向相机端的
定向轴。每个原语开始时记录非相机端虚拟端点的世界XY锚点，计算端点PD力、恒定向下端点
预载以及姿态PD力矩，再把端点力转换成COM等效力和附加力矩。组合结果经过总力/总力矩限幅
和向量slew限制，由 `permanent_wrench_composer` 以世界系COM wrench施加。

本次同时修复了接触累计跨步残留，以及float32临界容差导致合法起始姿态被错误拒绝的问题。
所有机制均为仿真专用权限，不声明真实外磁体或硬件能够直接实现同等wrench。

## 12. 权限网格与四原语调参过程

隔离起身权限网格共24条记录：首轮20组全部失败；第二轮中端点预载0.01 N、力矩权限1倍在
5.9 s通过，0.01 N、2倍失败；0.02 N、1倍同样5.9 s通过，0.02 N、2倍失败。随后每轮都执行
完整四原语序列，未按单动作选择不同参数：

| 轮次 | 关键变化 | 结果 |
|---|---|---|
| A0 | 初始修订权限 | 失败 |
| A1 | 姿态增益提高至0.02 | 失败 |
| A2 | 端点锚定阻尼提高 | 失败 |
| A3 | 端点向下预载0.5 N | 失败 |
| A4 | 端点向下预载降为0.1 N | 起身、30°通过；侧躺和圆锥失败 |
| A5 | 锚定阻尼0.4，保留0.1 N预载 | 四项全部通过并冻结 |

## 13. 冻结profile与平面结果

冻结文件为 `configs/local_primitives/simulation_profile.json`，规范化JSON profile digest
（SHA-256；不含文件缩进差异）为：

```text
d82bf6d381e99d7be07cdf614223139fd8353c56011b8dc0a2d9779555bdcc72
```

关键值：轴向Kp/Kd `0.02/0.0016`、滚转阻尼 `0.0016`、姿态力矩限幅 `0.02 N·m`、
端点锚定Kp/Kd `10.0/0.4`、端点向下预载 `0.1 N`、总力/力矩限幅 `1.25 N/0.02 N·m`、
力/力矩slew `50.0 N/s / 0.2 N·m/s`。动作时长 `(5.5,4.5,3.5,8.0) s`，硬超时
`(8.0,7.0,6.0,9.5) s`。

| 原语 | 结果 | 完成时间 | 最大力 | 最大力矩 | 最大240 Hz单步位移 |
|---|---|---:|---:|---:|---:|
| 侧躺到直立 | PASS | 5.900 s | 0.10552 N | 0.0017149 N·m | 0.2448 mm |
| 直立到侧躺 | PASS | 5.1208 s | 0.12626 N | 0.0012136 N·m | 0.1226 mm |
| 直立到30° | PASS | 3.900 s | 0.10407 N | 0.0006743 N·m | 0.0191 mm |
| 30°圆锥一周 | PASS | 8.404 s | 0.11020 N | 0.0009932 N·m | 0.0808 mm |

圆锥展开角为 `6.2940047 rad`，倾角RMSE为 `0.0061398 rad`。所有状态有限，单步连续性
小于5 mm；起身相机半球承载样本为0，末期非相机端支撑成立。

## 14. 胃部无适配迁移与渲染结果

新增胃部任务只继承TASK-003的场景、重置、CCD、接触、相机和时序，并替换为同一动作配置
工厂；没有 `__post_init__` 覆盖，也没有胃部专用增益、恢复、几何查询或状态写入。平面和胃部
运行时均记录相同profile digest。

使用 `0,1;reset;0,2;reset;0,2,3` 连续序列、Kit外部视图和1280×720/30 Hz胶囊相机执行：

- 平面7/7终态成功；
- 胃部6/7终态成功：3次起身、2次30°和1次圆锥成功；
- 胃部“直立到侧躺”受局部胃壁接触阻塞，在7 s硬超时结束。

超时后未修改冻结profile，未增加场景条件分支，因此保留为可复现实验偏差。

## 15. 测试与回归

- TASK-004与动态力专项：`102 passed`；
- 理想表面、覆盖与动作层选择性回归：`87 passed`；
- TASK-003动态力实时预检：`DYNAMIC_FORCE_PREFLIGHT_PASS`；
- 覆盖几何GPU回归：PASS；
- 原子胃部动作实时回归：11/11 `DONE`，最终PASS；
- Python `compileall`：PASS；
- 禁止项扫描及 `git diff --check`：PASS。

## 16. 证据文件

| 证据 | 字节 | SHA-256 |
|---|---:|---|
| `configs/local_primitives/simulation_profile.json`（原始文件） | 544 | `d68f774b27c9d48ce0c34897275021c1b6fe1b804471e534939930db80401fd5` |
| `/tmp/task004-authority-isolated/20260818_054740_733205Z/attempts.jsonl` | 18360 | `8663e939ec87683d9879fb57e06ab179ec4a0ac5816b5c40670986ea35af322e` |
| `/tmp/task004-flat-all-kd04-pin01/20260818_061826_058673Z/summary.json` | 9091 | `531b58778c08fa5d0e4abf86a6b65208baab1891cc7a30fd81efa6bb787d6b3d` |
| `/tmp/task004-flat-all-kd04-pin01/20260818_061826_058673Z/samples.jsonl` | 2643050 | `c98721c9b3a8b1cf7b03d65d458371b91835cca4babdac10dc69206895459539` |
| `/tmp/task004-rendered-flat/20260818_062616_404936Z/session.json` | 1540 | `57f4ed6c1ed950d2deca19d8d5cbbe4f6f33fc220e41091815707589acd9ee2c` |
| `/tmp/task004-rendered-flat/20260818_062616_404936Z/samples.jsonl` | 3505872 | `e572d5a601fb0ba789580c7cd127cc0c0916d611ed9fcec425fe9f6ab7a4c158` |
| `/tmp/task004-rendered-stomach/20260818_062958_020939Z/session.json` | 1536 | `48f474a4d48d21096140777d1b94960e0844ff257f520371de9944a3e4ba3b22` |
| `/tmp/task004-rendered-stomach/20260818_062958_020939Z/samples.jsonl` | 4041694 | `59e9e478cdd97bb01b35bbb0268463b05c2dd4b9631f5f19360facb6a0b4ebe9` |
| 胃部超时快照PNG | 414437 | `22d11de27f1299d83eb769b4512c76da1f44963b7cf4120ff14b6285d665b146` |
| 胃部最终快照PNG | 619600 | `f662510ce7ff3b01e176c9ff37841e01e99e32943f8cc475940a39f9c19d4c90` |

调参A0至A4汇总哈希依次为：`1c4c099d…`、`2db61dac…`、`ae94e967…`、
`ea24c3d5…`、`fbfcce7e…`；完整路径及命令记录保留在对应 `/tmp/task004-*` 会话目录。

## 17. 偏差与未声明事项

- 胃部直立到侧躺超时，故本任务不是complete；
- 未对胃部碰撞进行绕行、净空感知或专用参数适配；
- 未验证真实永磁体、执行器、硬件、临床安全性或人工长期主观可用性；
- 隔离平面环境只为消除接触缓存串扰并复现实验，不改变冻结profile或动作判据。
