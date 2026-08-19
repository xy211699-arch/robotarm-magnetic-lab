# 开环外磁发生器胶囊控制器交接文档

日期：2026-08-19

仓库：`xy211699-arch/robotarm-magnetic-lab`

基线提交：`8f89f85`（`origin/main`）

交接分支：`agent/open-loop-magnetic-controller-handoff`

交接性质：文档、代码清单与证据归档；未修改控制器、环境、资产或既有实验结果。

配套校验清单：
`handoffs/reports/OPEN_LOOP_MAGNETIC_CONTROLLER_CODE_MANIFEST.sha256`。该清单覆盖仓库中
构成此控制链路的控制器、桥接、任务配置、资产组合层、可视化和测试入口。

## 1. 交接结论

目前效果最完整的控制链路是早期建立的开环外磁发生器控制器，而不是后续TASK-005
十一动作动态控制器。该控制器只产生机械臂六轴和Ball三轴位置参考，胶囊始终是被动
动态刚体，其运动由有限尺寸永磁体磁场、磁力/磁矩、重力、接触、摩擦和阻尼共同产生。

仓库中已经包含以下部分：

- 期望磁场方向到Ball三轴的有限场数值逆解；
- 机械臂末端小位移到六轴关节增量的阻尼最小二乘规划；
- 倾斜方位、直立/侧躺转换、长轴被动滚动和连续组合轨迹；
- 240 Hz磁力/磁矩注入PhysX的Isaac Lab桥接；
- 平整桌面与静态胃部两个单环境任务；
- 主视口、胶囊相机、外部姿态跟随视图、逐帧遥测和验收汇总。

当前仓库仍不是完全自包含包。磁场解析模型、磁体参数、磁感线算法和XRDF位于独立
Isaac Sim扩展目录；基础Stage、胃部源USD及贴图也通过绝对路径或外部引用加载。第10节
列出了全部外部依赖和校验值。升级前应先完成依赖内聚，不能直接删除
`LegacyMagneticCollisionBridge`中的外部加载路径。

## 2. 控制与物理链路

```text
预设运动模式/阶段
    ↓
期望磁场轴序列（世界坐标）
    ↓
BallFieldPlanner有限场逆解
    ↓
ballxj / ballyj / ballzj绝对位置参考
    ↓
必要时由arm_gradient_plan生成j1..j6梯度位移参考
    ↓
外部N52立方磁铁magl的世界位姿变化
    ↓
FiniteMagnetSystem计算B、F、τ
    ↓
MagneticPhysicsAction每个240 Hz物理子步注入外力/力矩
    ↓
胶囊在重力、胃壁/桌面接触和摩擦下自然响应
    ↓
胶囊真值仅写入telemetry/summary，不回流到控制命令
```

这是严格开环控制。`TableTestHarness.capture()`会读取胶囊位姿、接触、磁力和速度用于
离线验收，但`BallFieldPlanner`和轨迹生成只使用预设方向、初始标称几何、机械臂/磁铁
状态，不使用实时胶囊位姿修正目标。

## 3. 时间、动作和坐标约定

| 项目 | 当前实现 |
|---|---|
| PhysX频率 | 240 Hz，`dt=1/240 s` |
| 开环动作频率 | 20 Hz，`decimation=12` |
| 动作维度 | 9：`j1..j6, ballxj, ballyj, ballzj` |
| 动作语义 | 相对复位关节姿态的归一化绝对位置偏置，不是积分增量 |
| 磁力更新 | 零维`MagneticPhysicsAction`，每个PhysX子步更新，不增加策略动作维度 |
| 世界坐标 | Z-up，重力沿世界-Z |
| 主磁铁充磁轴 | `magl`局部+Z，N52，50×50×50 mm立方体 |
| 胶囊磁铁充磁轴 | 胶囊局部+Z，N52，直径9 mm、长度8 mm轴向充磁圆柱 |
| 胶囊外形 | 直径13 mm、长度25 mm、壳厚2 mm |
| 胶囊质量 | 约5.735 g |
| Ball关节顺序 | Y-X-Z轴链：`ballxj, ballyj, ballzj` |
| Ball速度上限 | 0.8 rad/s |

基础环境的机械臂归一化动作范围为复位角附近±0.05 rad；平桌任务扩大为±0.25 rad。
组合运动实例临时使用机械臂±0.45 rad和Ball±π，以覆盖完整磁场半球和约100 mm梯度
位移。该临时放大只作用于组合脚本创建的环境实例，不修改训练任务接口。

## 4. 核心控制代码清单

### 4.1 轨迹与逆解

| 文件 | 主要职责 |
|---|---|
| `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/table_motion.py` | 运动模式枚举、五次平滑插值、倾角/方位轴生成、胶囊支撑高度、Ball轴链旋转、有限场方向逆解、机械臂梯度阻尼最小二乘 |
| `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/__init__.py` | 对外导出上述控制接口 |
| `scripts/table_motion/table_test_common.py` | 开环阶段执行器、场规划器构造、动作发送、遥测、验收判据及全部平桌运动序列 |
| `scripts/stomach_motion/stomach_test_common.py` | 复用同一控制器，将任务切换为胃部并保留胃液角阻尼 |

`BallFieldPlanner.solve()`的实现要点：

1. 目标方向归一化；
2. 固定当前`ballzj`，在`ballxj/ballyj`范围内搜索；
3. 代价为磁场方向误差加很小的关节连续性惩罚；
4. 可先执行13×13全局粗搜索，再做最多18轮坐标局部细化；
5. 输出绝对Ball角度、磁场强度、实际方向和方向误差；
6. 再按默认角与动作尺度转换为归一化绝对动作。

`arm_gradient_plan()`对6×6末端Jacobian做阻尼最小二乘。组合滚动锁定`j6`，使其保持
碰撞验证过的`-0.28688 rad`，只使用`j1..j5`产生磁场源平移；姿态行使用0.10权重，
优先满足梯度位移。

### 4.2 磁场与PhysX桥接

| 文件 | 主要职责 |
|---|---|
| `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/magnetic_action.py` | 零维内部ActionTerm；每个物理子步调用磁场桥，不改变9维外部动作 |
| `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/legacy_bridge.py` | 加载有限磁体模型、计算双向力/矩、耦合渐入、阻尼、限幅、滤波、PhysX外力注入、磁感线更新和ASM-XRDF球体间隙 |
| `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py` | 注册桥接、观测和动作项 |
| `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/vision.py` | 720p RGB-D圆形视野处理，独立于控制器 |

`LegacyMagneticCollisionBridge.physics_step()`在每个物理子步执行：

- 读取`magl`、胶囊内部磁铁、机器人基座和胶囊刚体世界状态；
- 按内部磁铁相对胶囊中心的-4 mm偏置计算真实磁体中心；
- 通过Magpylib有限立方体/有限圆柱体模型分别计算作用于主磁铁和胶囊的力/矩；
- 对力和力矩限幅，按0.5 s耦合渐入并做0.15 s一阶滤波；
- 胃部场景叠加线性/角黏性阻力；
- 将世界坐标外力/力矩写入机器人`magl`刚体和胶囊刚体；
- 每个策略步刷新ASM-XRDF球体间隙与3D磁感线诊断。

磁场不是PhysX内建电磁求解。PhysX只负责刚体、关节、重力和接触，解析磁力作为外部
wrench注入。

## 5. 运动模式实现

### 5.1 倾斜姿态保持与方位旋转

入口：

- `scripts/table_motion/test_03_tilt_azimuth.py`
- `scripts/stomach_motion/test_01_tilt_azimuth.py`

当前目标定义为胶囊磁轴相对世界XY平面-45°，相机位于局部-Z端，因此相机头目标为
+45°朝上。控制器沿固定极角改变方位角，并用五次曲线在场方向节点之间插值。

### 5.2 直立与侧躺转换

入口：

- `scripts/table_motion/test_04_upright_to_side.py`
- `scripts/stomach_motion/test_02_posture_transition.py`

期望磁场轴在世界竖直和水平之间连续变化。相同轨迹反向执行可完成侧躺到直立。胶囊
姿态不是被直接设置；脚本只在测试初始化阶段放置初始状态，正式阶段由磁矩和接触自然
响应。

### 5.3 长轴被动滚动

入口：

- `scripts/table_motion/test_05_long_axis_roll.py`
- `scripts/stomach_motion/test_03_long_axis_roll.py`

轴向充磁磁矩与胶囊长轴同轴，均匀磁场不能直接产生绕长轴力矩。实现因此保持Ball提供
轴向磁场，同时移动机械臂末端形成横向场梯度；磁力产生平移趋势，接触摩擦把平移转换为
绕胶囊长轴的被动滚转。验收同时比较净位移和`半径×累计滚角`，避免把纯滑动误判为滚动。

### 5.4 连续组合运动

入口：

- `scripts/table_motion/test_06_composite_motion.py`
- `scripts/stomach_motion/test_04_composite_motion.py`

当前阶段序列：

```text
侧躺初始化
→ 侧躺保持
→ 侧躺到头朝上直立（10 s）
→ 过渡到头朝上45°（5 s）
→ 固定倾角完成一圈方位旋转（24 s）
→ 返回直立（5 s）
→ 直立到侧躺（10 s）
→ 保持轴向场并沿世界+X移动磁场源（18 s）
→ 最终保持（6 s）
```

阶段由同步阻塞的`run_phase()`执行。当前没有暂停时钟、取消令牌、抢占切换或安全过渡
状态机；新一代控制器应将其改造成可逐步调用的非阻塞状态机，但必须保留本控制链路作为
回归基准。

## 6. Isaac Lab环境实现

| 文件 | 内容 |
|---|---|
| `source/.../robotarm_magnetic_lab_env_cfg.py` | 共享机器人、ASM、胶囊、相机、LED、9轴动作、240/20 Hz时序、磁场桥、观测、奖励和碰撞终止 |
| `source/.../robotarm_magnetic_table_env_cfg.py` | 平桌覆盖层、侧躺初态、ContactSensor、平桌动作范围和观察视角 |
| `source/.../robotarm_magnetic_stomach_env_cfg.py` | 胃部USD、静态胃壁ContactSensor、1 Hz策略相机、胃内照明和观察视角 |
| `source/.../__init__.py` | 注册`Template-Robotarm-Magnetic-Table-Lab-v0`与`Template-Robotarm-Magnetic-Stomach-Lab-v0` |
| `source/robotarm_magnetic_lab/robotarm_magnetic_lab/ui/capsule_camera_view.py` | 30 Hz工程相机窗口和外部姿态跟随窗口，不进入训练观测 |

平桌环境：

- 静/动摩擦0.55/0.48，恢复系数0；
- 胶囊线/角阻尼0.10/0.50；
- 长轴滚动脚本把磁力安全上限临时设为0.040 N；
- 胶囊始终是非运动学动态刚体。

胃部环境：

- 胃壁是静态、双面、精确三角网格碰撞，视觉网格与碰撞网格分离；
- 胃部包装层采用1.7倍统一缩放、世界Y轴180°翻转并保持Z-up仿真；
- 胃壁静/动摩擦0.20/0.15，恢复系数0；
- 胃壁当前不是FEM/可变形体，没有蠕动、CFD或组织形变；
- 策略相机1 Hz，工程预览30 Hz，二者不得混作训练采样频率。

## 7. USD与环境资产

| 仓库文件 | 作用 | SHA-256 |
|---|---|---|
| `assets/robotarm_magnetic_training.usda` | 基础Stage兼容层、胶囊尺寸/阻尼/柔顺接触覆盖 | `3e764afe22f1510cfb7568f947e7b5246244d37768085266319978a7a3d9aa99` |
| `assets/robotarm_magnetic_table_training.usda` | 平桌摩擦、胶囊阻尼和视觉滚动标记 | `3a149426378e77b58d9a3a1b89518e90c67da0772dc346e9f08d9738e57bea96` |
| `assets/robotarm_magnetic_stomach_training.usda` | 关闭平桌与标记，保留机器人/ASM/胶囊 | `002a342a1cbd24e181be47afb800d903e3359d501a05d533a761cbe8f6731feb` |
| `assets/stomach/stomach_environment_lab.usda` | 胃部坐标、缩放、翻转、碰撞/视觉分离和材质覆盖 | `e617e6ec2cfd58b0426cefa74299efe4730d6351b43cf7c9bc0f166c54dcb607` |

这些USDA是轻量组合层，不包含全部二进制几何和贴图。基础层仍引用：

```text
/home/multirobo/Desktop/sim of FF/Stage.usd
```

胃部层仍相对引用未纳入当前Git历史的：

```text
assets/stomach/stomach_capsule_test_v1_source.usd
```

因此Windows方案端可以审阅环境实现，但新Linux机器仅克隆本仓库不能完整运行场景。

## 8. 可视化与复现命令

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab
```

平桌完整组合：

```bash
./run_isaaclab.sh -p scripts/table_motion/test_06_composite_motion.py \
  --capsule_camera_view \
  --capsule_pose_view
```

胃部完整组合：

```bash
./run_isaaclab.sh -p scripts/stomach_motion/test_04_composite_motion.py \
  --capsule_camera_view \
  --capsule_pose_view
```

平桌单项测试：

```bash
./run_isaaclab.sh -p scripts/table_motion/test_01_baseline.py
./run_isaaclab.sh -p scripts/table_motion/test_02_axial_field_scan.py
./run_isaaclab.sh -p scripts/table_motion/test_03_tilt_azimuth.py
./run_isaaclab.sh -p scripts/table_motion/test_04_upright_to_side.py
./run_isaaclab.sh -p scripts/table_motion/test_05_long_axis_roll.py
```

胃部单项测试：

```bash
./run_isaaclab.sh -p scripts/stomach_motion/test_01_tilt_azimuth.py
./run_isaaclab.sh -p scripts/stomach_motion/test_02_posture_transition.py
./run_isaaclab.sh -p scripts/stomach_motion/test_03_long_axis_roll.py
```

追加`--contact_debug`可显示2 mm接触标记；它仅用于工程观察。无窗口运行使用
`--visualizer none --no-realtime`。日志分别写入：

```text
logs/table_motion/<scenario>/<timestamp>/telemetry.jsonl
logs/table_motion/<scenario>/<timestamp>/summary.json
logs/stomach_motion/<scenario>/<timestamp>/telemetry.jsonl
logs/stomach_motion/<scenario>/<timestamp>/summary.json
```

## 9. 已观测结果与证据

下表只列本机已有日志直接支持的结果。日志未提交Git，只记录路径和SHA-256。

| 场景 | 直接观测结果 | 证据 |
|---|---|---|
| 平桌无磁基线 | PASS；静置末速8.14 µm/s，跌落最大接触误差约0.327 mm | `logs/table_motion/baseline/20260730_221811/summary.json`，SHA `401205091d3caec374aba5bc013cd1d0787f1518e546a3632eba9c36da81c100` |
| 平桌场扫描 | PASS；方向覆盖133.62°，轴向力矩恒等误差0 | `logs/table_motion/field_scan/20260730_222205/summary.json`，SHA `3f0c3d3a2cd5bdab85b2715dee2e01ed208f35e33b7feadfab14b42c902b1505` |
| 平桌姿态转换 | PASS；0.20°直立到85.10°侧躺，接触率100% | `logs/table_motion/upright_to_side/20260730_222017/summary.json`，SHA `97954a556aa409bcac7f48f098bfe469f197226512e50418b3a74f585e3a0b0a` |
| 平桌长轴滚动 | PASS；55.86 mm净位移，累计滚角6.38 rad，滑移率0.258 | `logs/table_motion/long_axis_roll/20260730_221655/summary.json`，SHA `a753a2bbd4d198e782d4a7cab5509421ffac3ef6189f9f1bca99ec693f76549b` |
| 平桌旧组合基线 | PASS；+X 97.79 mm，接触率100%，ASM最小间隙9.83 mm | `logs/table_motion/composite_motion/20260731_004014/summary.json`，SHA `b70748a09e58e4ab3a653273bb5dc35cd72f6919e4d3a638a795a93bd6a57713` |
| 胃部当前45°倾斜方位 | PASS；相机头仰角33.95°、方位变化77.96°、接触率100% | `logs/stomach_motion/tilt_azimuth/20260806_185150/summary.json`，SHA `4d5ab4516d9dbef6c201e9e6ba8a3c2e0eb104d943732da98b30b9fff8785272` |
| 胃部当前45°组合 | 姿态阶段通过；方位356.19°，但滚动仅11.17 mm，整体FAIL | `logs/stomach_motion/composite_motion/20260731_131913/summary.json`，SHA `a685ebf5ae9f01e6069e671defcb0405d1de2c1e022c0afcc68a2144bc197c38` |

平桌组合证据生成于轨迹改为头朝上45°之前，只证明同一控制链路的旧组合基线；当前
45°版本尚未在本次交接中重新跑平桌组合，不应把旧97.79 mm结果宣称为当前提交的新验收。

## 10. 外部运行时依赖

`legacy_bridge.py`当前硬编码：

```text
/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim
```

实际加载文件如下。这些文件是控制链路运行所必需，但当前没有复制到交接仓库：

| 外部文件 | 作用 | SHA-256 |
|---|---|---|
| `robotarm/magnetic_sim/config.py` | 读取并校验磁体、机器人和物理参数 | `5d32740c62a75e06b7b876ed16f0043378ad45b72317b1f99637466b7f71ee07` |
| `robotarm/magnetic_sim/magnetics/field_models.py` | Magpylib有限立方体/圆柱体B、F、τ | `be2f4d4af8db2e3a04552add61cbbc84d89e2348c08864a0c9cc3e6283265965` |
| `robotarm/magnetic_sim/magnetics/streamlines.py` | 三维磁感线中点积分 | `a0a1c04e0250dd1eb9a2079475b655dd72081b53f9da0c0b8cf43802d42bb27c` |
| `robotarm/magnetic_sim/visualization/magnetic_field.py` | USD BasisCurves磁感线可视化 | `c8c5f502b4d1d4a6d25fa11fcb4e1e3e4b6a86eb9845a38c34682ea082c57069` |
| `data/config/default.json` | N52磁体、胶囊、力矩上限、阻尼、滤波和场线参数 | `e38563d558f6945f3041458060965ce6cd4b7044eacce573318c0f0fdcd319a6` |
| `data/planning/robot.xrdf` | ASM-机械臂球体包络和自碰撞间隙 | `5293e63d4112c67f479bdfa88ad8cb520c864c69939f0683a8dea5ca08146f40` |

Magpylib 5.2.3及其运行依赖当前放在扩展的`vendor/`目录。后续若进行控制器升级，第一项
工程任务应将上述纯计算代码、配置、XRDF和必要许可证内聚到本仓库，再把绝对路径改为
包资源或环境变量；迁移完成前必须保留SHA回归，避免磁力数值悄然变化。

## 11. 已知限制

1. 胶囊无在线姿态/位置反馈；胃部褶皱和摩擦偏差不会被补偿。
2. 轨迹阶段同步阻塞，不能安全暂停、取消、抢占或从中间状态恢复。
3. 磁场逆解使用初始化时的标称胶囊磁体位置；机械臂梯度运动后场参考点不会闭环更新。
4. 胃部100 mm滚动未实现，当前约11 mm；不得通过降低验收阈值掩盖。
5. XRDF只提供ASM-机械臂近似球体间隙，不是机械臂/ASM/胃壁全程连续网格规划。
6. 胃部是刚性静态三角网格，不是柔性、黏弹或蠕动组织。
7. 摩擦、阻尼、N52剩磁、光照和相机模型仍是暂定值，未完成实物标定。
8. 胶囊位姿虽然不进入控制器，却被测试脚本用于初始化和离线验收；训练/部署接口必须继续隔离真值。
9. 多处绝对路径和未入库二进制资产阻止干净机器直接运行。

## 12. 建议升级顺序

1. **依赖内聚**：迁移旧扩展纯计算代码、默认参数、XRDF、Magpylib依赖和许可证，消除绝对路径。
2. **冻结回归基线**：固定平桌/胃部种子、参数摘要和场/力数值测试，重新运行当前45°平桌组合。
3. **非阻塞执行器**：把`run_phase()`改为`reset/submit/step/pause/cancel`状态机；暂停时保持当前安全场目标。
4. **命令空间稳定化**：高层输出期望场方向、场梯度位移和持续时间；低层继续负责Ball逆解和机械臂梯度规划。
5. **安全层分离**：机械臂/ASM/胃壁路径安全只使用机器人、命令和环境几何；不要用胶囊真值做设备硬失败。
6. **胃部局部规划**：按局部表面法线和切向方向规划外磁体路径；允许胶囊自然偏差，但不要继续套用平桌世界+X增益。
7. **实物标定**：标定磁铁磁矩、相对安装、摩擦、滚动阻力、阻尼、胶囊质量惯量和组织顺应性。
8. **训练接口**：控制器作为专家/动作执行后端，模型输出受限的磁场原语或短时目标，不直接输出任意九轴角。

## 13. 交接验收清单

- [x] 控制链路、时间尺度与开环边界已说明。
- [x] 所有仓库内控制、桥接、环境、脚本和可视化文件已列入代码清单。
- [x] 关键仓库文件SHA-256写入伴随清单。
- [x] 外部磁场扩展、参数、XRDF和资产缺口已明确，不宣称自包含。
- [x] 平桌和胃部可视化/无窗口入口已给出。
- [x] 已有日志路径、哈希、通过项和失败项已区分。
- [x] TASK-005及后续控制器没有混入本控制器能力声明。
- [ ] 当前45°平桌组合尚未在本交接分支重新运行。
- [ ] 外部纯计算代码、Magpylib依赖、XRDF和二进制USD尚未迁入交接仓库。
- [ ] 新机器冷启动复现尚未验证。
