# 2026-07-17 交接报告之后的增量工作总结

> 整理日期：2026-07-27  
> 基准文档：`/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/docs/PROJECT_HANDOVER_REPORT.md`  
> 当前训练项目：`/mnt/isaac-linux/robotarm_magnetic_lab`

## 1. 当前阶段结论

项目已由 Isaac Sim 单场景扩展进一步迁移成 Isaac Lab 3.0 单环境任务。当前已具备：

- AUBO 六轴机械臂与 ASM 三轴球关节的九轴统一控制接口；
- 240 Hz PhysX 与磁力计算、20 Hz 策略接口、30 Hz 胶囊相机；
- 有限尺寸永磁体磁场、磁力和磁力矩桥接；
- 被动胶囊的重力、摩擦、顺应接触与磁驱动；
- 720p、120° 圆形胶囊 RGB-D 观测；
- 版本化模型输入输出协议；
- Episode 采集、完整性验证和动作块微调索引；
- 胃部 USD 资产的结构、拓扑、坐标和物理预审。

当前主要缺口是胃部资产尚未清洗并接入，正式任务奖励、专家教师策略、软组织参数和
Sim-to-Real 标定尚未完成。

## 2. 已完成工作清单

### 2.1 Isaac Lab 3.0 单环境迁移

- 建立项目：`/mnt/isaac-linux/robotarm_magnetic_lab`。
- 注册任务：`Template-Robotarm-Magnetic-Lab-v0`。
- 建立 `run_isaaclab.sh`，启动时清除 Conda/虚拟环境变量，解决 Python 标准库
  `SRE module mismatch` 和依赖污染。
- 验证 Isaac Lab Cartpole 与自定义任务能够启动。
- 通过兼容 USD 层移动 Articulation Root，使机器人和 ASM 被 Isaac Lab 识别为一个
  九自由度 articulation。
- 修复机器人、ASM、胶囊和磁感线不在同一坐标系的问题。
- 复位时直接进入已验证的机械臂目标关节姿态，不再回放旧 Script Editor 初始化轨迹。

### 2.2 机械臂、ASM 与 Ball 接口

- 固定九轴顺序：
  `j1..j6, ballxj, ballyj, ballzj`。
- 机械臂使用位置控制，Ball 使用三轴位置目标。
- Ball 关节速度限制为 0.8 rad/s，测试最大速度低于 1 rad/s。
- 保留正确的 `l6`–ASM 实际安装关系，没有继续用错误安装方向修正末端朝向。
- 使用经过碰撞检查的机械臂初始关节解，保留 `j6` 冗余角以减少 ASM 干涉。
- 增加 ASM–机械臂近似安全距离和碰撞终止接口。

限制：

- Isaac Lab 当前从已验证姿态开始，不执行旧 cuMotion 全路径规划。
- 现有 ASM 碰撞仍以近似包络/距离为主，不等于完整网格连续碰撞规划。

### 2.3 磁场和磁力迁移

- 将 Isaac Sim 扩展中的 Magpylib 有限尺寸解析模型接入 Isaac Lab。
- 主磁铁和胶囊磁铁的磁力、磁力矩在世界坐标中计算。
- 3D 磁感线随 `magl` 实时更新并保持正确空间锚定。
- 磁力从早期 20 Hz 更新改为每个 PhysX 子步 240 Hz 更新。
- 新增零维 `MagneticPhysicsAction`，不改变模型九维动作空间。
- 策略仍以 20 Hz 输出动作，磁力和接触以 240 Hz 运行。

### 2.4 胶囊几何和被动物理

- 胶囊参数：
  - 外径 13 mm；
  - 总长 25 mm；
  - 壳厚 2 mm；
  - 内部磁体直径 9 mm、长度 8 mm；
  - 总质量约 5.735 g。
- 胶囊是动态非运动学刚体，不直接写入位姿。
- 胶囊仅受磁力、磁力矩、重力、接触、摩擦和被动阻尼影响。
- 修正源 USD 中 Capsule 高度语义错误：
  - 原碰撞体实际约 38 mm；
  - 修正圆柱段为 12 mm，使总长为 25 mm；
  - 已保留源 Stage 备份。
- 校准参数：
  - 最大磁力矩 0.0009 N·m；
  - 静/动摩擦 0.18/0.15；
  - 角阻尼 25；
  - 被动角黏性阻力 0.0012 N·m/(rad/s)。
- 修复“完全不动”和“突破静摩擦后跳起”两种极端状态。

最终 320 步验证结果：

- 胶囊最大磁轴倾角：89.42°；
- 胶囊最大角速度：0.609 rad/s；
- 最大平移速度：9.54 mm/s；
- 平移约 17.1 mm；
- 最大地面间隙为负，未跳起；
- 主磁轴和胶囊磁轴总体跟随，但存在合理惯性、摩擦和阻尼滞后；
- 未检测到 ASM 碰撞。

### 2.5 胶囊相机与视觉接口

- 按 DS01/CX93510 系列设定 provisional 胶囊相机。
- 输出分辨率：1280×720。
- 圆形视野直径 FOV：120°。
- 使用等距广角重映射、圆形遮罩、边缘羽化和渐晕。
- RGB 为 float32 `[0,1]` 策略张量；深度与 RGB 像素对齐。
- 相机随胶囊刚体运动，四个近距离 5600 K LED 随胶囊运动。
- 平面测试中相机可能朝向地面并得到灰色低特征画面；这不是胶囊未运动，而是当前
  场景缺少胃壁纹理和几何特征。

### 2.6 自动验证工具

- `scripts/validate_interfaces.py`
  - 顺序激励九个动作通道；
  - 验证关节读写、胶囊运动、磁感线锚定、碰撞距离。
- `scripts/test_magnetic_tilt.py`
  - 检查重力；
  - 控制 Ball 倾转、保持、三轴进动和恢复；
  - 记录胶囊倾角、速度、位移、地面间隙和碰撞；
  - 输出 RGB-D 快照。
- 通过日志确认胶囊运动是对 Ball 磁轴的被动、滞后响应，不是姿态同步或主动驱动。

### 2.7 模型输入输出与微调数据工作流

- 建立协议版本 `robotarm_magnetic_policy 1.0.0`。
- 输入：
  - 720p 圆形 RGB；
  - 对齐深度；
  - 31 维低维状态；
  - Episode 语言任务指令。
- 输出：
  - 九维归一化关节位置偏置；
  - 动作是相对复位姿态的绝对偏置，不是逐步积分增量。
- 默认训练窗口：
  - 4 帧历史；
  - 8 步动作块；
  - 动作时域 0.4 s。
- 建立原子 Episode 写入器：
  - 未完成 Episode 不进入训练；
  - 保存命令动作和实际关节目标；
  - 教师真值与部署输入分离。
- 增加：
  - `collect_finetune_dataset.py`；
  - `validate_dataset.py`；
  - `build_finetune_index.py`。
- 已生成并验证样例：
  - 1 个 Episode；
  - 12 个同步 RGB-D/状态/动作样本；
  - 2 个时序微调样本；
  - RGB、深度、时间戳、形状和动作范围全部通过校验。

### 2.8 胃部资产预审

资产路径：

`/home/multirobo/Desktop/stomach_obj/stomach_physics.usd`

检查结果：

- Y-up，与当前 Z-up 项目不一致；
- 默认 Prim 为整个 `/World`；
- 自带 PhysicsScene、RenderProduct 和 RenderSettings，不宜直接作为训练资产引用；
- PhysicsScene 重力属性无效；
- 胃网格约 24,529 顶点、49,047 面；
- 21 条边界边，无非流形边和退化面；
- 已有静态三角网格碰撞，`approximation=none`；
- 胃腔为凹形，不能使用 convex hull/decomposition 替代；
- 仅有基础颜色纹理，且颜色纹理被复用于 opacity；
- 尚无湿润材质、法线贴图、组织力学或软体配置。

已确定工作原则：

- Isaac Sim/USD 层负责资产清洗、坐标、网格、材质和碰撞；
- Isaac Lab 负责生成位置、传感器、控制、奖励、随机化和数据采集；
- 不在胃资产中保留独立 PhysicsScene；
- 第一阶段先做静态/顺应接触胃壁，验证后再决定表面或体积 deformable。

## 3. 当前关键文件

```text
/mnt/isaac-linux/robotarm_magnetic_lab/
├── assets/robotarm_magnetic_training.usda
├── configs/interfaces/robotarm_magnetic_v1.json
├── docs/TRAINING_DATA_WORKFLOW.md
├── docs/POST_HANDOVER_WORK_SUMMARY.md
├── docs/PROJECT_RUN_LOG.md
├── scripts/
│   ├── collect_finetune_dataset.py
│   ├── build_finetune_index.py
│   ├── validate_dataset.py
│   ├── validate_interfaces.py
│   └── test_magnetic_tilt.py
└── source/robotarm_magnetic_lab/robotarm_magnetic_lab/
    ├── io/
    │   ├── schema.py
    │   └── episode_writer.py
    └── tasks/manager_based/robotarm_magnetic_lab/
        ├── robotarm_magnetic_lab_env_cfg.py
        └── mdp/
            ├── legacy_bridge.py
            ├── magnetic_action.py
            └── vision.py
```

原 Isaac Sim 扩展仍位于：

`/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim`

其 Magpylib 模型、配置和历史诊断仍是 Isaac Lab 桥接实现的来源。

## 4. 下一步优先级

1. 复制并清洗胃部源 USD，保留原始文件不修改。
2. 删除胃资产内部 PhysicsScene/RenderProduct，建立独立默认 Prim。
3. Y-up 转 Z-up、局部居中并验证真实尺寸。
4. 分离胃部渲染网格和凹形静态碰撞网格。
5. 修正内表面法线、纹理、opacity 和湿润材质。
6. 在 Isaac Lab 中作为独立静态资产接入。
7. 将胶囊复位点移动至胃腔，加入胶囊–胃壁接触验收。
8. 重跑重力、45°磁控、RGB-D 和数据记录测试。
9. 根据实物台架标定摩擦、阻尼和组织顺应性。
10. 再实现专家教师、正式奖励、域随机化和软体胃壁。

## 5. 尚未完成或不得误解为已完成

- 胃部尚未接入当前 Isaac Lab 环境。
- 当前样例数据仅验证记录管线，不是可用于训练的专家数据。
- 当前奖励是 bring-up 奖励，尚不能直接训练最终策略。
- 当前没有完成 EndoVLA 类模型微调。
- 当前没有完成真实相机标定或实物磁场参数标定。
- 当前没有完成软体胃壁、流体或蠕动仿真。
- 当前碰撞距离不等于全程精确网格运动规划。
