# TASK-009B 补充验收与不可达冻结区域交接报告

任务 ID：`TASK-009B-SUPPLEMENTAL-ACCEPTANCE-UNREACHABLE-ROI`

状态：`complete`

说明：本报告所述“补充验收自动项”为通过；原合同中的三视图主观验收仍以现场明确确认记录
为准。本报告状态表示本次获授权的报告整理、冻结区域实现、配置校验和交付已经完成，不把未
记录的人工判断描述为通过。

## 1. 交付身份

- 仓库：`xy211699-arch/robotarm-magnetic-lab`
- 实现分支：`feature/TASK-009B-stomach-coverage-environment`
- 补充验收基线：`0b5c36caedf63c15ce730520f94003523b551fa4`
- 补充验收实现：`a63c1eb414eec4d0b921a170cac4e951be3a8497`
- 补充验收报告提交：`7534102`（完整哈希由远端分支验证）
- 不可达区域实现基线：`7534102`
- 本报告提交前 HEAD：`4802919b9df56744bf9fc9d93931dde9f20d9d16`
- 最终报告提交：由推送后的远端分支 HEAD 验证，避免在报告内形成自引用哈希
- 工作树：`/tmp/robotarm-task009b`

环境身份：Ubuntu 24.04.4、Isaac Lab 3.0、Kit
`110.1.2+production.326809.f9bf0dda.gl`、RTX 5090、NVIDIA 595.84。正式训练链路固定
240 Hz PhysX、10 Hz控制、每个Actor边界24个物理子步。

## 2. TASK-009B补充验收合同与结果

### 2.1 验收边界

补充验收只验证以下内容，未修改已确认的70 mm可见距离、面积加权覆盖、入口区域、位姿库、
动作时序和冻结力度范围：

1. 正式训练环境每个10 Hz Actor边界恰好返回一帧新RGB；
2. Actor观测和覆盖计算使用同一帧；
3. 正式训练环境实际运行在GPU PhysX；
4. 固定60条入口位姿能够在GPU环境回载并HOLD；
5. 外部视角、胶囊相机和覆盖率三视图能够在同一时间线运行。

### 2.2 正式边界同步

正式环境执行顺序固定为：接收动作、推进24个物理子步、仿真时间增加0.1 s、采集新RGB、
以同一RGB更新覆盖、向Actor返回观测。连续1000个混合六模式边界全部满足：

- 每边界24个物理子步；
- 边界仿真时间差为0.1 s；
- 1000个Actor帧ID均唯一且逐一递增；
- Actor帧号与覆盖帧号完全一致；
- RGB与动力学状态有限；
- 累计面积覆盖单调不减。

浮点传感器调度导致415个边界未自动标记相机过期。正式观测路径在这些边界调用一次隔离的
`Camera._update_buffers_impl()`，不推进动作或物理。该接口属于Isaac Lab私有接口，版本升级
时必须复审。

连续100次reset均自动执行10个不计入回合预算的HOLD边界，共1000个连续新帧；稳定后
`episode_length_buf=0`，最后一帧同时初始化Actor观测和初始覆盖。

### 2.3 GPU执行与位姿回载

正式环境、PhysX SimulationView、相机张量和覆盖射线查询设备均记录为`cuda:0`。固定
train/validation/test各20条、共60条位姿在GPU环境中恢复并HOLD 1 s，全部满足：

- 位置、姿态、速度和RGB有限；
- 胶囊长轴无方向倾角不小于45度；
- 相机端和非相机端球心保持胃腔侧；
- 无不可恢复PhysX异常。

### 2.4 三视图状态

三视图实现包含60 Hz外部视角、严格复用策略传感器的10 Hz胶囊RGB，以及与该RGB同帧的
10 Hz覆盖视图。自动Kit烟雾测试验证窗口可创建；合同中的长期主观画面判断仍以用户现场的
`confirmed`/`rejected`记录为最终证据，未记录时不得自动写成通过。

### 2.5 补充验收命令与结果

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force tests/stomach_coverage tests/coverage
# 53 passed，退出码0

./run_isaaclab.sh -p \
  scripts/stomach_coverage/validate_formal_training_runtime.py \
  --device cuda:0
# 1000边界 + 100次reset，通过，退出码0

./run_isaaclab.sh -p \
  scripts/stomach_coverage/validate_gpu_pose_reload.py \
  --device cuda:0
# 60/60通过，退出码0
```

结构化证据位于主项目工件目录：

- `artifacts/task009b_formal_runtime_validation/20260826_104941_589132Z/`
- `artifacts/task009b_gpu_pose_reload_validation/20260826_105520_778472Z/`

文件级字节数和SHA-256已记录在
`handoffs/reports/TASK-009B-stomach-coverage-environment-report.md`，本报告不重复提交大日志。

## 3. 不可达冻结区域设计

### 3.1 指标边界

原始目标仍为49047个胃壁三角面、24529个正权重顶点、总面积
`0.0644836229259155 m2`，胃壁几何SHA-256为
`17ae0bc81e6c9b10d5846998206cafcfae37375ba62708aefd958e99539e9c1b`。

冻结区域只能表示操作者确认的物理或解剖不可达表面，不能因为当前控制器暂时失败而事后剔除。
启用配置后，同时保留三项指标：

- `reachable_coverage_fraction`：剔除冻结区域后重新分配顶点面积权重的正式指标；
- `raw_coverage_fraction`：原完整胃壁ROI覆盖率；
- `excluded_area_fraction`：冻结面积占原ROI的比例。

未显式传入不可达配置时，原覆盖合同保持不变。

### 3.2 标定方式

标定器支持两类选择并取严格并集：

1. **精细测地区域**：三维游标投影到真实最近三角面，以共享边邻接图沿胃壁测地扩展；半径
   为10--80 mm、步长5 mm，不跨胃腔误选对侧胃壁；
2. **世界AABB区域**：两次确定包围盒角点，选择三角面质心位于包围盒内的面片；用于一次
   冻结较大区域。

配置保存种子、半径、包围盒、各选择面片、并集、补集、面积权重哈希、胃壁哈希、操作者和
确定性配置哈希。覆盖视图中灰色为冻结区，红色为可达未覆盖，绿色为历史覆盖，蓝色为当前
可见。

### 3.3 操作审计与最终并集

精细区域来源于会话：
`logs/task009b_unreachable_calibration/20260827_031204_093731Z/events.jsonl`。

- 种子三角面：43041；
- 最近表面点：`[1.058005729945322, 0.23553354582749125, 0.0016917704633850176] m`；
- 测地半径：35 mm；
- 选中面片：2878；
- 面积：`0.0025240219856872115 m2`，占原ROI 3.9142%；
- 原配置哈希：`77a59a0d28ad0dc3f8815144712c447654020aaba4a08f6fbf10d2a9e2d7a950`。

大区域来源于会话：
`logs/task009b_unreachable_calibration/20260827_034123_811613Z/events.jsonl`。

- AABB最小点：`[0.9798418860441885, -0.025843150952134245, -0.03634503393723226] m`；
- AABB最大点：`[1.1114344854247182, 0.06371601786755308, 0.08190591491294959] m`；
- 选中面片：12363；
- 面积：`0.018799674267324363 m2`，占原ROI 29.1542%；
- 仅包围盒配置哈希：`5640e4f3eb08de230018e51fa1bd4d2f09cca25d4bd773b6627c82b972ae8d4d`。

本轮根据用户明确要求，从审计日志恢复精细区域并与AABB重新计算并集。两区域没有重复面片，
最终冻结配置为：

- 种子数：1；包围盒数：1；
- 排除面片：15241；
- 排除面积：`0.021323696253011573 m2`，占原ROI `33.06839052376156%`；
- 剩余可达面积：`0.043159926672903914 m2`；
- 配置SHA-256：`d4f1a29e238aae3aa448103dbf191beb1972a1c277765b1770793aff71ef02a7`；
- 配置文件SHA-256：`044bd8f26e24dcc092e6365c97a6b37a21688cb60a35fab88d7ace05868dcb31`；
- 正式配置：`configs/task009b/unreachable_region_v1.json`。

仅包围盒版本在Linux本地保留审计备份，其文件SHA-256为
`bcdefd3fa74fff4aade73d85b7c4d59e9291ae9073cfbd8dc314507c682c2de5`；该冗余生成文件不作为
正式运行配置。

## 4. 不可达区域实现提交与文件

实现提交：

- `d5f1a8910a02e64b506b49c25744bbafe1cd2263`：多种子测地区域、双覆盖指标和可视化接入；
- `b03d91626766e1598445bb3ffc4fc3a571304578`：作废误保存配置并保留审计；
- `26909a0b43b4783d5da61e39ce8bc8088e971a7d`：消除移动键与保存键冲突；
- `55d0e988ae50f3278f7623a6b72aa7ceaf56c427`：已有配置恢复与跨会话增量标定；
- `4802919b9df56744bf9fc9d93931dde9f20d9d16`：AABB选择与v2并集配置。

核心文件：

- `source/.../coverage/unreachable_region.py`
- `source/.../coverage/simulator_runtime.py`
- `source/.../ui/coverage_view.py`
- `scripts/stomach_coverage/calibrate_unreachable_regions.py`
- `scripts/stomach_coverage/teleop_stomach_coverage.py`
- `scripts/stomach_coverage/validate_formal_training_runtime.py`
- `configs/task009b/unreachable_region_v1.json`
- `docs/TASK009B_UNREACHABLE_REGION_CALIBRATION.md`

## 5. 验证命令与结果

最终配置通过Isaac Lab现场加载、胃壁哈希匹配、配置哈希、并集重算、补集重算、面积守恒、
面积权重哈希及保存后重载验证。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest -q \
  tests/coverage tests/stomach_coverage
# 53 passed，退出码0

./run_isaaclab.sh -p -m pytest \
  tests/stomach_coverage/test_unreachable_region.py -q
# 4 passed，退出码0
```

启用最终冻结配置的人工三视图命令：

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/teleop_stomach_coverage.py \
  --task Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0 \
  --device cuda:0 \
  --viz kit \
  --unreachable_region configs/task009b/unreachable_region_v1.json
```

## 6. 偏离、限制与未验证声明

- 未修改70 mm、120度圆形FOV、面积加权方法、入口区域、位姿库、10 Hz/240 Hz时序或力度；
- 不可达配置为显式选项，不传入时不会静默改变历史覆盖结果；
- 胃模型几何、种子、半径或AABB变化都会使配置哈希变化，旧训练结果不得与新分母混用；
- 冻结区域是人工实验定义，不证明其中每个三角面在所有未来硬件或控制器下数学上不可达；
- 本轮没有重新运行学习策略，也没有把冻结后的覆盖率解释为策略性能；
- 三视图长期主观验收若没有明确确认记录，仍属于人工待确认项。

