# TASK-009B Linux 执行报告

状态：`needs_input`

当前门禁：Gate 4已通过；Gate 5三视图实现及自动烟雾测试已通过，等待现场人工验收。

## 基线与实现

- 规划分支：`workflow/TASK-009B-stomach-coverage-environment`
- 规划提交：`c57ce69873cecd7c21db05f5e656bf4f77b4b626`
- TASK-009A 控制器基线：`335c5f563da51c50656729db86a7872809c58ada`
- 实现分支：`feature/TASK-009B-stomach-coverage-environment`
- Gate 1 提交：`3de692e`
- Gate 2 替代方案提交：`3fa7405ccb7d528a4f5b96d50154a32961e75abf`
- Gate 2 动力学定位提交：`ccbbef7`
- Gate 2 最终配置提交：`52e7383`
- 已取消的包围盒历史提交：`c2205c5cd73ee766c6ce32735cbf2c762fdf1dae`

## Gate 1：环境集成

状态：`pass`

实现了独立单环境胃部任务，直接使用 TASK-009A 的
`ParameterizedForceActionTermCfg`，未引入 TASK-008 宏动作路径。环境固定为 240 Hz
物理、10 Hz动作、每动作边界24个物理子步和每边界一次策略RGB采集。

自动化命令：

```bash
./run_isaaclab.sh -p -m pytest -q \
  tests/parameterized_force tests/runtime tests/stomach_coverage
```

Gate 1 live 命令：

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/validate_environment_integration.py \
  --headless --device cuda:0 \
  --output_directory /tmp/task009b-environment-integration
```

直接观测：HOLD、MOVE_POS、MOVE_NEG、VIEW_POS、VIEW_NEG、UP依次各执行一个0.1 s
边界；六个边界均为24子步，主动模式24/24施力、HOLD 0/24，RGB帧1至7每边界递增
一次，状态与RGB均有限，Actor观测只有`policy.rgb`。

外部证据：

- 目录：`/tmp/task009b-environment-integration/20260825_160138_439362Z`
- `environment_cycles.jsonl`：7777字节，SHA-256
  `620da6916435e4ffe4efd569d7af512d82f66f6d6fe8ed31f01ab29e0e93ef2b`
- `summary.json`：326字节，SHA-256
  `b8433bdaf3359435fa31c0db0f2fa33f1ad8682bed02181d4781c9363643ef88`

## Gate 2：稳定锚点与胃壁测地区域

状态：`pass`

用户新合同已完全取消三维包围盒；随后现场否决旧锚点扩大区域，当前替代实现包括：

- 从当前合理默认位姿开始，胶囊保持Dynamic并正常推进PhysX；
- 复用TASK-009B既有参数化MOVE/VIEW/UP控制器，以240 Hz物理、10 Hz控制和24子步/周期
  进行人工动力学定位，未修改已冻结的力度映射；
- Enter在控制边界切换HOLD并清除主动Actor力，但保留当前速度，随后只由惯性、重力、胃壁
  碰撞及摩擦自然落稳；
- 按240 Hz逐步检查，连续0.25秒满足2 mm/s及5 deg/s才报告稳定，2秒超时返回控制；
- Y确认稳定锚点，Backspace拒绝并从当前状态恢复动力学控制，R才恢复默认位姿；
- 通过点到三角形真实最近点确定种子，未使用最近顶点近似；
- 通过胃壁三角形共享边邻接图和Dijkstra距离，以5 mm步长生成10–80 mm测地区域；
- 高亮最近表面点和当前区域，区域仅一个连通分量时允许Enter保存；
- 锚点与区域配置绑定胃壁哈希并相互绑定，保存后执行精确重载校验。

自动化回归`24 passed`；现场动力学定位、按键确认和区域画面已由用户验收。

Kit键盘对部分按键返回裸字符串而非Input对象导致过一次回调异常，现继续兼容两种事件格式。

用户已完成一次30 mm入口区域保存：种子面片17589、1375个三角面、752个顶点、面积
0.0017662011374897796 m2、单连通分量，锚点哈希为
`9b5b33fef14bd183133818b60beac2d5b660d49a330f4a208023c3c9fd57bcec`。随后基于该锚点预览
扩大区域时发现高亮进入非目标胃壁，因此该锚点及其扩大方案已被用户否决，只作为历史证据
保留；随后已通过MOVE/VIEW/UP重新定位并生成下述最终锚点和区域。

最终确认配置：

- 锚点SHA-256：`0166638c32e7023995f9de2ad041afe464ecf65a99cac91959dba8709323b6fc`；
- 胃壁几何SHA-256：`17ae0bc81e6c9b10d5846998206cafcfae37375ba62708aefd958e99539e9c1b`；
- 种子面片：34914；测地半径：55 mm；
- 区域：6171个三角面、3213个顶点、0.00636984922518564 m2、单连通分量；
- 区域SHA-256：`1c38444c7e1f3fade5e6e1f81aa84fad53a84f09276b69b5bc683d6ca9ba26e7`；
- 锚点与区域均通过确定性哈希和重新加载校验。

复现标定命令：

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/calibrate_entry_anchor_region.py \
  --device cuda:0 \
  --viz kit
```

按键及验收方法见`docs/TASK009B_ENTRY_ANCHOR_REGION_CALIBRATION.md`。

## Gate 3：有效初始位姿库

状态：`pass`

生成器从已确认入口区域按面片面积采样，随机化局部切向方位、相机端方向和完整滚转角。
所有候选仅执行HOLD并在冻结门限下真实松弛，未用速度清零伪造稳定状态，也未放宽2秒、
0.25秒、2 mm/s、5 deg/s、相机在胃腔内和长轴角不小于45度的条件。

生成命令：

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/generate_entry_pose_library.py \
  --device cuda:0
```

生成结果：train/validation/test为1000/100/100，共1200条互不重复状态；实际尝试
1031/101/105次，拒绝原因为30条未在2秒内稳定、6条相机不在胃腔、1条长轴角不足。

固定回载命令：

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/validate_entry_pose_library.py \
  --device cuda:0
```

回载结果：三组各固定20条，共60条全部通过；位置、姿态、RGB和物理状态有限，无方向
长轴角均不小于45度。独立结构审计确认1200个ID、候选种子和位姿指纹全唯一，分组互斥，
清单与数据本体哈希一致。

外部证据：

- 位姿库：`/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009b_pose_library/20260826_040250_292641Z/pose_library_v1.jsonl`
- 位姿库：1935122字节，SHA-256 `7a7a20e175dcfade0c3f07ccc2a4dca377508485f726fc6999a0453cb7cea855`
- 回载日志：`/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009b_pose_library_validation/20260826_041331_486142Z/live_reload.jsonl`
- 回载日志：37279字节，SHA-256 `b88fd775db7f49be65bf0dfaddb8c082a1a1aa8ea125b30bc02c08fe2b908d6b`
- Git清单：`configs/task009b/pose_library_manifest_v1.json`，包含分组、接受种子、拒绝统计、
  固定回载ID、外部绝对路径和哈希。

## Gate 4：七十毫米面积加权覆盖

状态：`pass`

旧覆盖代码的50 mm和顶点计数比例已替换为冻结合同要求的70 mm与面积权重。每个目标
三角形面积的三分之一累加到三个顶点；当前和累计覆盖同时经过120度圆形FOV、70 mm距离、
胃腔侧法向和CUDA第一命中遮挡。累计率、记录写入、一致性检查、快照与运行时均使用面积。

自动化命令：

```bash
./run_isaaclab.sh -p -m pytest -q \
  tests/coverage tests/stomach_coverage tests/parameterized_force tests/runtime
```

结果：`57 passed`，覆盖非均匀面积、距离和视场边界、法向、第一命中、累计并集、reset和
单调性。

live命令：

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/validate_coverage_calculation.py \
  --device cuda:0 --raycast_device cuda:0
```

live从Gate 3固定20/20/20位姿加载。60个连续边界均产生非零当前可见面积，累计率严格单调
并达到53.4965347710799%；随后reset清为0，并在首个有效RGB边界得到
`C0=1.9400115426864994%`。共61个边界的RGB与物理状态均有限。

目标集合保持旧P0批准的胃腔内表面：24529顶点、49047三角面、总面积
0.0644836229259155 m2；胃壁几何SHA-256为`17ae0bc...9c1b`，顶点权重SHA-256为
`11adc45c...9c92`。0.1秒浮点调度有21个边界没有自动标记传感器outdated，验证器仅在这些
已记录边界执行不推进物理和动作的必要采集。

外部证据：

- 日志：`/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009b_coverage_validation/20260826_042440_575139Z/coverage_boundaries.jsonl`
- 日志：36309字节，SHA-256 `2758caf3b65655e7d560c635aecff6b0beeb52cb8be9fda6851a280571fdb856`
- 摘要：同目录`summary.json`，1736字节，SHA-256
  `f4b26c8d15d2827aa2b42eb5fed0cc7c334a8f72eb96d5f9638eb1c8f5b47773`
- Git清单：`configs/task009b/coverage_manifest_v1.json`。

## Gate 5：同时间线三视图

状态：`needs_input`

新增`teleop_stomach_coverage.py`，从Gate 3冻结位姿启动，并在同一时间线上提供60 Hz外部
主视口、严格复用策略记录传感器的10 Hz胶囊RGB窗口，以及10 Hz隔离覆盖窗口。未创建
额外30 Hz预览相机。覆盖窗口以红/绿/蓝分别显示未覆盖、历史累计和当前可见区域，HUD为
70 mm面积加权覆盖百分比；轨迹曲线只在至少两个点时创建有效拓扑。

自动化回归59/59通过。3周期无界面live按HOLD、MOVE_POS、VIEW_POS运行，RGB帧1→4，
面积覆盖率13.375%→15.307%→17.363%→17.669%，每个边界均为24个物理子步且状态有限。
2周期Kit烟雾测试确认主窗口、`Capsule Camera | Recorded 10 Hz`和
`P0 Stomach Coverage`均成功创建，记录传感器显示`extra_sensor=false`。

现场命令、按键及验收项见`docs/TASK009B_THREE_VIEW_ACCEPTANCE.md`。合同要求的主观画面
确认必须由用户完成，因此当前不能标记为`complete`。

## 待人工验收门禁

- Gate 5 三视图现场验收：等待用户运行可视化并确认画面。

本报告未把未执行项描述为通过；位姿库数据本体按合同仅保存在Linux外部工件目录。
