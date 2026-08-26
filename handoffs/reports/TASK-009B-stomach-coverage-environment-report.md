# TASK-009B Linux 执行报告

状态：`partial`

当前门禁：Gate 3已通过；Gate 4七十毫米面积加权覆盖计算正在实施。

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
保留。Gate 2现在等待通过MOVE/VIEW/UP重新定位后生成的新锚点和新区域。

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

- 位姿库：`/mnt/isaac-linux/robotarm_magnetic_lab_artifacts/task009b_pose_library/20260826_040250_292641Z/pose_library_v1.jsonl`
- 位姿库：1935122字节，SHA-256 `7a7a20e175dcfade0c3f07ccc2a4dca377508485f726fc6999a0453cb7cea855`
- 回载日志：`/mnt/isaac-linux/robotarm_magnetic_lab_artifacts/task009b_pose_library_validation/20260826_041331_486142Z/live_reload.jsonl`
- 回载日志：37279字节，SHA-256 `b88fd775db7f49be65bf0dfaddb8c082a1a1aa8ea125b30bc02c08fe2b908d6b`
- Git清单：`configs/task009b/pose_library_manifest_v1.json`，包含分组、接受种子、拒绝统计、
  固定回载ID、外部绝对路径和哈希。

## 未执行门禁

- Gate 4 七十毫米面积加权覆盖计算：正在实施；已确认旧实现仍是50 mm与顶点计数比例，
  不会将其误报为符合冻结合同。
- Gate 5 三视图现场验收：未执行；依赖覆盖计算通过。

本报告未把未执行项描述为通过；位姿库数据本体按合同仅保存在Linux外部工件目录。
