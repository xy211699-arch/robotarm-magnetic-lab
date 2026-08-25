# TASK-009B Linux 执行报告

状态：`needs_input`

停止门禁：Gate 2——胃入口三维包围盒等待现场标定与用户确认。

## 基线与实现

- 规划分支：`workflow/TASK-009B-stomach-coverage-environment`
- 规划提交：`c57ce69873cecd7c21db05f5e656bf4f77b4b626`
- TASK-009A 控制器基线：`335c5f563da51c50656729db86a7872809c58ada`
- 实现分支：`feature/TASK-009B-stomach-coverage-environment`
- Gate 1 提交：`3de692e`
- Gate 2 提交：`c2205c5cd73ee766c6ce32735cbf2c762fdf1dae`

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

## Gate 2：入口三维包围盒

状态：`needs_input`

已实现：

- 世界轴对齐半透明三维包围盒；
- 沿世界X/Y/Z平移中心并分别调整尺寸；
- 精确三角形/AABB分离轴相交筛选，而非质心近似；
- 候选面片双侧高亮；
- 面片数量、面积和共享完整边连通分量实时统计；
- 多连通区域警告；
- 版本化JSON配置、胃壁几何SHA-256、配置SHA-256和保存摘要；
- 独立标定模式不推进物理、不执行胶囊动作。

现场首次启动发现Kit 110的`UsdGeom.Mesh`绑定没有显示Primvar插值便捷方法，并且Hydra会
读取拓扑先于点缓冲写入的瞬时无效状态。修复后统一通过`UsdGeom.PrimvarsAPI`设置插值，
并在写入拓扑前创建合法点缓冲；没有改变包围盒选择、统计或保存语义。

自动化结果：相关回归`19 passed`。纯几何测试覆盖边穿越包围盒但无顶点位于盒内、面积、
连通分量和配置失效哈希。

当前执行工具容器没有暴露NVIDIA设备与图形显示，因此不能代替本机完成现场画面验收。
本机标定命令：

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/calibrate_entrance_region.py \
  --device cuda:0 \
  --viz kit
```

按键及验收方法见`docs/TASK009B_ENTRANCE_CALIBRATION.md`。用户应调整至橙红色高亮只覆盖
预期入口表面，尽量保持连通分量为1，按`S`保存，然后回复
`TASK009B_ENTRANCE_SAVED`终端行并明确确认高亮区域正确。

## 未执行门禁

- Gate 3 有效初始位姿库：未执行；入口未确认前禁止生成。
- Gate 4 七十毫米面积加权覆盖计算：未执行；依赖已验收位姿库。
- Gate 5 三视图现场验收：未执行；依赖覆盖计算通过。

本报告未把未执行项描述为通过，也未创建默认入口配置或最终位姿库。
