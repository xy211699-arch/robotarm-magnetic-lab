# TASK-009B Linux 执行报告

状态：`needs_input`

停止门禁：Gate 2——稳定锚点与胃壁测地区域等待现场双确认。

## 基线与实现

- 规划分支：`workflow/TASK-009B-stomach-coverage-environment`
- 规划提交：`c57ce69873cecd7c21db05f5e656bf4f77b4b626`
- TASK-009A 控制器基线：`335c5f563da51c50656729db86a7872809c58ada`
- 实现分支：`feature/TASK-009B-stomach-coverage-environment`
- Gate 1 提交：`3de692e`
- Gate 2 替代方案提交：`3fa7405ccb7d528a4f5b96d50154a32961e75abf`
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

状态：`needs_input`

用户新合同已完全取消三维包围盒。替代实现包括：

- 从当前合理默认位姿开始，胶囊保持Dynamic，暂停PhysX后用WASD只修改世界X/Y；
- 普通/Shift精细速度固定10/2 mm/s，移动期间持续核验仿真时间不推进；
- Enter前清除Actor力和速度，随后只由重力、胃壁碰撞及摩擦自然下落；
- 按240 Hz逐步检查，连续0.25秒满足2 mm/s及5 deg/s才报告稳定，2秒超时自动恢复；
- Y确认稳定锚点，Backspace拒绝并恢复释放前状态；
- 通过点到三角形真实最近点确定种子，未使用最近顶点近似；
- 通过胃壁三角形共享边邻接图和Dijkstra距离生成10/15/20/25/30 mm测地区域；
- 高亮最近表面点和当前区域，区域仅一个连通分量时允许Enter保存；
- 锚点与区域配置绑定胃壁哈希并相互绑定，保存后执行精确重载校验。

自动化结果：`22 passed`；纯几何、哈希、接口源合同与既有控制器回归通过。现场动态下落、按键确认和
区域画面仍必须由本机图形会话验收。

当前执行工具容器没有暴露NVIDIA设备与图形显示，因此不能代替本机完成现场画面验收。
本机标定命令：

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/calibrate_entry_anchor_region.py \
  --device cuda:0 \
  --viz kit
```

按键及验收方法见`docs/TASK009B_ENTRY_ANCHOR_REGION_CALIBRATION.md`。用户应先用WASD
选择释放位置并确认稳定锚点，再切换测地半径并确认单连通入口区域。将终端中的
`TASK009B_ENTRY_ANCHOR_SAVED`与`TASK009B_ENTRY_REGION_SAVED`完整行交给执行端。

## 未执行门禁

- Gate 3 有效初始位姿库：未执行；入口未确认前禁止生成。
- Gate 4 七十毫米面积加权覆盖计算：未执行；依赖已验收位姿库。
- Gate 5 三视图现场验收：未执行；依赖覆盖计算通过。

本报告未把未执行项描述为通过，也未创建默认入口配置或最终位姿库。
