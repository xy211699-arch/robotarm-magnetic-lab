# 十一动作动态控制器

## 当前状态

TASK-005 已实现公开动作 ID `0..10`、一秒状态机、动态刚体 COM wrench 适配、实时接触读取、
平面任务和键盘入口。确定性标定通过，但独立随机平面验收未通过，因此当前 disposition 为
`needs_decision`，不得用于声明动作层已经验收完成。

## 动作接口

| ID | 动作 | 时序 |
|---:|---|---|
| 0 | `HOLD_VIEW` | 240 子步闭环保持 |
| 1–8 | 相机九宫格外围 VIEW | 192 子步 quintic swing + 48 子步保持 |
| 9 | `MOVE_SIDE_POS` | 60 子步自由 + 120 子步 COM 力 + 60 子步自由 |
| 10 | `MOVE_SIDE_NEG` | 同上，方向相反 |

VIEW 相对动作开始时冻结的相机坐标系产生 15° 光轴目标。MOVE 仅在开始时检查倾角至少 60°
且最近 12 个物理子步存在侧壁接触；前置失败仍运行完整一秒 HOLD 并返回 `REJECTED`。

运行期胶囊保持启用重力的非运动学动态刚体。控制器只向胶囊施加世界系 COM force/torque，
不写入 pose、orientation 或速度。MOVE torque 始终为零。VIEW/HOLD 使用冻结支撑点切向闭环、
光轴 swing 闭环和任务授权的实时接触力矩补偿，不规划长轴 twist。

## 入口

平面键盘可视化：

```bash
./run_isaaclab.sh -p scripts/eleven_action/teleop_eleven_action.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 \
  --device cuda:0 --render_fps 120 --capsule_camera_view
```

数字小键盘 `7/8/9, 4/5/6, 1/2/3` 对应九宫格，`5` 为 HOLD，`Q` 为负向 MOVE，`E` 为正向
MOVE。动作执行期间的新按键被丢弃，不缓存、不排队、不抢占。

确定性标定：

```bash
./run_isaaclab.sh -p scripts/eleven_action/calibrate_eleven_action.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 \
  --device cuda:0 --seed 42 \
  --write_profile configs/eleven_action/dynamic_profile.json --headless
```

正式随机平面门禁：

```bash
./run_isaaclab.sh -p scripts/eleven_action/validate_eleven_action_flat.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 \
  --device cuda:0 --seed 20260818 --render_fps 120 --headless
```

当前正式门禁输出 `ELEVEN_ACTION_FLAT_ACCEPTANCE_FAIL`。详细数据和停止原因见
`handoffs/reports/TASK-005-eleven-action-dynamic-controller-report.md`。

## TASK-006 Hybrid Latch 状态

TASK-006 已实现锁存数据合同、纯控制生命周期和 CUDA Dynamic Lock 探针，但首选
`dynamic_lock_flags` 后端未通过阻断门：USD mask 可读回 `0b111`，GPU PhysX 中仍出现
最大 7.328 mm 位置漂移、66.77°光轴漂移，配对释放差异也超出 0.5 mm/1°门限。

按用户补充要求，本轮没有启用 kinematic 备用方案，后续 action term、1 Hz RGB barrier、
平面/胃部验收均未执行。当前 disposition 为 `needs_decision`，详见
`handoffs/reports/TASK-006-hybrid-latched-eleven-action-controller-report.md`。
