# TASK-004 局部动力学动作原语

## 交付状态

四项动作已在隔离平面任务中通过定量门禁，并以完全相同的冻结 profile 迁移到 TASK-003
胃部场景。胃部渲染中，起身、30°倾斜和30°圆锥一周成功；直立到侧躺受局部胃壁接触
阻塞并在7 s超时。因此本任务按合同标记为 **partial**，不对胃部主观可用性作保证。

本控制器是仿真专用控制器，不代表真实磁体、磁矩、执行器或硬件可实现同等扳手。
胶囊始终是非运动学动态刚体，运行期只施加世界系质心力/力矩并由 PhysX 积分；没有
位姿/速度写入、传送、表面投影、净空查询、避碰或胃部专用恢复。

## 任务与接口

任务 ID：

```text
Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0
Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0
```

动作是四个浮点数：

```text
[start_pulse, primitive_code, direction_x, direction_y]
```

- `start_pulse` 只在低到高边沿提交一次；原语运行期间持续发送0。
- `(direction_x, direction_y)` 表示世界 XY 方位，零向量默认世界 `+X`。
- 胶囊相机位于局部 `-Z` 端，定向轴 `u = R(q)[0,0,-1]` 从非相机端指向相机端。
- 控制器读取质心位置/速度和刚体 link 姿态，输出一个等效世界系质心 wrench。

| 编号/按键 | 原语 | 起始条件 | 目标 |
|---:|---|---|---|
| 0 / `1` | `SIDE_TO_UPRIGHT` | 倾角75°–105° | 世界 `+Z` 直立 |
| 1 / `2` | `UPRIGHT_TO_SIDE` | 倾角≤5° | 指定方位侧躺 |
| 2 / `3` | `UPRIGHT_TO_30_DEG` | 倾角≤5° | 指定方位倾斜30° |
| 3 / `4` | `CONE_30_DEG_ONE_REVOLUTION` | 30°±3° | 保持30°完成一周圆锥运动 |

## 控制机制

控制器在原语开始时记录非相机虚拟端点的世界 XY 锚点。对胶囊半长 `h=0.0125 m`、
定向轴 `u`：

```text
r_nc = -h u
p_nc = p_com + r_nc
v_nc = v_com + omega × r_nc
F_nc_xy = Kp(anchor_xy - p_nc_xy) - Kd v_nc_xy
F_nc_z = -F_pin
tau_total = tau_pose + r_nc × F_nc
```

端点力和姿态力矩组合后被总力/总力矩限幅及向量 slew 限制，再通过
`permanent_wrench_composer` 以 `positions=None, is_global=True` 施加。端点向下力只用于偏置
非相机端支撑，不是运动学约束。

## 冻结仿真 profile

文件：`configs/local_primitives/simulation_profile.json`

规范化JSON profile digest（SHA-256；不含文件缩进差异）：

```text
d82bf6d381e99d7be07cdf614223139fd8353c56011b8dc0a2d9779555bdcc72
```

关键值：

```text
axis_kp_nm_per_rad          = 0.02
axis_kd_nms_per_rad         = 0.0016
roll_damping_nms_per_rad    = 0.0016
pose_torque_limit_nm        = 0.02
anchor_kp_n_per_m           = 10.0
anchor_kd_ns_per_m          = 0.4
endpoint_pin_force_n        = 0.1
total_force_limit_n         = 1.25
total_torque_limit_nm       = 0.02
force_slew_limit_n_per_s    = 50.0
torque_slew_limit_nm_per_s  = 0.2
motion_duration_s           = (5.5, 4.5, 3.5, 8.0)
hard_timeout_s              = (8.0, 7.0, 6.0, 9.5)
```

平面和胃部任务都通过 `make_local_primitive_action_cfg()` 嵌入同一内容和 digest；胃部 wrapper
只继承 TASK-003 的场景、重置、CCD、接触、相机和时序，不含任何胃部控制适配。

## 定量结果

平面 gate：

| 原语 | 状态 | 完成时间 | 最大总力 | 最大总力矩 | 最大240 Hz单步位移 |
|---|---|---:|---:|---:|---:|
| 侧躺→直立 | 成功保持 | 5.900 s | 0.10552 N | 0.0017149 N·m | 0.2448 mm |
| 直立→侧躺 | 成功保持 | 5.121 s | 0.12626 N | 0.0012136 N·m | 0.1226 mm |
| 直立→30° | 成功保持 | 3.900 s | 0.10407 N | 0.0006743 N·m | 0.0191 mm |
| 30°圆锥一周 | 成功保持 | 8.404 s | 0.11020 N | 0.0009932 N·m | 0.0808 mm |

圆锥实际展开角为 `6.2940 rad`，倾角 RMSE 为 `0.00614 rad`。起身相机半球承载样本为0，
末期非相机端支撑成立。全部动作小于10 s、状态有限且连续性小于5 mm门限。

胃部连续渲染使用同一脚本序列：

```text
0,1;reset;0,2;reset;0,2,3
```

其中所有三次起身、两次30°和一次圆锥运动成功；直立到侧躺在胃壁接触下超时。合同禁止
针对该超时修改 profile、查询胃壁几何或加入恢复。

## 运行方法

平面定量验证：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab

./run_isaaclab.sh -p \
  scripts/local_primitives/validate_local_primitives_flat.py \
  --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 \
  --seed 42 \
  --direction_azimuth_deg 0 \
  --headless
```

平面可视化：

```bash
./run_isaaclab.sh -p \
  scripts/local_primitives/teleop_local_primitives.py \
  --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 \
  --direction_azimuth_deg 0 \
  --capsule_camera_view \
  --viz kit
```

胃部可视化：

```bash
./run_isaaclab.sh -p \
  scripts/local_primitives/teleop_local_primitives.py \
  --task Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0 \
  --direction_azimuth_deg 0 \
  --capsule_camera_view \
  --viz kit
```

操作键：`1`–`4`执行对应原语，`Backspace`复位，`F12`保存 JSON 与 PNG 快照，`Esc`退出。
启动器持续显示 Kit 外部视图；加 `--capsule_camera_view` 后显示1280×720、30 Hz胶囊相机
调试窗口。每帧遥测、分解 wrench、profile digest、终态与快照写入
`logs/local_primitives_teleop/<UTC时间>/`。

自动序列示例：

```bash
./run_isaaclab.sh -p \
  scripts/local_primitives/teleop_local_primitives.py \
  --task Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0 \
  --scripted_sequence "0,1;reset;0,2;reset;0,2,3" \
  --direction_azimuth_deg 0 \
  --capsule_camera_view \
  --viz kit
```
