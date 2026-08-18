# TASK-004 局部动力学动作原语

## 当前交付状态

TASK-004 当前为 **partial**。平面任务、纯控制器、COM 世界系力/力矩动作和定量验证器已经实现；
但平面验收门禁未通过，因此按照任务合同，胃部任务包装器和共用可视化启动器没有创建。
这不是可直接用于胃部实验的已验收控制器。

## 已实现接口

平面任务 ID：

```text
Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0
```

动作是四个浮点数：

```text
[start_pulse, primitive_code, direction_x, direction_y]
```

- `start_pulse` 仅在从低到高的边沿提交一次请求；后续持续发送 `0`。
- `(direction_x, direction_y)` 只表示世界 XY 方位，零向量默认世界 `+X`。
- 控制器运行期间再次提交会返回 `busy`，成功后进入闭环保持，可接受下一条合法动作。

动作编号：

| 编号 | 名称 | 起始姿态门限 | 目标 |
|---:|---|---|---|
| 0 | `SIDE_TO_UPRIGHT` | 倾角 75°–105° | 世界 `+Z` 直立 |
| 1 | `UPRIGHT_TO_SIDE` | 倾角不大于 5° | 指定方位侧躺 |
| 2 | `UPRIGHT_TO_30_DEG` | 倾角不大于 5° | 指定方位、相对 `+Z` 倾斜 30° |
| 3 | `CONE_30_DEG_ONE_REVOLUTION` | 30°±3° | 保持 30°并完成一周锥面旋转 |

胶囊相机位于局部 `-Z` 端，定向轴从非相机端指向相机端：

```text
u = R(q) [0, 0, -1]
```

位置和速度读取质心状态；姿态轴使用刚体 link 坐标系，避免 PhysX 主惯性 COM 坐标系旋转
污染胶囊几何轴。运行期只通过 `permanent_wrench_composer` 在质心施加世界系力与力矩，
没有根位姿/速度写入、磁体控制、表面法向估计、净空查询、避碰、投影或穿透修复。

## 当前共享参数

当前保留的是已验证尝试中姿态变化最大的合法参数组，但它仍未通过门禁：

```text
axis_kp_nm_per_rad                 = 3.0e-5
axis_kd_nms_per_rad                = 8.0e-6
roll_damping_nms_per_rad           = 1.0e-6
torque_limit_nm                    = 3.0e-5
anchor_kp_n_per_m                  = 3.0
anchor_kd_ns_per_m                 = 0.15
horizontal_force_limit_weight_ratio = 1.0
downward_preload_weight_ratio      = 0.15
motion_duration_s                  = (5.5, 4.5, 3.5, 8.0)
hard_timeout_s                     = (8.0, 7.0, 6.0, 9.5)
```

## 验证命令

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab

./run_isaaclab.sh -p -m pytest tests/local_primitives -q --disable-warnings

./run_isaaclab.sh -p \
  scripts/local_primitives/inspect_local_primitives_prerequisites.py \
  --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 \
  --headless

./run_isaaclab.sh -p \
  scripts/local_primitives/validate_local_primitives_flat.py \
  --task Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0 \
  --seed 42 \
  --direction_azimuth_deg 0 \
  --headless \
  --output_directory /mnt/isaac-linux/robotarm_magnetic_lab/logs/local_primitives_flat
```

验证器依次执行 `0`、`0→1`、`0→2`、`0→2→3` 四组序列，并将逐帧 JSONL 与
汇总 JSON 写到 Git 仓库外。当前预期结果是明确的 `LOCAL_PRIMITIVES_FLAT_VALIDATION_FAIL`，
不是 PASS。

## 下一步决策

现有合同给定的轴向增益上限和力矩上限不足以在重力与平面接触下把胶囊从侧躺抬起。
Windows 方案端需要先批准新的动力学权限边界，例如提高允许力矩/重新定义力矩前馈；批准前
不得迁移胃部任务，也不得通过改质量、摩擦、重置姿态或运行期传送伪造通过。
