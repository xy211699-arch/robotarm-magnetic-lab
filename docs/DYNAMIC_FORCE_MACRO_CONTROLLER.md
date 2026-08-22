# TASK-008 六动作动态力控制器

## 1. 适用范围

本控制器在 TASK-003 的真实 Dynamic 胶囊刚体路径上工作。它保留重力、接触、CPU PhysX、
刚体 CCD 和场景 CCD，不在动作执行期间写入胶囊位姿或速度。控制器不使用磁场模型、关节
控制、理想表面投影或运动学锁定。

Actor 每秒提交一个标量动作 ID，执行器同步推进一秒仿真并返回边界 RGB。策略观测仅包含
RGB；胶囊真值、力、轨迹和覆盖率仅供标定与评估使用。

## 2. 动作 ID 与物理定义

| ID | 名称 | 作用 |
|---:|---|---|
| 0 | `HOLD` | 全程零主动力、零主动力矩；Dynamic 胶囊仍可受重力和接触力运动 |
| 1 | `MOVE_POS` | 两端各施加同向 `+d` 力，产生平移合力 |
| 2 | `MOVE_NEG` | 两端各施加同向 `-d` 力，产生反向平移合力 |
| 3 | `VIEW_POS` | 只在相机端施加 `+d` 力，产生相机光轴正向偏转力矩 |
| 4 | `VIEW_NEG` | 只在相机端施加 `-d` 力，产生反向偏转力矩 |
| 5 | `UP` | 相机端向上、另一端向下施加等量反向力，形成只抬相机端的纯力偶 |

胶囊半径为 `0.0065 m`，圆柱段高度为 `0.012 m`，总长为 `0.025 m`。相机侧位于胶囊
局部 `-Z`，两个半球中心由圆柱段半高得到：`z=-0.006 m` 和 `z=+0.006 m`。

令相机指向的胶囊长轴为 `u_cam`，世界上方向为 `z_w`，横向单位方向为：

```text
d = normalize(z_w × u_cam)
-d = -(+d)
```

若该叉积不可归一化，控制器直接报告数值契约错误，不猜测替代方向。

对于质量 `m`、重力加速度 `g` 和比例 `r`：

- MOVE：两端分别为 `0.5*r_move*m*g*d`；
- VIEW：相机端为 `r_view*m*g*d`；
- UP：令 `d_up = z_w - (z_w·u_cam)u_cam`，相机端为
  `+0.5*r_up*m*g*d_up`，另一端为其反向力。两端合力为零，力矩与旧单点力等价，
  但非相机端会被明确压向支撑面；相机端已朝正上时力偶自然归零，恰好朝下时采用确定性
  横向方向脱离不稳定点。

多个端点力通过等价质心力/力矩一次性提交到 PhysX：

```text
F_com = Σ F_i
τ_com = Σ ((p_i - p_com) × F_i)
```

## 3. 时间协议

- 物理：240 Hz；
- 环境和渲染：60 Hz；
- 胶囊相机和覆盖评估：30 Hz；
- Actor：1 Hz；
- 每个动作严格为 240 个物理子步、60 个环境步、1 秒仿真时间。

MOVE/VIEW 的 `0..47` 子步等待、`48..191` 子步施力、`192..239` 子步等待。UP 在
`0..239` 全程施力；HOLD 全程为零。边界 RGB 必须在第 240 子步结束后、清除 UP 力之前
采集。推理墙钟时间不计入一秒仿真动作。

## 4. 平面标定与验收

启动前置检查：

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab
./run_isaaclab.sh -p scripts/dynamic_force_macro/inspect_prerequisites.py \
  --task Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0 \
  --headless --output /tmp/task008-preflight.json
```

执行完整标定与一次性留出集验收：

```bash
./run_isaaclab.sh -p scripts/dynamic_force_macro/calibrate_validate_table.py \
  --task Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0 \
  --calibration_samples 20 --held_out_samples 20 \
  --initial_ratio 0.9 --growth 1.25 --max_ratio 3.0 \
  --refinement_rounds 3 --headless \
  --output_dir /tmp/task008-dynamic-force-calibration
```

标定集与留出集使用不相交种子。每个候选值为每个非 HOLD 动作运行 20 次；MOVE 正负
共享一个比例，VIEW 正负共享一个比例，UP 单独标定。候选从 `0.9` 开始乘 `1.25`，上限
为 `3.0`；首次通过后在最近失败值和首次通过值之间做三次确定性二分。每个动作至少
`16/20` 成功且没有 FAULT 才通过。

成功阈值为：MOVE 沿施力方向净位移不少于 `5 mm`；VIEW 命令平面内有符号转角不少于
`15 deg`；UP 相机侧仰角不少于 `45 deg` 且过程中不越过世界竖直方向。

标定输出全部保存在 `/tmp` 外部证据目录，不提交 Git。胃部场景只加载冻结后的
`selected_profile.json`，不得重新搜索或静默调整比例。

## 5. 胃部三视图验收

```bash
./run_isaaclab.sh -p scripts/dynamic_force_macro/teleop_stomach.py \
  --task Template-Robotarm-Magnetic-Dynamic-Force-Macro-Stomach-Lab-v0 \
  --move_force_ratio 0.40 --view_force_ratio 0.25 --up_force_ratio 0.85 \
  --viz kit
```

按键：`Space=HOLD`、`D/A=MOVE_POS/MOVE_NEG`、`E/Q=VIEW_POS/VIEW_NEG`、`W=UP`、
`Backspace=胶囊与覆盖率一起复位`、`F12=快照`、`Esc=退出`。按键按下沿只触发一次动作，
操作系统按键重复不会重启宏动作。动作结束后暂停物理，只刷新 Kit 界面，直到下一次按键。

界面显示默认外部视口、30 Hz 胶囊 RGB和独立累计覆盖窗口；不再创建中文动作/力度状态
窗口。覆盖率使用 120°圆形 FOV、50 mm 距离、首次命中遮挡和相机朝向三角面法线
门控；该门控仅由 TASK-008 显式开启，旧覆盖率默认行为不变。

脚本化六动作冒烟：

```bash
./run_isaaclab.sh -p scripts/dynamic_force_macro/teleop_stomach.py \
  --task Template-Robotarm-Magnetic-Dynamic-Force-Macro-Stomach-Lab-v0 \
  --move_force_ratio 0.40 --view_force_ratio 0.25 --up_force_ratio 0.85 \
  --scripted_actions 0,1,2,3,4,5 --max_actions 6 --viz kit
```

## 6. 输出和限制

标定输出包含两个种子清单、全部候选逐试验记录、冻结配置及 SHA-256、留出集逐动作结果。
胃部输出包含 `macro_actions.jsonl`、30 Hz `frames.jsonl`、边界 RGB、累计掩码、轨迹、
快照、元数据和文件清单。

FAULT 仅表示非有限状态、求解中断、完整穿越平面或逃离测试区域；位移/角度不足、弹跳、
滑动和普通接触切换均为普通失败。胃部只做定性迁移，不声明成功率。当前主机若出现
`NVML_ERROR_LIB_RM_VERSION_MISMATCH`，RTX 相机和 CUDA Warp 覆盖评估无法完成，必须先
恢复一致的 NVIDIA 内核模块与用户态库，再执行上述 live 标定和三视图验收。
