# 胃部场景磁控运动验收

本目录把已经在平桌场景验收的开环控制器迁移到水平放置的胃部任务中。
控制器、磁场逆解和机械臂梯度运动保持不变；场景切换为
`Template-Robotarm-Magnetic-Stomach-Lab-v0`，并保留胃部任务配置的流体
角阻尼、重力、磁力和胃壁接触。

胶囊每次从胃部下端的既有重置点
`(1.0608155, 0.1145374, 0.0065) m` 开始。三个基本功能和组合运动分别是：

1. `test_01_tilt_azimuth.py`：保持倾角并改变方位；
2. `test_02_posture_transition.py`：直立到侧躺的平滑姿态转换（同一磁场轨迹反向
   使用即可实现侧躺到直立）；
3. `test_03_long_axis_roll.py`：保持轴向磁场并用机械臂产生磁场梯度，使胶囊被动
   绕纵轴滚动；
4. `test_04_composite_motion.py`：侧躺 → 直立 → 30° 倾斜方位旋转一圈 →
   直立 → 侧躺 → 沿世界 +X 方向滚动约 100 mm。

从项目根目录启动可视化：

```bash
./run_isaaclab.sh -p scripts/stomach_motion/test_01_tilt_azimuth.py
./run_isaaclab.sh -p scripts/stomach_motion/test_02_posture_transition.py
./run_isaaclab.sh -p scripts/stomach_motion/test_03_long_axis_roll.py
./run_isaaclab.sh -p scripts/stomach_motion/test_04_composite_motion.py
```

胶囊相机窗口可选：

```bash
./run_isaaclab.sh -p scripts/stomach_motion/test_04_composite_motion.py \
  --capsule_camera_view
```

无窗口验收：

```bash
./run_isaaclab.sh -p scripts/stomach_motion/test_04_composite_motion.py \
  --visualizer none --no-realtime
```

每轮输出写入
`logs/stomach_motion/<scenario>/<timestamp>/telemetry.jsonl` 和
`summary.json`。当前目标是验证控制链路迁移和视觉效果，不把胃部褶皱导致的
位姿/距离误差作为控制失败。

首次无窗口迁移验证中，模式1全部判据通过；组合运动的五个姿态阶段也全部通过，
但胃壁上的最后滚动段只沿世界 +X 移动约 11.3 mm，没有复现平桌的 100 mm。
这份结果保留为真实基线，不通过放宽阈值伪装成定量验收成功。当前脚本可用于观察
控制链路；若以后要求胃内精确位移，需要针对局部曲面、摩擦和磁场源路径重新规划。
