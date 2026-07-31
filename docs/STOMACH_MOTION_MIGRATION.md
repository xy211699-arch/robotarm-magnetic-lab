# 平桌磁控系统向胃部任务的迁移报告

## 范围

本次仅迁移已在平桌场景验收的开环控制系统，不引入胶囊反馈，也不补偿胃壁褶皱、
曲率或接触误差。平桌基准文件与参数保留，胃部运动使用独立启动入口和日志目录。

## 场景与接口

- Isaac Lab任务：`Template-Robotarm-Magnetic-Stomach-Lab-v0`
- 胃部姿态：沿用当前水平放置的 `stomach_environment_lab.usda`
- 胶囊初始中心：胃部下端 `(1.0608155, 0.1145374, 0.0065) m`
- 动作：九维绝对关节动作，`j1..j6 + ballxj/ballyj/ballzj`
- 控制频率：策略/动作20 Hz，物理240 Hz
- 保留物理：重力、胃壁碰撞、接触传感器、有限尺寸磁力/磁矩、胃液角阻尼
- 胶囊仍是完全被动刚体，不直接写入速度或施加非磁性主动驱动

## 已迁移功能

1. 固定倾角并改变方位；
2. 直立与侧躺之间的连续姿态转换；
3. Ball维持轴向磁场、机械臂产生场梯度的纵轴被动滚动；
4. 连续组合：侧躺 → 直立 → 30°倾斜整圈 → 直立 → 侧躺 →
   世界+X方向滚动参考。

四项功能复用 `controllers/table_motion.py` 中的有限场逆解与轨迹平滑器。
组合滚动继续锁定j6，使ASM保持已验证的避碰安装角，只让j1..j5产生磁场源平移。

## 验证结果

模式1无窗口测试：

- 方位变化：76.67°（请求80°）
- 最终轴向误差：5.79°
- 接触率：100%
- 磁场逆解最大方向误差：0.033°
- 结果：PASS

组合测试：

- 初始侧躺倾角：88.15°
- 第一次直立倾角：2.84°
- 旋转阶段中位倾角：29.40°
- 方位累计旋转：361.25°
- 第二次直立倾角：2.55°
- 最终侧躺倾角：83.09°
- 胃壁接触率：100%
- ASM最小间隙：9.83 mm
- 无碰撞、无提前终止
- 最终+X位移：11.29 mm（请求100 mm）

因此，姿态与磁场控制链路已完成迁移；胃内100 mm定量滚动尚未验收。按本阶段要求，
胃壁导致的位移误差不反馈修正，日志仍如实保留失败判据。

## 文件与运行

- `scripts/stomach_motion/stomach_test_common.py`
- `scripts/stomach_motion/test_01_tilt_azimuth.py`
- `scripts/stomach_motion/test_02_posture_transition.py`
- `scripts/stomach_motion/test_03_long_axis_roll.py`
- `scripts/stomach_motion/test_04_composite_motion.py`
- `scripts/stomach_motion/README.md`

可视化组合测试：

```bash
./run_isaaclab.sh -p scripts/stomach_motion/test_04_composite_motion.py
```

加胶囊相机窗口：

```bash
./run_isaaclab.sh -p scripts/stomach_motion/test_04_composite_motion.py \
  --capsule_camera_view
```

验证日志：

- `logs/stomach_motion/tilt_azimuth/20260731_014520/summary.json`
- `logs/stomach_motion/composite_motion/20260731_014631/summary.json`
