# 新胃模型磁控控制器迁移验收

## 实施边界

基于人工验收通过的新胃视觉分支 feature/new-stomach-legacy-visual（基线 245149e4dde11ff437ceea50fe5ebbb7022f9f6d），在隔离工作树 /tmp/robotarm-new-stomach-control 新增任务 Template-Robotarm-Magnetic-New-Stomach-Vector-Lab-v0。旧任务和原 zero_agent 可视化入口不变。机器人、胶囊初始位姿及质量、摩擦、磁体和磁场桥参数均未改；未添加直接作用于胶囊的动作力。

## 原实现复用与新胃接线

| 层级 | 实际实现 |
| --- | --- |
| 动作与轨迹 | 原 ActuatorVectorAction 及 actuator_vector、ball_feedback、source_kinematics 等原样复制。输入为每秒一组 4×9 相对增量；首秒执行，后三秒仅预览；240 Hz 物理伺服。 |
| 安全 | 原 mesh_cover_audit、native_mesh_clearance、batched_arm_clearance、world_collision 与停止路径检查；5 mm 硬安全余量不变。新适配器仅将规划碰撞网格绑定到 /World/envs/env_0/Stomach/geometry/CollisionMesh 并核对胃 USD 哈希。 |
| 磁作用 | 原 MagneticPhysicsActionCfg/磁场桥，胶囊保持动态刚体。实测桥配置：磁力保护上限 20 N、磁矩保护上限 0.0009 Nm、滤波 0.15 s。 |
| 高层演示 | 原 RollEscapeProbe/SurfaceRollProbe 实现倾倒、贴壁滚转与多方位扫描。它在已获授权的仿真低层控制里读取胶囊真值；该真值不是部署策略的输入。 |
| 观察 | 原 DualObserver 生成外部机械臂/红色胶囊标记与胃内贴图视频。新胃覆盖 ROI 尚未重建，故画面不显示虚构的覆盖率。 |

新胃的合法细小三角边在旧 1 μm 焊接阈值下会退化，仅对新胃运行时采用 0.1 μm 阈值，并要求内向 leftHanded 碰撞网格。orientation_v1.json 只更新当前已验收 USD 的 SHA-256，旋转、缩放、位置未变。

## 运行入口

    cd /tmp/robotarm-new-stomach-control
    export ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor
    ./run_isaaclab.sh -p scripts/stomach_migration/validate_new_stomach_magnetic.py --mode hold --seconds 2 --device cuda:0 --viz none
    ./run_isaaclab.sh -p scripts/stomach_migration/validate_new_stomach_magnetic.py --mode area_scan --seconds 300 --record-video --device cuda:0 --viz none

可选模式：hold、axes、tip_roll、area_scan。可用 --viz kit 打开 Isaac Lab 主视口；两路相机画面需要播放生成的 MP4。记录的 t=0 位于初始化 1 秒零增量 HOLD 之后，该预备步不计入 --seconds。每轮生成 artifacts/new_stomach_magnetic/<UTC时间戳>/，包含 summary.json、status.json、geometry_audit.json、boundaries_1hz.jsonl、joint_state_240hz.jsonl、state_10hz.jsonl；录像模式另有 video/external.mp4、video/internal.mp4 与终帧 PNG。以终端 NEW_STOMACH_MAGNETIC_SUMMARY 定位运行目录。

## 已取得的证据

- 当前复测总计 61 项通过：新胃场景 24 项（其中包含新任务接线 4 项）、现用磁控模块 37 项。另有 4 项仅检查未迁移的旧 hierarchical_magnet_action/magnetic_chunk_action 源码，不能将其失败归为新任务失败。
- 新任务 1 秒 zero_agent 烟雾通过，动作管理器显示 36 维 magnet 与 0 维磁作用钩子。2 秒 HOLD 和 6 秒九轴扫描完成，无投影、拒绝或跟踪超差，实测关节增量非零。
- 60 秒 area_scan 目录 artifacts/new_stomach_magnetic/20260929T053851.353981Z/：60 个 1 Hz 边界、601 个 10 Hz 状态；第 50 秒进入滚转阶段。胶囊水平累计路径 62.34 mm、首尾三维净位移 9.20 mm；磁源至胶囊磁体中心距离 199.55–205.00 mm；机器人自身最小间距 18.57 mm、与胃网格最小间距 36.41 mm；零投影、零拒绝、零跟踪失败。
- 2 秒双视角短测目录 artifacts/new_stomach_magnetic/20260929T054919.010476Z/：两路各 21 帧，外部画面可见机械臂和红色胶囊标记，胃内画面可见胶囊和胃壁贴图。因胃壁可视面朝内，外部相机不显示胃外表面；该视角用于观察机械臂与胶囊相对运动，不取代胃外观审查。

## 待完成/限制

300 秒双视角长测尚在运行；结束后必须核对 300 个 1 Hz 边界、3001 个 10 Hz 状态及每路帧、72000 个物理步、源—胶囊距离、5 mm 余量和跟踪状态。60 秒结果不能替代长时安全结论。新胃专属入口和面积加权覆盖 ROI 尚未重新建立，旧胃覆盖清单指向旧网格，禁止直接复用或声称覆盖率提高。
