# A版：新胃模型机械臂—外磁体—胶囊联合控制归档

用户于2026-10-08明确选择A版上传。本分支以新胃磁控代码为主入口，保留旧版代码和历史证据，不合并其他工作目录的未提交修改。

## 版本身份与实现边界

- 基线：`e4fde7904b04bea4f6f99288b3dcd88546bad80f`。
- 归档分支：`release/new-stomach-magnetic-v1-20261008`。
- 任务：`Template-Robotarm-Magnetic-New-Stomach-Vector-Lab-v0`。
- 物理240 Hz，控制1 Hz；单次输入4×9相对增量，序列化为36个数；首秒执行，后三秒只作轨迹安全预览，不排队执行。
- 每行9维依次为末端3维平移、3维旋转增量与Ball三轴角度增量。不是9维绝对关节位置，也不是36个执行器。
- 使用已有ActuatorVectorAction、五次轨迹、关节伺服、停止路径检查和网格碰撞检查。硬安全余量5 mm；原安装方向、胶囊/磁体参数不变。
- 胶囊保持Dynamic刚体，由解析磁场模型的磁力/磁矩、重力及接触自然响应；此任务没有MOVE/VIEW/UP直接施力动作。
- 仿真高层演示允许读取胶囊真值；不宣称它是可直接部署的纯视觉控制器。

## 代码定位

以下路径中，`PKG`表示`source/robotarm_magnetic_lab/robotarm_magnetic_lab`。

| 作用 | 路径 |
| --- | --- |
| 磁控运行/双视频验收入口 | `scripts/stomach_migration/validate_new_stomach_magnetic.py` |
| 新胃任务与1 Hz动作管理器 | `PKG/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_new_stomach_vector_env_cfg.py` |
| 新胃视觉/接触场景 | `PKG/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_new_stomach_env_cfg.py` |
| 新胃碰撞网格适配/哈希核对 | `PKG/tasks/manager_based/robotarm_magnetic_lab/mdp/new_stomach_actuator_action.py` |
| 复用的执行器动作 | `PKG/tasks/manager_based/robotarm_magnetic_lab/mdp/actuator_vector_action.py` |
| 相对增量解码/IK/五次轨迹 | `PKG/tasks/manager_based/robotarm_magnetic_lab/controllers/actuator_vector.py` |
| 原网格安全模块 | 同级`native_mesh_clearance.py`、`mesh_cover_audit.py`、`batched_arm_clearance.py`、`ball_envelope.py`及`mdp/world_collision.py` |
| 磁场桥及240 Hz磁作用 | 同级`mdp/legacy_bridge.py`、`mdp/magnetic_action.py` |
| 贴壁滚转/多方向扫描 | `PKG/baselines/magnetic_roll_escape.py`、`magnetic_surface_roll.py` |
| 胃内/外双观察相机 | `PKG/ui/magnetic_observer.py` |
| 新胃物理与贴图资产 | `assets/stomach/new_stomach_v1/` |
| 已确认朝向/摆位 | `configs/new_stomach_v1/orientation_v1.json` |
| 管道不可达候选 | `PKG/coverage/new_stomach_tubes.py`、`scripts/stomach_migration/preview_new_stomach_coverage_regions.py` |
| 新胃/磁控回归测试 | `tests/stomach_migration/`、`tests/magnetic_following/` |
| 外部磁场源码与参数快照 | `dependencies/legacy_magnetic_sim/` |

## 可复用入口

在本归档分支的项目根目录运行，仍需下述外部依赖。以下命令是入口说明，本次归档没有启动这些仿真。

```bash
export ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor

# 2秒零增量磁控检查
./run_isaaclab.sh -p scripts/stomach_migration/validate_new_stomach_magnetic.py \
  --mode hold --seconds 2 --device cuda:0 --viz none

# 主视口短时观察
./run_isaaclab.sh -p scripts/stomach_migration/validate_new_stomach_magnetic.py \
  --mode area_scan --seconds 60 --device cuda:0 --viz kit

# 用户手动启动的5分钟双相机记录
./run_isaaclab.sh -p scripts/stomach_migration/validate_new_stomach_magnetic.py \
  --mode area_scan --seconds 300 --record-video --device cuda:0 --viz none
```

支持`hold`、`axes`、`tip_roll`、`area_scan`。先执行1秒初始化HOLD，再记录用户指定时长。默认输出`artifacts/new_stomach_magnetic/<UTC时间戳>/`，包括状态/摘要、1 Hz边界、240 Hz关节与10 Hz胶囊记录；录像模式输出`video/internal.mp4`、`video/external.mp4`。本次没有重新执行长时实验，不把历史未完结长测写成已通过。

## 覆盖率状态

新胃候选保留完整右侧圆柱管道和另一管口的排除：176109个不可达候选面片、252920个可达候选面片；候选排除面积27.75%。这仅排除覆盖目标，不删除胃部碰撞或渲染网格。

状态仍为`needs_input`，尚未启用正式新胃覆盖分母、入口位姿库或学习任务。旧胃TASK-009B/C/010的覆盖曲线和训练结果不能标成新胃结果。详见[候选标定说明](NEW_STOMACH_COVERAGE_TUBE_CALIBRATION.md)。

## 外部运行依赖与复现限制

本次不改现有运行路径，不把机器人资产重新打包，不自动安装或升级依赖。

1. `run_isaaclab.sh`仍调用`/mnt/isaac-linux/IsaacLab/isaaclab.sh`，使用其匹配的Isaac Sim Python；不要直接用Conda Python启动。
2. `legacy_bridge.py`仍从`/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim`加载磁场源码、配置及XRDF。为避免这些代码未在本库留档，已加入七份只读依赖快照和`snapshot_manifest.json`，没有重接运行加载路径。前六份与当前外部源逐字节一致；URDF副本统一为LF并去除一处导出器注释末尾空格，忽略注释的标准化XML与源文件一致，清单保留两份哈希。
3. 快照包括配置、有限磁体场与力/矩计算、磁感线积分、USD线显示、`default.json`、`robot.xrdf`、`robotarm.urdf`。第三方vendor未上传，当前Magpylib版本为5.2.3，NumPy/SciPy使用现有Isaac安装，不做升级。
4. 运行仍要求`/home/multirobo/Desktop/robotarm/urdf/robotarm.urdf`及原机器人USD场景。`assets/robotarm_magnetic_training.usda`引用`/home/multirobo/Desktop/sim of FF/Stage.usd`；其SHA-256为`0d5afec84d42989a7910899b4ef78369c454e34185c89052ab25b33fac4a6671`。本机USD递归审计解析出19个layer，未解析依赖0个，涉及`Desktop/asm/urdf/asm_5`、`Desktop/robotarm/urdf/robotarm_2`及`Desktop/sim of FF`。只复制Stage.usd并不足以复现，这些大型外部资产不进入本次普通Git提交。
5. 新胃视觉USD和贴图已在基线Git历史中，本次随基线发布，不重新导出、LFS化或修改。新的依赖快照不是运行时替代补丁；其他机器需自行配置上述路径并核对哈希，禁止盲目覆盖已有文件。
6. 本机的`legacy_bridge.py`日志仍写入`/mnt/isaac-linux/robotarm_magnetic_lab/logs/runtime.txt`，不随克隆目录改变。该固定地址在本次保持不变。

当前保护参数：磁力上限20 N、磁矩上限0.0009 Nm、滤波0.15 s。它们是保护截断而不是每步实际施力；演示的`--force-mn`是高层目标，不代表胶囊接收直接Actor力。

## 保留的旧版本

- 旧胃开环组合运动：`scripts/stomach_motion/test_04_composite_motion.py`。
- 旧胃直接施力训练：`scripts/stomach_coverage/train_task010.py`及相关supervisor；六模式、10 Hz，MOVE 0.70–1.40 mg、VIEW 0.20–0.50 mg、UP 0.80–1.05 mg。
- `zero_agent.py`不代表联合轨迹演示，启动新胃基础任务也不自动启用A版磁控动作。

保留这些入口是为了历史复核，不代表本次重测或统一成一种控制语义。
