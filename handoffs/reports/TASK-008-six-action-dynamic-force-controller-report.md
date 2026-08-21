# TASK-008 Linux 执行报告

## 1. 结论

**Disposition：`partial`。**

六动作动态力接口、240 Hz 动作项、同步 1 秒边界执行器、平面标定程序、法线感知覆盖率和
胃部三视图入口均已实现；全仓库 163 项回归通过。真实平面标定和胃部三视图验收未能运行，
原因是宿主 NVIDIA 内核模块与用户态库版本不匹配，RTX 与 CUDA Warp 均无法初始化。
因此没有选择力比例、没有运行留出集，也没有声称动作物理验收通过。

## 2. Git 与范围

- 规划分支：`origin/workflow/TASK-008-six-action-dynamic-force`
- 规划提交：`a232ac4d060efe17669ecfe8341331ed88fa8555`
- 固定基线：`06b15caf9a69bc9c20f85522ce4abbb32c8b9245`
- 实施分支：`feature/TASK-008-six-action-dynamic-force-controller`
- 最终代码实现头（不含本报告后续文字提交）：`bfc7ae4eb9e61ce4e100e2baac2fb44cafcb22f5`
- 未修改 USD、TASK-003 原控制器、磁场、理想表面、VLM/RL、latch、virtual-magnet 或既有实验报告。

实施提交：

| 提交 | 内容 |
|---|---|
| `31ef55a` | 冻结六动作、阶段、端点力、等价质心力矩和验收公式 |
| `41985f7` | 240 Hz 动作项与同步 60 步/1 秒执行器 |
| `896e917` | 实时几何/等价力矩路径前置检查及任务配置测试 |
| `76a6816` | 确定性平面标定和留出集框架 |
| `63cc6c8` | USD 朝向敏感的相机朝向法线覆盖门控 |
| `c352c19` | 修正 Isaac Lab `wxyz`/SciPy `xyzw`，加固 settle、重采样和 live 检查 |
| `f6b16f0` | 单按键宏动作及胃部三视图验收入口 |
| `bfc7ae4` | 补齐派生平面测试区域逃逸与完整穿越 FAULT 分类 |

## 3. 实现结果

### 控制契约

- ID 严格为 `HOLD=0, MOVE_POS=1, MOVE_NEG=2, VIEW_POS=3, VIEW_NEG=4, UP=5`。
- 时钟严格为物理 240 Hz、环境/渲染 60 Hz、相机/覆盖率 30 Hz、Actor 1 Hz。
- 每次调用推进 60 个环境步/240 个物理子步；UP 边界图像先采集再清力。
- 端点几何使用 13 mm 直径、12 mm 圆柱段、25 mm 总长，相机侧为局部 `-Z`。
- 已选择等价 COM wrench 路径；纯测试验证合力和 `Σ(r×F)` 与多点力定义一致。
- 新运行时代码禁止位姿/速度写入；只有随机试验 reset 使用允许的 reset writer。

### 标定协议

- 标定/留出清单种子分离；无效 settle setup 确定性重采样且不消耗 20 个有效槽位。
- settle 只运行 240 子步零主动力 HOLD，并检查有限状态、接触和速度稳定。
- 候选序列为 `0.9, ×1.25, ..., 3.0`；首次通过后严格三次中点细化。
- MOVE 正负共享比例、VIEW 正负共享比例、UP 独立；默认门槛为每动作 `16/20` 且零 FAULT。
- 未因 live 阻塞而生成任何候选结果、配置摘要或留出集结果。

### 覆盖率与胃部入口

- `MeshInput/ReferenceMesh` 保存 USD `rightHanded/leftHanded` 朝向并纳入几何哈希。
- 法线门控要求首次命中面法线与相机到命中点射线点积严格小于负容差。
- 旧 P0 调用默认 `require_camera_facing_normal=False`；TASK-008 显式启用。
- CPU PhysX 与 `cuda:0` Warp 光线设备解耦。
- 胃部按键映射和重复抑制已实现；动作间只刷新 Kit UI，不推进物理。
- 外部视口、胶囊 RGB、覆盖窗口和状态面板代码已接入，但因 RTX 驱动阻塞未完成运行验收。

## 4. 验证证据

### 已通过

```text
./run_isaaclab.sh -p -m pytest -q -p no:cacheprovider tests --disable-warnings
163 passed, 67 warnings

./run_isaaclab.sh -p scripts/action_layer/validate_pure.py
ACTION_LAYER_PURE_TESTS total=18 failed=0

./run_isaaclab.sh -p -m compileall -q \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab scripts/dynamic_force_macro
exit=0

git diff --check
exit=0
```

专项测试另外得到：TASK-008 + coverage `49 passed`；此前阶段性结果为 `18/18`、`44/44`、
`48/48`，最终全仓库结果覆盖并超过这些集合。

禁止调用扫描在新增运行文件中未发现：`write_root_pose`、`write_root_velocity`、
`set_transforms`、`set_velocities`。策略配置仅暴露 RGB，不包含覆盖率、胶囊位姿或接触真值。

### 未通过/未执行

1. live 前置检查进入 CPU PhysX 场景并打印 `dt=1/240`、环境 `1/60`，但 RTX 相机初始化
   长时间无进展；安全终止，没有 `TASK008_PREFLIGHT_PASS`，没有 preflight JSON。
2. `validate_coverage_geometry.py --check all` 在创建 `cuda:0` Warp mesh 时失败：
   `Invalid device identifier: cuda:0`。
3. 完整平面候选搜索、一次性留出集、冻结 profile 和胃部六动作三视图均未执行，因为它们
   依赖当前不可用的 RTX/CUDA。未采用 CPU 覆盖率替代、kinematic 替代或伪造 profile。

主机诊断：

```text
nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 595.84

/proc/driver/nvidia/version
NVRM version: ... 595.71.05
```

## 5. 外部证据

| 绝对路径 | 字节 | SHA-256 | 内容 |
|---|---:|---|---|
| `/mnt/isaac-linux/isaacsim/kit/logs/Kit/IsaacLab/3.0/kit_20260821_114530.log` | 461058 | `b0e2ff4b95ba721a19efed76042bd69be13c8b2fab3d5a2e1dcbbbc910f1e0c2` | 首次包路径错误及驱动诊断 |
| `/mnt/isaac-linux/isaacsim/kit/logs/Kit/IsaacLab/3.0/kit_20260821_114609.log` | 467404 | `92b00390548c0f346f24667ee109d5d873f14bc3d409790fa2cb7cd9d0c32e07` | 修复包路径后 CPU PhysX 启动及 RTX 阻塞 |

没有生成标定清单、profile、图像、视频或数据集；因此没有把生成物提交到 Git。

## 6. FAULT 分类与未验证声明

实现仅将非有限状态、仿真中断、完整穿越平面或逃离派生测试区域归为 FAULT；弹跳、普通
接触切换、滑动、位移不足和角度不足为普通失败。由于 campaign 未启动，FAULT 计数为
**未测量**，不是零。

以下项目保持未验证：五个非 HOLD 动作的 `success/20`、选中比例及 SHA-256、候选搜索是否
能在 `3.0mg` 内通过、胃部运动主观有效性、三窗口实时同步效果和 30 Hz GPU 覆盖耗时。

## 7. 驱动恢复后的严格续跑顺序

1. 使 NVIDIA 内核模块与 `libnvidia-ml/libcuda` 版本一致，并重启后确认 `nvidia-smi` 正常；
2. 执行 live preflight，必须出现 `TASK008_PREFLIGHT_PASS`；
3. 执行 CUDA 覆盖几何验证；
4. 运行完整标定和一次性留出集，不得观察留出结果后回调标定；
5. 只有所有五动作达到 `16/20`、零 FAULT，才运行冻结 profile 的胃部脚本化六动作；
6. 三视图入口运行成功后仍将主观有用性留给 Windows 人工验收。
