# TASK-008 Linux 执行报告

## 1. 结论

**Disposition：`partial`。**

六动作动态力接口、240 Hz 动作项、同步 1 秒边界执行器、平面标定程序、法线感知覆盖率和
胃部三视图入口均已实现；全仓库 163 项回归通过。2026-08-22 重启后 NVIDIA 驱动恢复，
live preflight 与 CUDA 覆盖几何验证通过，完整平面标定也已执行。MOVE 两方向通过标定，
但 VIEW 和 UP 在授权的 `0.9mg--3.0mg` 搜索范围内未达到全部动作 `16/20`，因此没有
冻结完整可用 profile、没有消费留出集，也没有进入胃部动作验收。最终结论仍为 `partial`，
但阻塞原因已由“驱动不可用”更新为“动态动作物理标定未达门槛”。

续测还发现宏动作派生类遗漏了 TASK-003 的刚体 CCD 作者逻辑：场景 CCD 为真但胶囊 body
CCD 为假。已仅在 TASK-008 动作项中补齐该契约；修复后 preflight 同时验证 scene/body
CCD 为真，未修改资产、质量、惯量、摩擦或碰撞几何。

## 2. Git 与范围

- 规划分支：`origin/workflow/TASK-008-six-action-dynamic-force`
- 规划提交：`a232ac4d060efe17669ecfe8341331ed88fa8555`
- 固定基线：`06b15caf9a69bc9c20f85522ce4abbb32c8b9245`
- 实施分支：`feature/TASK-008-six-action-dynamic-force-controller`
- 最终代码实现头（不含本报告后续文字提交）：`deb5307`
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
| `deb5307` | 恢复 TASK-003 动态刚体的 body CCD 契约并增加回归断言 |

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
- 已完成候选搜索并生成摘要；仅 MOVE 被选择，VIEW/UP 为 `null`，因此按协议不消费留出集。

### 覆盖率与胃部入口

- `MeshInput/ReferenceMesh` 保存 USD `rightHanded/leftHanded` 朝向并纳入几何哈希。
- 法线门控要求首次命中面法线与相机到命中点射线点积严格小于负容差。
- 旧 P0 调用默认 `require_camera_facing_normal=False`；TASK-008 显式启用。
- CPU PhysX 与 `cuda:0` Warp 光线设备解耦。
- 胃部按键映射和重复抑制已实现；动作间只刷新 Kit UI，不推进物理。
- 外部视口、胶囊 RGB、覆盖窗口和状态面板代码已接入；因无完整合格 profile，未进入胃部
  动作及主观三视图验收。

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

### 2026-08-22 重启后 live 续测

1. `nvidia-smi` 正常识别 RTX 5090、驱动 `595.84`；内核模块与用户态版本一致。
2. live preflight 生成 `/tmp/task008-preflight.json`：动态刚体、重力、scene/body CCD、
   240/60/30/1 Hz、胶囊几何和禁止位姿/速度写入扫描均通过。
3. CUDA 覆盖几何验证通过：GPU 与标量 first-hit 距离/面索引一致。
4. 完整标定建立了五个非 HOLD 动作各 20 个标定状态和 20 个独立留出状态；所有 setup
   均为零主动力 HOLD 落稳，静置接触力约 `0.05626 N`，与 5.735 g 胶囊自重一致。
5. 候选结果：
   - MOVE：`0.9mg`，MOVE_POS=`20/20`、MOVE_NEG=`20/20`，0 FAULT；
   - VIEW：随比例提高，最佳/上限 `3.0mg` 为 VIEW_POS=`15/20`、
     VIEW_NEG=`16/20`，0 FAULT；共享比例未达门槛；
   - UP：最佳 `2.197265625mg` 为 `6/20`；提高到 `2.74658203125mg` 和
     `3.0mg` 后分别降至 `2/20`、`1/20`，0 FAULT。
6. `selected_profile.json` 仅记录 `move_force_ratio=0.9`，VIEW/UP 为 `null`，它是失败
   状态记录而不是可运行冻结 profile。按防污染协议未执行任何留出动作，摘要里的空
   `held_out` 数组是“未消费”，不是 0% 实测成功率。
7. 因完整 profile 不存在，合同规定的胃部冻结 profile 六动作与人工三视图验收未执行。

### 首轮未通过/未执行（历史，已由上述续测取代）

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
| `/tmp/task008-preflight.json` | 2507 | `5561a53f6faad1d5afded8efb5657a4e89bf770bbcf35d312055215de852db3a` | 重启后 live 预检 |
| `/tmp/task008-dynamic-force-calibration/calibration_manifest.json` | 25547 | `976bee04da96bb1d230a26031b31272f1b6af3ae334bf85bbe0456d765516b0f` | 标定初始状态清单 |
| `/tmp/task008-dynamic-force-calibration/held_out_manifest.json` | 25217 | `c0eb734006e8f8bbf9c868d5d2d9647731b084f9734f30875917b29fe6227ff2` | 未消费的独立留出清单 |
| `/tmp/task008-dynamic-force-calibration/selected_profile.json` | 124 | `62aefcdb0168967541d1b4f76f991f796256a3498a55fa650368cc806668b922` | 不完整 profile：VIEW/UP 为 null |
| `/tmp/task008-dynamic-force-calibration/summary.json` | 542183 | `5bbb6304049925f3c47919ffab555f243c9a420781b15cc82d4dba170e659db8` | 全候选逐试验结果 |
| `/mnt/isaac-linux/isaacsim/kit/logs/Kit/IsaacLab/3.0/kit_20260822_103210.log` | 699222 | `00feeeca70e60946d1f64cc1123df65b71aa365859ca00775844aef5deb4bfb1` | 完整标定 Kit 日志 |

外部运行证据未提交到 Git；仓库只保存代码、测试、报告与凝练运行日志。

## 6. FAULT 分类与未验证声明

实现仅将非有限状态、仿真中断、完整穿越平面或逃离派生测试区域归为 FAULT；弹跳、普通
接触切换、滑动、位移不足和角度不足为普通失败。完整候选 campaign 已执行，所有候选的
FAULT 计数均为零；这不等价于动作全部成功。

以下项目保持未验证：冻结 profile 的独立留出成功率、胃部运动主观有效性、三窗口实时
同步效果和胃部场景 30 Hz GPU 覆盖耗时。候选搜索已证实当前模型在 `3.0mg` 内不能使
全部五动作通过。

## 7. 下一步决策点

驱动、preflight、CUDA覆盖几何和完整授权搜索均已完成。下一步不能直接把不完整 profile
用于胃部，也不能在看到留出结果后回调标定。方案端需要根据本报告授权修改 VIEW/UP 的
力方向、力矩构造、1秒时序或参数模型；修改后应重新生成新的标定/留出清单，完整重跑，
只有五动作均达到 `16/20` 且零 FAULT 后才进入冻结 profile 的胃部与人工三视图验收。
