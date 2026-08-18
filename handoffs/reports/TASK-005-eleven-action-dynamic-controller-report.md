# TASK-005 Linux 执行报告：十一动作动态控制器

## 结论

- **Disposition：`needs_decision`**。
- 十一动作接口、动态 COM wrench 控制器、平面任务、接触链路、键盘入口和自动标定/验收工具已实现。
- 81 组授权 VIEW/HOLD 增益网格的确定性 canonical 标定通过；最小共享 MOVE 系数为 `k=0.9`。
- 独立 seed 的 130 个平面正式样本未通过。依据合同停止条件，未执行 100 动作压力测试，也未迁移胃部任务。
- 未合并 `main`，未修改资产、质量、惯量、重力、材料、摩擦、恢复系数、既有 reset 或相机标定。

## Git 基线

- TASK-004 已验收基线：`87a80adcc367a3210fc1f8cfadea410f340e3918`
- TASK-005 planning head：`9865af8ac3e891a50b214b709d1a35f5257b8bc5`
- feature 分支：`feature/TASK-005-eleven-action-dynamic-controller`
- 报告前实现 head：`4a00c3336400f75cbab2bff286b84988ea9fac5c`
- 最终 feature head：本报告所在提交（可用 `git rev-parse HEAD` 复核；推送后在最终回复给出完整值）

实现提交：

1. `03dc624` `feat: define eleven-action dynamic contract`
2. `377d153` `feat: add local surface and contact geometry`
3. `89370c7` `feat: implement one-second eleven-action controller`
4. `5faa69b` `feat: apply eleven-action COM wrench in Isaac Lab`
5. `172fc65` `feat: add eleven-action flat keyboard task`
6. `4a00c33` `test: record failed randomized flat gate`

## 已实现范围

- 公开动作严格为 `0..10`，action dimension 为 1，无 actor mask。
- 非 FAULT 动作严格运行 240 个 240 Hz 物理子步。
- VIEW 为相机 frame 相对 15°；192 子步 quintic swing，48 子步保持。
- HOLD/VIEW 冻结局部法向、底部材料点和切向 anchor；动作结束从真实状态重建 READY_HOLD。
- MOVE 采用 60/120/60 子步自由/固定 COM force/自由阶段，torque 始终为零。
- 接触由 Isaac Lab 原生 `ContactSensor` 读取；GPU PhysX 下旧 contact callback 无事件的问题已绕过。
- VIEW/HOLD 使用最新物理子步接触点与接触力，仅补偿垂直于当前光轴的接触 swing 力矩；不控制轴向 twist。
- 运行期只调用 `permanent_wrench_composer.set_forces_and_torques_index(..., positions=None, is_global=True)`。
- 键盘入口动作执行期间丢弃新请求，不缓存、不排队、不抢占。

## 冻结 profile 与标定

最终 profile SHA-256：`dba847d76829edee4233be3bb8df663313c06dce2f47237fc91b4e2d0672889e`。

关键参数：

- `axis_kp_nm_per_rad = 0.02`
- `axis_kd_nms_per_rad = 0.0032`
- `support_kp_n_per_m = 20.0`
- `support_kd_ns_per_m = 0.8`
- `move_force_k = 0.9`
- force/torque/slew 上限保持合同值。

81 个候选中 11 个通过 canonical VIEW/HOLD 门禁。字典序最佳候选：

- 最大 VIEW 角误差：`1.836816999°`
- 最大支撑漂移：`0.001100819 m`
- wrench 积分：`0.816911653`
- MOVE POS 成功率：`0.90`
- MOVE NEG 成功率：`1.00`

calibration 使用 seed 42；为避免 81 组标定反复渲染 720p 图像，仅 calibration clone 禁用相机观测并把 render interval 设为 240。正式任务和 validation 保持相机与 120 FPS 调度。

## 正式平面门禁结果

命令使用 seed `20260818`，共 130 个样本，FAULT 为 0。所有结果子步均为 240。

| 动作 | 结果 | 有效样本 | 观测范围 |
|---:|---|---:|---|
| HOLD 0 | FAIL | 10 | 末端偏转 `0.00018°..5.56638°` |
| VIEW 1 | FAIL | 10 | 偏转 `8.91086°..21.50551°` |
| VIEW 2 | FAIL | 10 | 偏转 `6.22904°..19.41546°` |
| VIEW 3 | FAIL | 10 | 偏转 `8.14390°..16.00126°` |
| VIEW 4 | PASS | 10 | 偏转 `13.16245°..16.16608°` |
| VIEW 5 | FAIL | 10 | 偏转 `11.54413°..16.56103°` |
| VIEW 6 | FAIL | 10 | 偏转 `10.29485°..21.66088°` |
| VIEW 7 | FAIL | 10 | 偏转 `4.12710°..18.72896°` |
| VIEW 8 | FAIL | 10 | 偏转 `10.42929°..21.05913°` |
| MOVE POS 9 | PASS | 10 valid + 10 invalid | valid 位移成功率 `1.00` |
| MOVE NEG 10 | FAIL | 10 valid + 10 invalid | valid 位移成功率 `0.90`，1 个预期 valid 被 REJECTED |

VIEW 最大支撑漂移均低于 2 mm；失败主因是接触下 swing 的欠阻尼/相位敏感跟踪误差，而非支撑切向漂移。诊断姿态的 swing 速度在 0.2/2/5/10 秒仍约为 `1.58/0.96/1.89/1.10 rad/s`，延长 settle 不能稳定消除振荡。九组授权轴增益在正式最差姿态诊断中也均未完全通过。

## 实际命令与测试

主要执行命令：

```bash
./run_isaaclab.sh -p scripts/eleven_action/calibrate_eleven_action.py --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 --device cuda:0 --seed 42 --write_profile configs/eleven_action/dynamic_profile.json --headless
./run_isaaclab.sh -p scripts/eleven_action/validate_eleven_action_flat.py --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 --device cuda:0 --seed 20260818 --render_fps 120 --headless
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/eleven_action -q --disable-warnings
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/local_primitives tests/dynamic_force -q --disable-warnings
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest tests/ideal_surface tests/coverage tests/action_layer/test_atomic_protocol.py tests/action_layer/test_executor.py tests/action_layer/test_safety.py tests/action_layer/test_atomic_stomach_teleop_cfg.py tests/action_layer/test_atomic_keyboard_protocol.py -q --disable-warnings
```

结果：

- TASK-005 `tests/eleven_action`：`74 passed`
- TASK-004 与 dynamic force：`104 passed`
- ideal surface、coverage 与 action layer 选择性回归：`87 passed`
- 相关接触补偿隔离子集：`39 passed`

## 外部证据

以下文件保留在 Git 外：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `logs/eleven_action_calibration/20260818_155237_354525Z/calibration.json` | 383421 | `efff2b54b9db6f28e067c4387de9b2acefdbf6fd5bd0faf037fab6fff07554f9` |
| `logs/eleven_action_flat_validation/20260818_161220_922752Z/summary.json` | 1827 | `cd74c087b5db532b6345635b0d49fd1666fa5353c50f804976c8adf4d5a41220` |
| `logs/eleven_action_flat_validation/20260818_161220_922752Z/samples.jsonl` | 53623 | `937df15ccdcb6f67fab0e539701e0746bcd2cd6846365391d6dc4c702e312478` |
| `logs/eleven_action_teleop/20260818_134014_256537Z/session.json` | 63 | `003a279b20d792b3bf84c315f0a5d58493efb0f3387a1c5f07362a40a189a57b` |

## 偏差、警告与未验证项

- GPU PhysX 明确警告：`CCD disabled when GPU dynamics is enabled`。任务配置和刚体属性请求 CCD，但后端禁用；未静默切换 CPU、未修改物理参数。
- 启动日志还包含 UJITSO cooking service 不可用警告，不影响环境初始化和样本落盘。
- 渲染调度为 120 FPS，但脚本未记录实测 wall FPS；报告为未验证，不以目标值代替实测值。
- 平面门禁失败后按合同停止：没有 100 动作 sequence/hash、压力结果、碰壁取消延迟批量指标、胃部 digest、胃部压力、胃部交互证据或主观效果结论。
- 未运行胃部任务，不能声明 flat/stomach 同 digest 验证完成。

## 需要 Windows 方案端决策

现有授权增益网格、动作时长、15°目标、力矩/力/slew 上限和冻结物理条件下，确定性 canonical 标定可通过，但独立随机接触状态不能稳定通过。继续工作至少需要方案端明确授权下列一种方向：

1. 修改 VIEW/HOLD 控制结构（例如更完整的接触动力学观测器或受控轴角速度轨迹）；
2. 修改允许的增益/力矩/slew 范围；
3. 修改正式 trial 的 post-settle 定义与有效状态分层；
4. 调整 1 秒动作时长或验收阈值。

在收到新任务合同前，Linux 端不自行选择上述扩权方案。
