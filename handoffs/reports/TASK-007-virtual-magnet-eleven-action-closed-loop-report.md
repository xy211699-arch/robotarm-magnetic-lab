# TASK-007 Linux 执行报告

## 结论

- 处置：`needs_decision`
- 实现代码 HEAD：`1e1b9f2975b115edbc40dfd843c4eaf5f3425439`
- 强制门禁状态：平桌无扰动开发样本未达到所有动作阈值，因此未启动正式
  20 样本 held-out 门禁、扰动配对门禁、胃部门禁或 100 动作序列。
- 该结论不是程序 `FAULT`：运行状态、有限磁体输出和 PhysX 状态均保持有限；
  阻塞来自同一共享参数对 VIEW 方向的可复现强方向性。

## 已完成实现

1. 将 Magpylib 5.2.3 有限尺寸磁体模型内聚到仓库，并保留来源、许可证和哈希。
2. 实现 11 个动作、三结果、240/60/1 Hz 时序及 0.8 s 运动 + 0.2 s 稳定协议。
3. 实现有限模型 6D 中央差分逆解、可控子空间条件数、信任域、工作距离和磁矩姿态精化。
4. 实现动态胶囊 240 Hz 有限模型力/矩桥；不驱动机械臂或 Ball，不写胶囊位姿/速度。
5. 实现 15° VIEW 五次轨迹及角速度前馈、5 mm MOVE、HOLD、无效 MOVE 全秒 HOLD 替代。
6. 实现确定性试验清单、终端验收汇总、外部证据哈希和单键单动作可视化。
7. 修复试验随机化后虚拟磁体初态不同步，以及空闲 `-1` 路径遥测字段缺失问题。

## 有限模型与配置证据

- 有限模型清单：`handoffs/reports/TASK-007-magnetic-dependency-manifest.sha256`
- 有限模型配置 SHA-256：`bcabf92c4189e8740cf00ac12bd8e242e7294c2a59e080fe1e6903522dc33427`
- 最终控制器配置 SHA-256：`082800dc49acc364fca1f86f80c5eda682e4c5e2a0a5c032e824a7be36fc052e`
- 胶囊保持动态刚体；虚拟磁体仅为无碰撞、无刚体 API 的调试 Xform。
- 实际施加力/矩逐子步断言等于有限模型滤波输出；不存在理想期望力直接注入路径。

## 回归结果

- 仓库测试：`57 passed`，`0 failed`。
- 旧有限模型数值回归及既有轴向场扫描/长轴滚动回归在前置提交中通过。
- 运行时合同检查已确认：240 物理子步、60 次反馈、动态胶囊、无机械臂/Ball 动作项。

## 开发校准结果

曾获得的单样本通过结果：

- `VIEW_UP`：误差 1.615°，切向漂移 1.610 mm，末段线速度 0.195 mm/s，
  角速度 0.0567 rad/s。
- 稳定初态 `HOLD_VIEW`：误差 0.139°，漂移 0.017 mm，线速度 0.407 mm/s，
  角速度 0.0591 rad/s。
- `MOVE_SIDE_POS/NEG` 曾分别产生 5.103/5.530 mm 有效位移，但对应角速度
  0.144/0.147 rad/s，略高于 0.1 rad/s 门限。
- 两个无效 MOVE 均正确执行 240 子步并返回 `REJECTED`，且稳定门限通过。

同一配置的八方向开发冒烟中，只有 `VIEW_UP` 通过；多数方向终点误差为
9.5–17.4°。延后低磁矩制动可减小角度误差，却把末段角速度提高到
0.28–0.90 rad/s。减小磁矩可满足稳定性，却无法在 1 s 内消除方向误差。
终端有限模型原始磁矩与期望磁矩同号且数值相近，故不是末端逆解符号错误。

## 外部开发证据

| 路径 | 字节 | SHA-256 | 内容 |
|---|---:|---|---|
| `/tmp/task007-view-stabilization-torque-cap.json` | 3076 | `534ef0206a7be7c20a148698b10732c0d845d2fdbb4005d382771528bef5bc16` | 单 VIEW 全门限通过 |
| `/tmp/task007-all-actions-dev1.json` | 35349 | `a5d91336583d35046afe6aac374bb7f51a13fe57bce8cc0b36f8e632480d673e` | 11 动作及无效 MOVE 冒烟 |
| `/tmp/task007-view-directions-dev.json` | 21899 | `fdaeee6eddbca6cd56ff540d413322ac683cfd13d80c63d28d5cbf63c9525ea0` | 八方向方向性证据 |
| `/tmp/task007-hold-view-cap05.json` | 5758 | `c0fbd42317a93b8f798d93244680eb4a1ed072679270cc0f0ec704c352a8204e` | HOLD 低磁矩稳定证据 |
| `/tmp/task007-view-torque12.json` | 3070 | `59db2478616ad5194cbe84cec7f06d85c28c55001ab12da6eb5e9c7669481bac` | 运动段 1.2 mN·m VIEW 通过 |
| `/tmp/task007-all-actions-dev2.json` | 35358 | `b56629fe3090c53d28ec1b774bddbb06f0d2e4ca1bf7cbe0bd800fa3372a2669` | HOLD 稳定时序对比 |

这些是开发证据，不是 held-out 验收结果。正式平桌/胃部每动作成功数均为“未运行”，
不得解释为 0/20 或通过。

## 未执行门禁

- 平桌无扰动 held-out：未运行（开发门禁未满足）。
- 平桌扰动及反馈禁用配对：未运行。
- 胃部开发与 held-out：未运行，遵守强制门禁顺序。
- 平桌和胃部 100 动作不重置序列：未运行。

## 需要方案端决策

当前 1 秒动作与最后 0.1 秒严格静稳门限，在固定胶囊物性/接触参数下对八方向
VIEW 形成冲突。下一步至少需要批准以下一种设计变化：

1. 允许磁体逆解显式补偿接触约束下的胶囊总力矩（含磁力作用点到质心的力臂）；
2. 允许 VIEW 动作按方向使用在线自适应制动切换，而非单一固定磁矩时序；
3. 放宽动作时长或终端角速度门限；
4. 引入表面切平面相关的磁体名义位姿，而不是单一胶囊相对名义位姿。

在决策前继续扩大磁矩只会重复产生“角度改善、末段速度失败”，不应进入正式门禁。

## 可视化命令

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab

ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/virtual_magnet/teleop_virtual_magnet.py \
  --scene flat --device cuda:0 --render_fps 120
```

按键：`0` 为 HOLD，`1..8` 为八个 VIEW，`9` 为正向 MOVE，`-` 为反向 MOVE，
`R` 重置，`Esc` 退出。执行中输入和按键重复会被丢弃，不排队。
