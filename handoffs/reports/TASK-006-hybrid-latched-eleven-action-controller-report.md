# TASK-006 Linux 执行报告：Hybrid Latched 十一动作控制器

## 结论

- **Disposition：`needs_decision`**。
- 已完成锁存数据合同、240 子步纯控制生命周期、Dynamic Lock 运行时适配器和 CUDA 阻断探针。
- 首选后端 `dynamic_lock_flags` 的 USD 属性读回正确，但 GPU PhysX 运行中没有可靠形成六自由度锁定：10 个保持样本出现明显漂移，100 个配对释放样本超出合同阈值。
- 依用户 2026-08-19 的明确补充要求，Dynamic 解锁配对实验失败后立即停止，**未启用或测试 kinematic 备用方案**。
- 未执行 TASK-006 action term 集成、1 Hz RGB barrier、平面随机验收、100 动作压力测试或胃部迁移。

## Git 基线与分支

- 精确实施基线：`67b7bf44747f08422add0cee7e6b94280bbeff6d`
- 原始规划提交：`3a48cbdb3384307c5c4b0e00d9ba6796f4c2ae5a`
- 规划分支头：`adcc86fe10749800fb40fb7016186520941c7132`
- 实施分支：`feature/TASK-006-hybrid-latched-v1`
- 报告前实现头：`b0a622c`
- 最终 head：本报告所在提交；推送后由 Linux 最终回复给出完整 SHA。

规划历史说明：合同要求“规划提交的直接父提交”为实施基线。原始规划提交 `3a48cbd` 的直接父提交确为 `67b7bf4`；分支头 `adcc86f` 是其上的纯文档修正提交，只修正 TASK-006 分支/交接文字，不包含实现改动。本次从合同指定的远程分支头创建 feature 分支，并保留这两层规划历史。

实现提交：

1. `35aecc6` `feat: define hybrid latch contract`
2. `af0f579` `feat: add hybrid latch lifecycle`
3. `b0a622c` `test: probe dynamic six-dof latch runtime`

## 已实现范围

- 新增冻结 profile `hybrid_latched_v1`；物理 240 Hz、策略 RGB 1 Hz、VIEW 门限 3°/2 mm、释放差异门限 0.5 mm/1°。
- 公共结果仍严格只有 `COMPLETED`、`REJECTED`、`FAULT`；动作 ID、15°目标、240 子步和 `0.9mg` 未改变。
- 新增 `LATCHED_READY`、内部 latch intent/reason/backend、不可变接触快照和边界遥测。
- HOLD 与 REJECTED MOVE 在纯控制器中全程锁存、零 wrench、仍计满 240 子步；VIEW 可在目标门或相机接触首次满足时发出 LOCK，但不提前结束动作；MOVE 保留 60/120/60 时序。
- Dynamic runtime 只在锁定/解锁边界清零 COM wrench 与速度，写 `lockedPosAxis/lockedRotAxis`，从不写 root pose。

## CUDA Dynamic Lock 阻断实验

命令：

```bash
./run_isaaclab.sh -p scripts/eleven_action/probe_hybrid_latch_backend.py \
  --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 \
  --device cuda:0 --backend dynamic_lock_flags --seed 20260819 --headless
```

运行环境：

- 设备：`cuda:0`，NVIDIA GeForce RTX 5090
- 物理步长：`1/240 s`
- GPU dynamics：启用
- GPU 后端警告：`CCD disabled when GPU dynamics is enabled`
- `selected_backend` 保持 `dynamic_lock_flags`
- Dynamic profile SHA-256：`dba847d76829edee4233be3bb8df663313c06dce2f47237fc91b4e2d0672889e`
- Latch profile SHA-256：`e2f3759be70f5e4b70647c8fd52cc299fa82f2bf38ac6a166364d19ca1941ad3`

采样量：10 个分层姿态的一秒保持试验；动作 1–10 每个 10 组相同起始状态配对释放，共 100 对。

| 检查项 | 合同阈值 | 实测最坏值 | 结果 |
|---|---:|---:|---|
| LOCK mask 读回 | pos/rot 均为 `0b111` | 全部为 `0b111` | PASS |
| UNLOCK mask 读回 | pos/rot 均为 `0` | 全部为 `0` | PASS |
| API 时刻位置跳变 | ≤ `1e-9 m` | `0 m` | PASS |
| 锁定一秒位置漂移 | 数值容差 `1e-7 m` | `0.007328268 m` | FAIL |
| 锁定一秒光轴漂移 | 数值容差 `1e-4°` | `66.771131°` | FAIL |
| 锁定线速度 | ≤ `1e-7 m/s` | `7.05991e-5 m/s` | FAIL |
| 锁定角速度 | ≤ `1e-7 rad/s` | `0.0219963 rad/s` | FAIL |
| 配对释放位置差 | ≤ `0.0005 m` | `0.012459569 m` | FAIL |
| 配对释放光轴差 | ≤ `1°` | `68.614106°` | FAIL |

直接观察表明：USD 属性值可立即读回，但 CUDA 运行中的锁标志没有对所有分层接触姿态稳定约束 PhysX 刚体；因此“属性存在/读回正确”不能作为后端可用证据。该失败是阻断门结果，不以修改阈值、CCD、物理参数或重复写 pose 掩盖。

## 测试与回归

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/eleven_action -q --disable-warnings
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/local_primitives tests/dynamic_force -q --disable-warnings
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run_isaaclab.sh -p -m pytest \
  tests/ideal_surface tests/coverage \
  tests/action_layer/test_atomic_protocol.py \
  tests/action_layer/test_executor.py \
  tests/action_layer/test_safety.py \
  tests/action_layer/test_atomic_stomach_teleop_cfg.py \
  tests/action_layer/test_atomic_keyboard_protocol.py -q --disable-warnings
```

- TASK-006/TASK-005 十一动作测试：`87 passed`
- TASK-004 与 dynamic force：`104 passed`
- ideal surface、coverage 与 action layer 选择性回归：`87 passed`

## 外部证据

以下证据保留在 Git 外：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `logs/hybrid_latched_task006/backend_probe/20260819_061053_239299Z/summary.json` | 1688 | `4fed5694123bbdf4d1299e5dde203a87e5c1176b2e1b2d1d60d996835d643bd0` |
| `logs/hybrid_latched_task006/backend_probe/20260819_061053_239299Z/probe_rows.jsonl` | 373197 | `27abe7fef640528edf50512c5b6e75b9b7076cac812ee037b7106554f82b16d5` |

`probe_rows.jsonl` 含每次锁定/解锁读回、API 前后位姿、保持速度/漂移以及每个动作前 0.05 秒的 direct/latched 配对轨迹。

## 偏差与未验证项

- 合同原本允许 Dynamic 失败后显式选择 kinematic；用户本轮明确要求失败先返回报告且不启用 kinematic。该用户补充要求优先执行。
- 未改 `selected_backend`，未修改 CCD、资产、几何、质量、惯量、重力、材料、solver、机器人、磁体、VLM/RL、reward 或 coverage。
- 因首个 GPU gate 失败，未验证 policy RGB latch barrier、平面随机动作效果、固定 100-ID 序列、胃部相同 digest 运行、键盘可视化或 wall FPS。
- 当前实现停留在运行时后端门禁前的可复现阶段，不能声明 TASK-006 控制器可用于训练或胃部任务。

## 需要 Windows 方案端决策

下一份合同需要明确选择新的边界稳定机制。可评估但本次未执行的方向包括：

1. 授权并单独门禁原设计中的 kinematic 后端；
2. 使用 PhysX/Fabric 运行期原生刚体锁定接口，而不是仅在活动仿真期间改 USD schema 属性；
3. 在场景初始化前预置锁属性，并验证运行期解锁/重锁是否真正同步到 GPU PhysX；
4. 设计新的模拟特权 latch，但仍需满足无 pose snap、释放配对和 RGB barrier 约束。

Linux 端不会在没有新任务合同的情况下自行选择上述方案。
