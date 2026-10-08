# 新胃连续 RL 集成与训练可行性预验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏 release A 版和旧 TASK-010 的前提下，验证新胃连续动作执行、10 Hz 覆盖奖励、1 Hz 观测与连续 GRU-PPO 训练链路，并给出可复现的并行训练可行性报告；**不启动正式四组长训练**。

**Architecture:** 新增独立新胃 RL task 与适配层，复用现有磁场物理、安全规划和覆盖算法。执行器支持 1s Single 或四段 0.25s Chunk；在真实物理子步中每 24 步计算一次几何可见性，累计 C10 与四项旧奖，1 Hz 相机边界另记 C1。通过最小连续 PPO smoke 后，逐级验证 1/2/4/8/12 环境吞吐和隔离性。

**Tech Stack:** Python、Isaac Lab/Isaac Sim、PhysX、PyTorch、Warp 可见性、SciPy 轨迹规划、pytest。

**Spec:** `docs/superpowers/specs/2026-10-08-new-stomach-rl-preflight-design.md`（随本计划一并复制到仓库，内容来自同名交付规格文件）；既有事实依据 `docs/RELEASE_A_NEW_STOMACH_MAGNETIC.md`、`docs/NEW_STOMACH_COVERAGE_TUBE_CALIBRATION.md`。

## Global Constraints

- **版本锚点：**`release/new-stomach-magnetic-v1-20261008`，检索时 HEAD 为 `1e89c9ccb48545c4513915be94d4c17d9cdebaaa`。执行时记录当前 HEAD；如变化，先对照代码差异再执行。
- **协作：**遵守 `AGENTS.md`：Windows Codex 规划/验收；Linux Codex 在独立 `feature/new-stomach-rl-preflight-20261008` 分支实现；不得推送 main，不丢弃用户现有未提交改动。
- **已冻结的实验约束：**Single=9D×1s；Chunk=4×9D×0.25s、四段全执行；相机及策略 1 Hz；物理 240 Hz；C10 10 Hz 主奖、C1 1 Hz 辅助；70 mm；新胃右管道+Tube B 排除掩码；四项 TASK-010 奖励权重不变；未来正式测试固定 50 个位姿。
- **旧任务不动：**`scripts/stomach_coverage/train_task010.py`、旧 CNN+GRU、旧 FORCE/MOVE/VIEW/UP、旧胃 pose/coverage manifest、原发布 task `Template-Robotarm-Magnetic-New-Stomach-Vector-Lab-v0` 不改既有行为。
- **安全不降级：**保持 5 mm 碰撞安全余量、关节限位、速度/加速度、停止路径、磁力和磁矩保护；不得改变胃/机器人 USD、场模型参数或通过直接设置胶囊 pose 制造运动。
- **信息隔离：**仅 Actor 使用 1 Hz RGB（Vision）及安全投影后实际提交的历史指令；Blind 不读图像。胶囊 pose/coverage/磁场真值仅可进 reward、诊断或明确声明的特权 Critic，不能进入 Actor 或现实安全控制判断。
- **资产与文件：**已确定的新胃 pose 库须核对真实路径和 SHA-256；若找不到，返回 `needs_input`，不得拿旧胃文件替代。大图像、checkpoint、日志放 Git 外，报告提供路径/哈希。
- **停止规则：**动作安全、覆盖正确性、奖励一致性、Actor 隔离等**正确性 Gate**失败时不得继续后续实验；容量测试如果只验证到 1/2/4/8 环境，允许记为 `limited_pass` 并在最大已验证环境数上完成最后的短时端到端验收，但不得冒称 12 环境可用或启动正式训练。任何影响研究语义或安全接口的歧义，返回 `needs_decision`，禁止 Codex 自选近似方案。

## Review Focus（各项必须有对应测试）

1. **动作混淆：**验证新 `4×0.25s` 四段都执行，旧 `4×1s` 首秒提交/后三秒预览保持原语义（Gate 1）。
2. **帧率混淆：**1 Hz 相机不得因 C10 更新变成 10 Hz；C10 必须使用真实 0.1s 物理状态，不得复用过期 camera pose 或插值终点（Gate 2）。
3. **错误胃/掩码：**检测新胃 mesh SHA、429029/176109/252920 面片数、目标面积、70 mm；旧胃 pose、旧可见性掩码拒绝（Gate 0/2）。
4. **跨环境串扰：**多环境重置、磁场力、轨迹引用、视觉帧号、cover mask、GRU hidden state 均按 env_id 隔离（Gate 5）。
5. **奖励与观测泄漏：**C0 不得获得策略奖励，10 个 0.1s reward 正确累加成 1s PPO reward；无效/未提交的预测动作不进入历史，真值不进入 Actor（Gate 3/4）。

## 文件结构与职责

下列为**计划新增**文件，不代表当前仓库已存在。`PKG=source/robotarm_magnetic_lab/robotarm_magnetic_lab`。

- `PKG/tasks/manager_based/robotarm_magnetic_lab/controllers/new_stomach_rl_trajectory.py`：Single/Chunk 纯轨迹数学，1s 预算与 C2 连续性。
- `PKG/tasks/manager_based/robotarm_magnetic_lab/mdp/new_stomach_rl_action.py`：新 task 的 action term、物理子段执行及诊断，不替换旧 `actuator_vector_action.py`。
- `PKG/coverage/new_stomach_rl_coverage.py`：新胃掩码、C10/C1 独立状态、基于真实物理位姿的 GPU 几何可见性。
- `PKG/runtime/new_stomach_rl_reward.py`：10 Hz TASK-010 四项奖励复用与每秒加总。
- `PKG/tasks/manager_based/robotarm_magnetic_lab/mdp/new_stomach_rl_terms.py`：1 Hz 视觉/动作历史、奖励及特权 Critic 适配。
- `PKG/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_new_stomach_rl_env_cfg.py`：独立任务 ID、时钟、Term 与终止条件；明确覆盖 bring-up 中性奖励。
- `PKG/learning/new_stomach_rl_actor.py`、`new_stomach_rl_distribution.py`、`new_stomach_rl_runner.py`：GRU 连续策略、9D/36D log-prob、最小 PPO smoke/日志。
- `scripts/new_stomach_rl/validate_preflight.py`、`benchmark_preflight.py`、`train_smoke.py`：各 Gate 可复现 CLI。
- `tests/new_stomach_rl/test_action.py`、`test_coverage.py`、`test_reward.py`、`test_observation.py`、`test_ppo.py`、`test_multi_env.py`：对应 TDD。
- `handoffs/reports/NEW-STOMACH-RL-PREFLIGHT-20261008.md`：Gate 结果与阻塞项。不得先写“通过”结论。

---

### Task 0 / Gate 0：只读环境、版本与数据预检

**Files:** Read `AGENTS.md`、`docs/RELEASE_A_NEW_STOMACH_MAGNETIC.md`、`docs/NEW_STOMACH_COVERAGE_TUBE_CALIBRATION.md`、`configs/new_stomach_v1/orientation_v1.json`、`controllers/actuator_vector.py`、`mdp/actuator_vector_action.py`；Create `scripts/new_stomach_rl/validate_preflight.py` 中的 `audit_assets()` 与 `tests/new_stomach_rl/test_preflight.py`。

- [ ] Step 1：在**独立 worktree** 记录 `git status`、`git rev-parse HEAD`、设备/驱动/GPU/Isaac Sim/Isaac Lab/PyTorch 版本及外部磁场、URDF、USD 依赖；若与 release 版本不一致，明确差异。
- [ ] Step 2：定位用户**已经确定**的新胃初始位姿清单与数据，校验新胃几何 SHA、train/validation/test 切分、reset 后动态稳定性；记录绝对路径和哈希。缺少文件则 `needs_input`，不得自动生成或引用 `configs/task009b/pose_library_manifest_v1.json`。
- [ ] Step 3：核对候选不可达掩码原始工件；若 Git 忽略的 `candidate_face_mask.npz` 在 Linux 本地不存在，可按已记录的**确定性几何规则**重新生成并逐面片对照哈希/标记，不得改变选择规则。若证据无法重建，返回 `needs_input`。
- [ ] Step 4：先写测试 `test_reject_old_stomach_pose_manifest`、`test_mask_geometry_identity`、`test_release_action_semantics`，运行失败后仅补充预检实现并重跑。预期：不匹配/缺失有明确错误，匹配资产可通过。
- [ ] Step 5：执行已有相关单测 `tests/stomach_migration tests/magnetic_following`，记录通过/失败实数及原因；提交预检代码和门禁清单。**Gate 0 未过：停止。**

### Task 1 / Gate 1：新 Single/Chunk 执行接口及安全验证

**Files:** Create `controllers/new_stomach_rl_trajectory.py`、`mdp/new_stomach_rl_action.py`、`tests/new_stomach_rl/test_action.py`；Read/reuse `controllers/actuator_vector.py` 和当前安全几何组件。

**Interfaces:** `build_schedule(mode: Literal['single','chunk'], action: np.ndarray, start_state: JointState) -> PlannedTrajectory`；`PlannedTrajectory` 至少包含 241×9 joint reference、joint velocity/acceleration、4 个子段边界、裁剪/投影比例、认证安全余量及真正提交的动作记录。必要的 JointState/PlannedTrajectory 类型在新增模块定义，不能复制不存在的旧 API。

- [ ] Step 1：写失败测试：Single 输入 9D、Chunk 输入 4×9D；尺寸非法/NaN 拒绝；全零 HOLD；单动作执行 240 ticks；四段动作在 60/120/180/240 tick 均反映对应子段（改动最后一段只应影响最后 0.25s 及相关连续性规划，不允许只执行第 1 行）；旧 4 秒 preview 单测原样通过。
- [ ] Step 2：明确新动作仍是末端平移/旋转增量 + Ball 增量，不转成 `v1/v2` 绝对关节 offset。采用统一**每 1s 的物理动作预算**：Single 上限沿用 `SCALES`，Chunk 各 0.25s 子段按 `SCALES/4` 起始标度；实际轨迹共同受原关节速度/加速度/5mm clearance 约束；计划失败按现有投影/HOLD 逻辑执行，并记录比例。
- [ ] Step 3：实现可核验的 0.25s 轨迹规划，确保 240Hz 位置/速度/加速度连续、各关节限位及停止距离满足原约束。**不得**把原 1s 五次轨迹仅换成 0.25s 时间戳。若同一标度下无法产生实际可运动的可行轨迹，返回 `needs_decision` 而非私自放宽约束。
- [ ] Step 4：单环境依次运行 zero/正负轴向/组合/冲突场景；检查全 240 tick、四个时段、实际电机命令、磁场力矩仍按 240Hz 生效、碰撞时安全拒绝而非硬传送。记录与旧预览行为回归的区别。
- [ ] Step 5：运行 `pytest tests/new_stomach_rl/test_action.py tests/magnetic_following/test_actuator_vector.py -q`（使用仓库 Isaac Python），全部通过才提交。**Gate 1 未过：停止。**

### Task 2 / Gate 2：新胃 C10/C1 覆盖率、70mm 与可达掩码

**Files:** Create `PKG/coverage/new_stomach_rl_coverage.py`、`tests/new_stomach_rl/test_coverage.py`；Read/reuse `coverage/new_stomach_tubes.py`、`coverage/batched_visibility.py`、`coverage/area_weights.py`、`coverage/simulator_runtime.py`。

**Interfaces:** `reset(env_ids) -> None`；`update_geometry_at_physics_tick(physics_tick: int, capsule_pose: Tensor) -> CoverageDelta | None`（每 24 个真实子步产生一条 C10）；`observe_camera_frame(frame_id, sim_time_s, optical_pose) -> CoverageDelta`（每真实 1Hz 帧更新 C1），实现者须确保一次相同采样边界不双计数。两条路径共享**相同** FOV/法向/first-hit/70mm 判定和可达权重，但维护不同累计 mask。

- [ ] Step 1：写失败测试：面片 429029，排除 176109，目标 252920、面积约 0.040205301m²，错 mesh 哈希拒绝；70mm 内边界可见、边界外不可见；遮挡/背向不可见；累计非降；排除面始终不计。
- [ ] Step 2：实现 C10：从**每 24 个 240Hz 物理子步后的真实胶囊 pose**合成相机外参，做几何 raycast；不得读取 1Hz 更新后可能过期的 camera pose，不得在 10Hz 强制 RGB render，也不得只用一秒首尾 pose 插值。
- [ ] Step 3：实现 C1：仅同步真正的 1Hz RGB 采集时间戳更新。检查 `0 <= C1 <= C10 <= 1`，两条曲线各自单调；固定 120s 回合含 C10 的 `C0+1200` 点与 C1 的 `C0+120` 点，记录两者的时间戳/采样依据。
- [ ] Step 4：用固定参考姿态进行纯函数/单环境一致性验证，校验同一时刻两个分支可见 mask 完全一致；重置后 mask 清零、C0 重算。GPU raycast 错误立即失败，不能退化为近似面积。
- [ ] Step 5：运行 `pytest tests/new_stomach_rl/test_coverage.py -q` 并执行 `validate_preflight.py --gate coverage`，提交证据。**Gate 2 未过：停止。**

### Task 3 / Gate 3：四项奖励与 1 Hz PPO 边界

**Files:** Create `PKG/runtime/new_stomach_rl_reward.py`、`PKG/tasks/manager_based/robotarm_magnetic_lab/mdp/new_stomach_rl_terms.py`、`tests/new_stomach_rl/test_reward.py`。

**Interfaces:** `reset_reward(env_ids) -> None`；`update_reward_10hz(capsule_pose, C10) -> RewardTerms`；`finish_policy_second() -> Tensor[N]`，其输出为当前 1s 内 10 个 0.1s 分项的**和**。迁移 `Task010RecoveryTracker` 的既有 5s 状态机与固定权重，不改旧模块默认值。

- [ ] Step 1：测试 `R_k=100ΔC_k+0.1ΔE_k-0.002I_stuck+0.2I_resumed`、十步加总及不同 K 一秒总奖励一致的时间定义；C0 不给策略奖励、reset HOLD 不计、零新增覆盖不会凭空增加 coverage reward。
- [ ] Step 2：保留旧 5s 无进展窗口及 escape/locked/resumed 转移机制，针对新磁控零动作、可行运动、持续受阻进行单环境轨迹预实验；将四项贡献分开记录以识别脱困奖励刷分。
- [ ] Step 3：新增新 task 的**独立 RewardsCfg**，显式覆盖 inherited `alive/action_rate/joint_velocity/collision` 联调奖励；物理碰撞/安全失败仍有终止逻辑，不把其与奖励混为一谈。
- [ ] Step 4：`pytest tests/new_stomach_rl/test_reward.py -q` + 单环境 120s 随机/脚本策略回合；验证所有项有限、时间归一化、分项之和严格等于总奖。通过再提交。**Gate 3 未过：停止。**

### Task 4 / Gate 4：独立新胃 RL 任务、Actor/GRU 与短程 PPO smoke

**Files:** Create `robotarm_magnetic_new_stomach_rl_env_cfg.py`、`learning/new_stomach_rl_actor.py`、`new_stomach_rl_distribution.py`、`new_stomach_rl_runner.py`、`scripts/new_stomach_rl/train_smoke.py`、`tests/new_stomach_rl/test_observation.py`、`test_ppo.py`；参照但不改 `learning/task010_*`。

**Interfaces:** 新独立 task ID `Template-Robotarm-Magnetic-New-Stomach-RL-Preflight-v0`；四模式 `A/B/C/D` 共享一个 GRU 主干（输入视觉 512D 及**上一秒已提交执行**的统一 4×9 指令记录），输出维度分别为 9 或 36。Blind 的 512D 置零；Vision 继续 frozen ResNet18；continuous policy 用正确计算 log_prob 的有界动作分布，同一实现供四组复用。

- [ ] Step 1：写失败测试：1Hz RGB 只更新一次/秒；A/C 的视觉全零；B/D 能收到对应帧；初始/重置历史动作清零、未提交 preview 不进入历史；Actor 张量不包含真值 pose、coverage、环境编号与测试 split。
- [ ] Step 2：确认执行记录映射为相同物理意义的 4×9 历史（Single 需按四个真实 0.25s 子区间记录已下发指令，不能将单个 1s **相对增量**无缩放复制四份）；Actor 不使用未投影 raw command，Critic 特权路径需隔离并在日志标注。
- [ ] Step 3：写失败测试：9/36 维动作采样范围、含 tanh 变换的 log-prob 一致性、有限梯度、GRU reset mask、checkpoint 保存恢复、不同维度 entropy/kl 归一化说明；实现单步和短 rollout PPO 更新，不重用旧版 mode+Beta 输出头。
- [ ] Step 4：1 环境分别对代表模式 A 与 D 运行不超过 2–5 PPO updates；检查 policy/value loss、KL、reward、动作与状态均有限，checkpoint 可恢复、没有 actor/critic 观测泄漏。**这只是可训练性验证，不要求覆盖率提升。**
- [ ] Step 5：PPO 1s 折扣需与旧 0.1s 时间基准区分：smoke 可先采用每秒 `gamma = 0.999^10`，`GAE lambda=0.95` 暂定，并在报告显式标为**正式训练前待冻结**；不可直接静默沿用旧 `gamma=0.999` 每 1s。
- [ ] Step 6：通过 `pytest tests/new_stomach_rl/test_observation.py tests/new_stomach_rl/test_ppo.py -q` 和短程 smoke 后提交。**Gate 4 未过：停止。**

### Task 5 / Gate 5：单/多环境隔离、性能与训练预算估算

**Files:** Create `scripts/new_stomach_rl/benchmark_preflight.py`、`tests/new_stomach_rl/test_multi_env.py`；仅在必要、明确安全可证明时局部改新 RL task 支持。现行 `ActuatorVectorAction` 强制 `num_envs=1`，**不能仅删除断言**。

- [ ] Step 1：先列出哪些组件持有单环境索引、CPU IK/碰撞缓存、磁场桥、RGB 帧号及输出日志；针对 `env_id=0/1` 写失败测试：相同行为一致、不同动作分叉、只 reset 一个环境、覆盖和奖励不串扰、GRU hidden state 和随机数独立。
- [ ] Step 2：1 环境充分通过之后，按 `1 -> 2 -> 4 -> 8 -> 12` 逐级尝试。若 USD、磁控桥、镜头绑定或碰撞缓存无法安全复制，立即报告 `partial/needs_decision`，不得强行放宽 5mm 余量、物理频率或覆盖几何。
- [ ] Step 3：每个支持的并行数至少做 `8` warmup + `32` policy steps × `2` 次重复；记录 `env_steps/s`、`transitions/s`、单步墙钟分解（IK、安全规划、磁场、10Hz raycast、RGB、PPO）、峰值 GPU/CPU 内存、成功/安全投影/HOLD 比例、异常及全部版本。对最大通过的配置再完成至少两个 64-step rollout，验证长一些时的稳定性。
- [ ] Step 4：仅在隔离测试成功的并行度运行代表模式 A、D 的 2–5 update PPO smoke；不要开始 3 seeds×4 groups×1000 updates。报告使用真实测得的吞吐量外推正式 **768000 transition/seed（若沿用 12×64×1000）** 的耗时与显存，明确推算假设和不确定性；如果只有 1/2/4 个 env 可用，也报告对应预算，不冒称 12 环境已支持。
- [ ] Step 5：运行 `pytest tests/new_stomach_rl/test_multi_env.py -q` 及 `benchmark_preflight.py`；提交原始 CSV/JSON 的路径、SHA-256 和摘要。**Gate 5 支持 `pass / limited_pass / blocked`：如果至少 1 环境完整通过，仅并行上限较低，可在已验证的最大并行数进入 Gate 6；若安全/数值正确性出错则 blocked 并停止。任何情况都不启动正式训练。**

### Task 6 / Gate 6：端到端回归、报告与交接

**Files:** Create `handoffs/reports/NEW-STOMACH-RL-PREFLIGHT-20261008.md`；Read `AGENTS.md` 中 Evidence Requirements。

- [ ] Step 1：对通过的最大稳定配置，使用**train/validation 而非 test** 位姿完成 120s 可复现回合：记录 240Hz 动力学、1Hz RGB 帧号、C10 `1201` 点、C1 `121` 点、四项奖励、分项累计、动作投影、安全违规/停止原因；验证最后状态与下一次 reset 没有继承覆盖或 GRU history。
- [ ] Step 2：运行新增与原回归测试；记录准确测试计数，不能把发布时已报告的 `65 passed` 当成本轮实际测试结果。至少检查一次断开重启/恢复 checkpoint 后观察频率、Reward 与动作含义未变化。
- [ ] Step 3：报告 Git base/head/分支、所有新增与修改文件、执行命令、软硬件环境、每个 Gate 的 `pass/fail/not_run`、时间与吞吐表、原始 artifact 路径+SHA-256、失败原因、权衡与尚未验证的真实 GPU 情况。给出正式 A/B/C/D 多种子训练的估算资源，但**不得自动启动**。
- [ ] Step 4：最终状态只能是 `complete`（全部门禁通过）、`partial`（部分通过或单环境训练可行但多环境未过）、`needs_input`（用户已确定但文件不可获取）、`needs_decision`（必须改变安全/接口/研究设计）。不要把 blocked gate 写成 passed。
- [ ] Step 5：保留旧 release 可完整运行；所有实现只提交 feature 分支供 Windows Codex 审查，未经明确要求不直接合并或推送 main。

## 建议的执行命令（新增脚本需实现对应 CLI）

```bash
# 在隔离 worktree 中确认代码身份
git rev-parse HEAD && git status --short

# 纯测试：使用 release 文档约定的 Isaac Lab Python，规避无关 ROS pytest 插件
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl tests/stomach_migration tests/magnetic_following -q

# 新增脚本的拟定接口：按 Gate 单独运行、记录外部工件
./run_isaaclab.sh -p scripts/new_stomach_rl/validate_preflight.py --gate coverage --device cuda:0 --headless
./run_isaaclab.sh -p scripts/new_stomach_rl/train_smoke.py --group A --num_envs 1 --updates 2 --device cuda:0 --headless
./run_isaaclab.sh -p scripts/new_stomach_rl/benchmark_preflight.py --num_envs 1,2,4,8,12 --device cuda:0 --headless
```

> 注意：上述新脚本与参数是**交付接口规范**，当前 release 并不存在，不得把“命令示例”当作已经验证可运行。

## 完工定义与交接要求

- 必须同时提交：独立新胃 RL task、动作/覆盖/Reward/Actor-PPO 预验证代码、纯单测、实际 GPU smoke 原始结果、并行性能结果、汇总报告、全部可复现命令和配置哈希。
- **若没有正确的新胃位姿库/目标掩码，或新 0.25s 控制器安全性无法保证：不得声称完成。若只是最大稳定环境数小于 12，结论应标记 `limited_pass` 并明确正式训练预算风险，不可虚报 12 环境通过。**
- 最终给用户呈现三个明确判断：① 新胃 1s 控制闭环是否正确；② 10Hz/C1 及四项 Reward 是否可信；③ 以现有算力启动正式四组实验需要怎样的并行数、预估时长与风险。
- 本计划属于**预验证阶段**，不包含正式 PPO 调参、四组大规模对比、跨胃泛化、扰动恢复、sim2real 迁移。
