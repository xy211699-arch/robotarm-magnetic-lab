# TASK-010 冻结 ResNet18 加 GRU 开发种子实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Linux 执行端若没有这些可选技能，不构成阻塞；必须按相同顺序逐项实施、测试、提交并保留门禁证据。

**Goal:** 在已验收的十二环境向量化基础设施上实现冻结 ResNet18 加单层 GRU 的纯视觉时序 PPO 基线，并把本次工作严格收束在 Gate 1 至 Gate 4 的开发种子启动门禁。

**Architecture:** 新任务继承 `TASK-009D-0` 的物理、相机、覆盖与同步 reset 合同，Actor 只读取当前 RGB 经冻结 ResNet18 得到的 512 维特征、上一实际动作和自身 GRU 状态；训练期 Critic 独立读取 65 维特权状态。项目内自定义混合离散—连续分布、循环 Actor、非对称 Critic、PPO 与 runner，通过 RSL-RL 的 `class_name` 接口接入，不修改系统安装的 RSL-RL。Gate 4 只验证脱离 Codex 的启动、查询、恢复和失败留痕，不等待完整开发种子结束。

**Tech Stack:** Isaac Lab 3.0 源码工作区、Isaac Sim 6.0.0.1 目标配置，以及 Linux 先前报告的 Python 3.12、PyTorch 2.11、RSL-RL 5.4.1 与 NVIDIA RTX 5090；所有实际版本、torchvision 权重和关键接口仍须由 Gate 0 当场只读核验。测试使用 Gymnasium 与 pytest。

**Spec:** `docs/design/2026-08-29-task010-cnn-gru-development-seed-design.md`

## 全局约束

- Linux 实施分支必须从 `origin/feature/TASK-009D0-vectorized-training-infrastructure` 的精确提交 `1533bfa59f3d2d7b2f1769a9890efb354a5e4de6` 创建，分支名固定为 `feature/TASK-010-cnn-gru-development-seed`。不得从 Windows 规划分支承载实现代码。
- 保持 `TASK-009D-0` 已有任务 ID、十二环境配置和默认零可达覆盖拒绝行为不变；新增 TASK-010 子类和独立任务 ID，仅允许提取经过回归测试的共享纯函数或受保护钩子。
- 物理频率保持 240 Hz，Actor、RGB、覆盖与动作边界保持 10 Hz，每个动作推进 24 个物理子步；正式回合恰有 1200 个动作与包含 `C_0` 的 1201 个覆盖点。
- 原始 RGB 保持 `1280×720`，中心裁剪为 `720×720` 后抗锯齿缩放到 `224×224`。不改变相机内参，不新增图像增强、光度随机化、几何随机化或环境随机化。
- 动作模式保持 `HOLD=0`、`MOVE_POS=1`、`MOVE_NEG=2`、`VIEW_POS=3`、`VIEW_NEG=4`、`UP=5`，动作张量保持 `[mode_id, alpha]`。D0 控制器中 MOVE 总力 `0.70–1.40 mg`、VIEW 相机端力 `0.20–0.50 mg`、UP 相机端力 `0.80–1.05 mg` 的映射不得改写。
- 覆盖继续使用 120 度圆形视场、70 mm 距离、第一命中遮挡、胃腔侧法向、不可达掩码与面积加权累计覆盖，不为训练速度更换近似指标。
- Actor 不得读取位姿、速度、接触、胃壁法向、覆盖、覆盖网格、剩余时间、恢复状态、位姿 ID 或 split。训练 rollout 保存 512 维冻结视觉特征，不保存原始 RGB。
- Critic 的 65 维输入严格按设计文档的字段顺序构造并写入机器可读 schema；Actor 与 Critic 不共享可训练层或归一化器。
- 动作固定为六模式 categorical 与五个非 HOLD 条件 Beta 分布。HOLD 的实际 `alpha=0`，不计 Beta 对数概率；联合熵使用模式概率加权的条件 Beta 熵。
- 可达覆盖 `C_0=0` 在 `C_0^{raw}>0` 且 RGB、物理状态均有限时对 TASK-010 有效。不得重采样、删除位姿或改写位姿库来绕过该状态。
- 120 秒任务上限必须报告真正终止，价值目标为零且不 bootstrap。只有采样器人为中断才允许 bootstrap；不得把真实 120 秒终点沿用为 RSL-RL `time_outs`。
- 训练严格同步：十二个环境共同开始、共同结束、共同 reset。GRU 隐状态跨 rollout 携带并在边界 detach，只在真实 reset 掩码处清零。
- 本任务禁止修改 Python 环境、CUDA、驱动、Isaac Lab、Isaac Sim、RSL-RL 或 torchvision 的安装状态。依赖接口不匹配时停止并报告，不得原地升级、降级或重装。
- Gate 4 不运行 `991000` 的 1000 次 PPO 更新。完整开发种子及 250、500、750、1000 次更新验证由用户在 Codex 任务之外独立启动；本次报告必须明确标记其“未执行、未验证”。
- 大型检查点、RGB、视频与实时训练产物写入 `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/`，不得提交 Git。Git 只接收代码、测试、小型配置、schema、计划、设计与文本报告。
- 每个任务都先加入能失败的测试或验证断言，再做最小实现，再运行相关回归，最后独立提交。任何门禁失败都先保存证据，不得用放宽阈值、删除样本或跳过异常来换取通过。

---

## 文件与接口总图

`configs/task010/cnn_gru_development_v1.json` 是模型、PPO、奖励、回合、验证、日志和开发种子预算的唯一冻结配置。`runtime/task010_config.py` 负责严格 schema 校验、规范化配置哈希和外部工件哈希，不允许调用者静默覆盖冻结字段。

`runtime/task010_visual_encoder.py` 负责确定性裁剪、缩放、ImageNet 归一化和冻结 ResNet18 特征提取。`runtime/task010_recovery.py` 负责持续停滞检测、锚点逃逸进度、十秒奖励上限与覆盖恢复锁。`runtime/task010_privileged.py` 负责 65 维 Critic 输入、字段切片、选择性运行归一化与有限性检查。

`learning/task010_distribution.py` 实现六模式与五个条件 Beta 的联合分布。`learning/task010_actor.py` 实现 519 维 Actor 观测到 256 维单层 GRU 的循环策略。`learning/task010_critic.py` 实现独立的 `65→256→256→256→1` ELU Critic。`learning/task010_ppo.py` 与 `learning/task010_runner.py` 只适配 Linux 已安装 RSL-RL 的确切接口，并使用记录的源码哈希防止版本漂移。

`mdp/task010_terms.py` 暴露冻结视觉特征、上一实际动作、65 维特权观测和恢复奖励。`task010_vector_env.py` 继承 D0 生命周期并改写零可达初始覆盖及真实终止语义。`robotarm_magnetic_task010_env_cfg.py` 与 `agents/task010_rsl_rl_ppo_cfg.py` 组成独立 Gym 任务注册。

`train_task010.py` 是唯一训练入口，`validate_task010_checkpoint.py` 和 `summarize_task010_validation.py` 负责固定二十验证位姿。`task010_training_supervisor.py` 提供 `start`、`status`、`resume` 三个公开命令与内部 `_worker` 子命令，并以原子文件记录状态、心跳、标准输出、标准错误和 traceback。

---

### 任务 1：建立实施分支、冻结配置并审计实际依赖接口

**文件：**
- 新建：`configs/task010/cnn_gru_development_v1.json`
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_config.py`
- 新建：`scripts/stomach_coverage/inspect_task010_prerequisites.py`
- 新建：`tests/stomach_coverage/test_task010_config.py`

**接口：**
- 产出：`Task010Config`、`load_task010_config(path: Path) -> Task010Config`、`canonical_config_sha256(config: Task010Config) -> str`。
- 产出：依赖审计 JSON，至少包含 Python、PyTorch、torchvision、RSL-RL、Isaac Lab 版本，GPU 名称，`torchvision.models.ResNet18_Weights.IMAGENET1K_V1` 可用性，以及 RSL-RL 关键类的方法签名和源码 SHA256。
- 固定开发种子配置：`seed=991000`、`num_envs=12`、`rollout_steps=64`、`max_updates=1000`、验证更新 `[250,500,750,1000]`。
- 固定检查点策略：每 50 次更新保存滚动检查点，并永久保留第 250、500、750、1000 次更新的验证检查点。

- [ ] **步骤 1：从精确 D0 提交创建 Linux 实施分支并验证工作区**

```bash
git fetch origin
test "$(git rev-parse origin/feature/TASK-009D0-vectorized-training-infrastructure)" = "1533bfa59f3d2d7b2f1769a9890efb354a5e4de6"
git switch -c feature/TASK-010-cnn-gru-development-seed 1533bfa59f3d2d7b2f1769a9890efb354a5e4de6
git status --short --branch
```

- [ ] **步骤 2：先写配置失败测试，锁定全部数值与禁止覆盖行为**

```python
def test_development_config_is_frozen():
    cfg = load_task010_config(CONFIG_PATH)
    assert (cfg.training.seed, cfg.training.num_envs) == (991000, 12)
    assert (cfg.ppo.rollout_steps, cfg.training.max_updates) == (64, 1000)
    assert cfg.validation.updates == (250, 500, 750, 1000)
    assert cfg.model.resnet_weights == "IMAGENET1K_V1"
    assert cfg.model.actor_observation_dim == 519
    assert cfg.model.critic_observation_dim == 65
    assert cfg.ppo.gamma == 0.999
    assert cfg.ppo.lam == 0.95

def test_unknown_or_overridden_frozen_field_is_rejected(tmp_path):
    raw = json.loads(CONFIG_PATH.read_text())
    raw["augmentation"] = {"random_crop": True}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="augmentation must remain disabled"):
        load_task010_config(path)
```

- [ ] **步骤 3：运行测试并确认因模块和配置尚不存在而失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_config.py -q
```

- [ ] **步骤 4：实现严格配置加载与依赖审计，不修改任何已安装包**

审计脚本必须通过 `importlib.metadata.version`、`inspect.signature`、`inspect.getsource` 和 SHA256 记录以下对象的实际接口：`OnPolicyRunner.learn`、`OnPolicyRunner.save`、`PPO.act`、`PPO.process_env_step`、`PPO.update`、`RolloutStorage.recurrent_mini_batch_generator`、RSL-RL 循环记忆模块以及模型抽象接口。审计结果写入命令行指定的 JSON，不把本机绝对路径写入冻结配置。

- [ ] **步骤 5：运行配置测试与只读依赖审计**

```bash
python -m pytest tests/stomach_coverage/test_task010_config.py -q
python scripts/stomach_coverage/inspect_task010_prerequisites.py --config configs/task010/cnn_gru_development_v1.json --output /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate0/prerequisites.json
```

- [ ] **步骤 6：核验审计结果后提交**

```bash
git add configs/task010 source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_config.py scripts/stomach_coverage/inspect_task010_prerequisites.py tests/stomach_coverage/test_task010_config.py
git commit -m "feat: freeze task010 training contract"
```

### 任务 2：新增 TASK-010 环境并修正初始覆盖与终止语义

**文件：**
- 修改：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/task009d0_vector_env.py`
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/task010_vector_env.py`
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_task010_env_cfg.py`
- 修改：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- 新建：`tests/stomach_coverage/test_task010_environment_contract.py`

**接口：**
- D0 新增受保护钩子：`_initial_reachable_coverage_is_valid(reachable: Tensor, raw: Tensor) -> Tensor`，D0 默认仍要求 reachable 大于零。
- TASK-010 覆盖钩子：仅要求 raw 大于零且观测、物理状态有限。
- TASK-010 任务 ID：`Template-Robotarm-Magnetic-Task010-CNN-GRU-Coverage-Lab-v0`。
- 终止输出：第 1200 步 `terminated=True`、`truncated=False`，`extras["time_outs"]=False`。

- [ ] **步骤 1：先写 D0 不变性、TASK-010 零可达有效和 120 秒真终止测试**

```python
def test_d0_keeps_positive_reachable_c0_requirement():
    env = object.__new__(Task009D0VectorEnv)
    valid = env._initial_reachable_coverage_is_valid(torch.tensor([0.0]), torch.tensor([0.1]))
    assert valid.tolist() == [False]

def test_task010_accepts_zero_reachable_when_raw_is_positive_and_finite():
    env = object.__new__(Task010VectorEnv)
    valid = env._initial_reachable_coverage_is_valid(torch.tensor([0.0]), torch.tensor([0.1]))
    assert valid.tolist() == [True]

def test_task_horizon_is_terminal_not_timeout(task010_env_fixture):
    terminated, truncated, extras = step_to_horizon(task010_env_fixture)
    assert terminated.all().item()
    assert not truncated.any().item()
    assert not extras["time_outs"].any().item()
```

- [ ] **步骤 2：运行聚焦测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_environment_contract.py -q
```

- [ ] **步骤 3：提取最小钩子、实现 TASK-010 子类与独立注册**

不得复制 D0 整个环境。D0 reset 顺序、十个 HOLD 稳定边界、同步批 reset、力清零、位姿库和覆盖几何保持原样。TASK-010 只覆盖初始覆盖有效性和终止语义，并对 RGB、根状态、关节状态、原始覆盖与可达覆盖做逐环境有限性断言。

- [ ] **步骤 4：运行 TASK-010 与全部 D0 环境回归**

```bash
python -m pytest tests/stomach_coverage/test_task010_environment_contract.py tests/stomach_coverage/test_task009d0_environment_contract.py tests/stomach_coverage/test_task009d0_isolation_protocol.py -q
```

- [ ] **步骤 5：提交环境增量**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab tests/stomach_coverage/test_task010_environment_contract.py
git commit -m "feat: add task010 environment lifecycle"
```

### 任务 3：实现确定性图像预处理与冻结 ResNet18 特征缓存

**文件：**
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_visual_encoder.py`
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/task010_terms.py`
- 修改：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py`
- 新建：`tests/stomach_coverage/test_task010_visual_encoder.py`

**接口：**
- `center_crop_circular_rgb(rgb: Tensor) -> Tensor`：输入 `[N,720,1280,3]` 或 `[N,3,720,1280]`，输出 `[N,3,720,720]`。
- `preprocess_task010_rgb(rgb: Tensor) -> Tensor`：输出有限的 `[N,3,224,224]` float32 ImageNet 标准化张量。
- `FrozenResNet18Encoder.forward(rgb: Tensor, frame_ids: Tensor) -> Tensor`：输出 `[N,512]`，同一环境同一帧只编码一次。
- `task010_actor_observation(env) -> Tensor`：拼接 `[visual_feature_512, previous_actual_action_7]`，输出 `[N,519]`。

- [ ] **步骤 1：写预处理、圆形不变形、冻结参数、缓存与无梯度失败测试**

```python
def test_preprocess_preserves_centered_circle_as_circle(circle_batch):
    output = preprocess_task010_rgb(circle_batch)
    assert output.shape == (2, 3, 224, 224)
    assert abs(measured_width(output) - measured_height(output)) <= 1

def test_encoder_is_frozen_and_cached(rgb_batch):
    encoder = FrozenResNet18Encoder()
    first = encoder(rgb_batch, torch.tensor([4, 4]))
    second = encoder(rgb_batch, torch.tensor([4, 4]))
    assert first.shape == (2, 512)
    assert torch.equal(first, second)
    assert encoder.forward_image_count == 2
    assert not any(p.requires_grad for p in encoder.parameters())
    assert not first.requires_grad
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_encoder.py -q
```

- [ ] **步骤 3：实现预处理、权重身份校验和逐环境帧缓存**

加载权重时必须记录 torchvision 版本、权重枚举名和权重文件 SHA256。模块永久 `eval()`，覆盖 `train()` 以阻止 BatchNorm 切回训练模式；所有前向在 `torch.inference_mode()` 中执行。缓存键由 `env_id` 与单调 `frame_id` 组成，reset 后失效，不允许跨帧复用。

- [ ] **步骤 4：运行测试并验证 rollout 观测不含原始图像**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_encoder.py tests/stomach_coverage/test_task010_environment_contract.py -q
```

- [ ] **步骤 5：提交视觉编码路径**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_visual_encoder.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp tests/stomach_coverage/test_task010_visual_encoder.py
git commit -m "feat: add frozen task010 visual encoder"
```

### 任务 4：实现持续停滞恢复状态机与不可刷取奖励

**文件：**
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_recovery.py`
- 修改：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/task010_terms.py`
- 新建：`tests/stomach_coverage/test_task010_recovery.py`

**接口：**
- `RecoveryPhase = NORMAL | ESCAPING | WAITING_COVERAGE | LOCKED`。
- `Task010RecoveryTracker.update(position, rotation, coverage, dt_s) -> RecoveryStep`。
- `RecoveryStep` 必须给出 `phase_one_hot_4`、`stagnation_progress`、`max_escape_progress`、`timer_fraction`、`escape_progress_delta`、`no_progress`、`coverage_resumed`。
- 总奖励：`100*delta_coverage + 0.1*positive_escape_progress_delta - 0.002*no_progress + 0.2*coverage_resumed`。

- [ ] **步骤 1：写五秒触发、物理逃逸、二秒覆盖恢复、十秒上限与锁止测试**

```python
def test_stagnation_needs_all_three_conditions_for_five_seconds():
    tracker = Task010RecoveryTracker(num_envs=1, dt_s=0.1)
    feed_stationary(tracker, seconds=4.9, coverage_gain=0.001)
    assert tracker.phase.item() == RecoveryPhase.NORMAL
    feed_stationary(tracker, seconds=0.1, coverage_gain=0.0)
    assert tracker.phase.item() == RecoveryPhase.ESCAPING

def test_escape_reward_cannot_be_farmed_after_ten_seconds():
    tracker = triggered_tracker()
    rewards = feed_monotonic_escape(tracker, seconds=12.0)
    assert rewards[100:].sum().item() == 0.0
    assert tracker.phase.item() == RecoveryPhase.LOCKED
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_recovery.py -q
```

- [ ] **步骤 3：实现逐环境向量化状态机**

停滞条件必须同时满足五秒窗口内最大位移不超过 `1 mm`、最大姿态变化不超过 `5°`、覆盖增益不超过 `0.001`。逃逸进度以锚点后的位移相对 `3 mm` 与姿态变化相对 `15°` 的较大裁剪值定义，只奖励历史最大值的正增量。到达物理阈值后仅在两秒内覆盖增加至少 `0.001` 时发放一次 `0.2`，随后回到正常状态；十秒后锁止，直至覆盖恢复，不重置回合或 GRU。

- [ ] **步骤 4：运行奖励测试和 D0 覆盖回归**

```bash
python -m pytest tests/stomach_coverage/test_task010_recovery.py tests/stomach_coverage/test_task009d0_coverage_runtime.py -q
```

- [ ] **步骤 5：提交恢复奖励**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_recovery.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/task010_terms.py tests/stomach_coverage/test_task010_recovery.py
git commit -m "feat: add task010 recovery reward"
```

### 任务 5：构造固定 65 维非对称 Critic 观测

**文件：**
- 修改：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task009d0_coverage_runtime.py`
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_privileged.py`
- 修改：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/task010_terms.py`
- 新建：`tests/stomach_coverage/test_task010_privileged.py`

**接口：**
- `Task009D0CoverageRuntime.coverage_grid_3x3x3() -> Tensor`：使用冻结胃局部包围盒与顶点面积权重输出 `[N,27]` 累计可达覆盖。
- `Task010PrivilegedBuilder.build(env, recovery_step) -> Tensor`：输出 `[N,65]`。
- 固定字段：位置 3、旋转 6、线速度 3、角速度 3、上一动作 7、接触标志与强度 2、胃壁法向与有效位 4、可达与原始覆盖 2、覆盖网格 27、剩余时间 1、恢复自动机 8。
- `TASK010_CRITIC_SLICES` 必须将每个字段名映射到不重叠且覆盖 `[0,65)` 的切片。

- [ ] **步骤 1：写维度、顺序、网格面积权重、接触缺失和有限性测试**

```python
def test_privileged_schema_is_exact_and_complete(builder_fixture):
    obs = builder_fixture.build()
    assert obs.shape == (12, 65)
    occupied = [i for s in TASK010_CRITIC_SLICES.values() for i in range(s.start, s.stop)]
    assert occupied == list(range(65))
    assert torch.isfinite(obs).all()

def test_no_contact_has_zero_normal_and_false_valid(builder_fixture):
    obs = builder_fixture.build(net_force=torch.zeros(12, 3))
    assert torch.equal(obs[:, TASK010_CRITIC_SLICES["wall_normal"]], torch.zeros(12, 3))
    assert torch.equal(obs[:, TASK010_CRITIC_SLICES["wall_normal_valid"]], torch.zeros(12, 1))
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_privileged.py -q
```

- [ ] **步骤 3：实现固定局部网格和 Critic builder**

网格单元由冻结胃模型局部坐标包围盒一次性分配，单元覆盖是该单元已覆盖可达面积除以该单元可达总面积；空单元固定为零。接触法向来自 capsule 接触传感器净力的单位向量，强度为净力范数，阈值固定 `1e-4`。旋转使用连续 6D 表示。所有百分比使用 `[0,1]` 而非百分点。

- [ ] **步骤 4：运行 Critic 观测和覆盖回归**

```bash
python -m pytest tests/stomach_coverage/test_task010_privileged.py tests/stomach_coverage/test_task009d0_coverage_runtime.py -q
```

- [ ] **步骤 5：提交非对称观测**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task009d0_coverage_runtime.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_privileged.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/task010_terms.py tests/stomach_coverage/test_task010_privileged.py
git commit -m "feat: add task010 privileged observations"
```

### 任务 6：实现六模式与条件 Beta 联合分布

**文件：**
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/__init__.py`
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_distribution.py`
- 新建：`tests/stomach_coverage/test_task010_distribution.py`

**接口：**
- `Task010ModeBetaDistribution(logits: Tensor, concentration_raw: Tensor)`，其中 logits 为 `[B,6]`，raw 为 `[B,5,2]`。
- `sample() -> Tensor[B,2]`、`mode() -> Tensor[B,2]`、`log_prob(actions) -> Tensor[B,1]`、`entropy() -> Tensor[B,1]`、`parameters_for_storage() -> dict[str, Tensor]`、`kl(other) -> Tensor[B,1]`。
- 浓度参数固定为 `1 + softplus(raw)`；HOLD 样本与确定性输出的 `alpha` 都为零。

- [ ] **步骤 1：写支持域、HOLD、联合对数概率、加权熵和解析 KL 测试**

```python
def test_hold_has_zero_strength_and_no_beta_log_prob():
    dist = fixture_with_forced_mode(0)
    action = dist.mode()
    assert torch.equal(action[:, 1], torch.zeros_like(action[:, 1]))
    assert torch.allclose(dist.log_prob(action), torch.log_softmax(dist.logits, -1)[:, :1])

def test_entropy_is_categorical_plus_probability_weighted_conditional_entropy():
    dist = balanced_fixture()
    expected = dist.categorical.entropy() + (
        dist.categorical.probs[:, 1:] * dist.beta.entropy()
    ).sum(-1)
    assert torch.allclose(dist.entropy().squeeze(-1), expected, atol=1e-6)
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_distribution.py -q
```

- [ ] **步骤 3：实现数值稳定的联合分布**

`alpha` 在计算 Beta `log_prob` 前按 dtype 的安全 epsilon 裁剪，但实际送入环境的非 HOLD 动作仍限制在 `[0,1]`。KL 必须由 categorical KL 加按旧策略模式概率加权的五个 Beta KL 构成，以匹配联合策略分解。任何非有限 logits、浓度、样本、对数概率、熵或 KL 都立即抛出包含张量名称和更新步的异常。

- [ ] **步骤 4：运行分布测试与 CPU 随机性质测试**

```bash
python -m pytest tests/stomach_coverage/test_task010_distribution.py -q
```

- [ ] **步骤 5：提交动作分布**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning tests/stomach_coverage/test_task010_distribution.py
git commit -m "feat: add task010 hybrid action distribution"
```

### 任务 7：实现符合 RSL-RL 接口的循环 Actor

**文件：**
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_actor.py`
- 新建：`tests/stomach_coverage/test_task010_actor.py`

**接口：**
- 输入 519 维：视觉特征 512 与上一实际动作 7。
- 网络：`512→256` 的 LayerNorm 加 SiLU 视觉投影，`7→32` 的 SiLU 动作投影，拼接 288 后 `288→256` LayerNorm 加 SiLU，单层单向 GRU 输入与隐状态均为 256。
- 输出：六模式 logits 与五组 Beta 的十个 raw 参数。
- 实现依赖审计确认的 RSL 模型方法：`forward`、`get_output_log_prob`、`output_distribution_params`、`output_entropy`、`get_kl_divergence`、`get_hidden_state`、`reset`。

- [ ] **步骤 1：写前向形状、循环记忆、reset 掩码、确定性动作与参数隔离测试**

```python
def test_actor_carries_and_resets_hidden_state():
    actor = Task010Actor()
    obs = torch.zeros(12, 519)
    actor(obs, stochastic_output=True)
    carried = actor.get_hidden_state().clone()
    actor.reset(torch.tensor([True] + [False] * 11))
    reset_state = actor.get_hidden_state()
    assert torch.equal(reset_state[:, 0], torch.zeros_like(reset_state[:, 0]))
    assert torch.equal(reset_state[:, 1:], carried[:, 1:])

def test_actor_rejects_privileged_width():
    with pytest.raises(ValueError, match="expected 519"):
        Task010Actor()(torch.zeros(12, 65))
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_actor.py -q
```

- [ ] **步骤 3：按审计签名实现 Actor 和初始化**

视觉、动作、融合层使用正交初始化，增益 `sqrt(2)`，偏置零。GRU 输入权重 Xavier、循环权重按门分块正交、偏置零。模式头增益 `0.01`、偏置零；Beta raw 头增益 `0.01`，两个浓度偏置均为 `log(exp(1)-1)`，使初始浓度接近 2。

- [ ] **步骤 4：运行 Actor、分布和序列一致性测试**

```bash
python -m pytest tests/stomach_coverage/test_task010_actor.py tests/stomach_coverage/test_task010_distribution.py -q
```

- [ ] **步骤 5：提交循环 Actor**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_actor.py tests/stomach_coverage/test_task010_actor.py
git commit -m "feat: add recurrent task010 actor"
```

### 任务 8：实现独立 Critic 与选择性运行归一化

**文件：**
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_critic.py`
- 新建：`tests/stomach_coverage/test_task010_critic.py`

**接口：**
- `Task010SelectiveNormalizer` 只更新配置指定的连续字段，不归一化 one-hot 与布尔标志位；覆盖比例、网格比例、剩余时间和恢复进度属于连续字段，必须纳入配置明确的归一化掩码。
- `Task010Critic.forward(obs: Tensor) -> Tensor`：`65→256→256→256→1`，三层隐藏层均使用 ELU。
- `Task010Critic.freeze_normalizer()` 在验证和测试时禁止统计量变化。

- [ ] **步骤 1：写输出形状、选择性归一化、冻结统计和 Actor 参数不共享测试**

```python
def test_critic_shape_and_selective_normalization():
    critic = Task010Critic()
    obs = torch.randn(12, 65)
    values = critic(obs)
    assert values.shape == (12, 1)
    before = critic.normalizer.running_mean.clone()
    critic.freeze_normalizer()
    critic(obs * 2)
    assert torch.equal(before, critic.normalizer.running_mean)

def test_actor_and_critic_share_no_parameter_storage():
    actor, critic = Task010Actor(), Task010Critic()
    assert parameter_data_ptrs(actor).isdisjoint(parameter_data_ptrs(critic))
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_critic.py -q
```

- [ ] **步骤 3：实现 Critic、归一化字段掩码和初始化**

隐藏层正交初始化增益 `sqrt(2)`、偏置零，价值输出层正交初始化增益 1、偏置零。运行方差使用稳定在线更新并设置最小方差，checkpoint 必须保存计数、均值、方差与冻结标志。

- [ ] **步骤 4：运行 Critic 与 65 维 schema 测试**

```bash
python -m pytest tests/stomach_coverage/test_task010_critic.py tests/stomach_coverage/test_task010_privileged.py -q
```

- [ ] **步骤 5：提交 Critic**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_critic.py tests/stomach_coverage/test_task010_critic.py
git commit -m "feat: add asymmetric task010 critic"
```

### 任务 9：接入循环 PPO、正确 bootstrap 与数值诊断

**文件：**
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_ppo.py`
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/agents/task010_rsl_rl_ppo_cfg.py`
- 修改：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/agents/__init__.py`
- 新建：`tests/stomach_coverage/test_task010_ppo.py`

**接口：**
- `Task010PPO` 继承或最小适配已审计的 RSL-RL `PPO`，不复制未经核对的 `main` 分支实现。
- 参数固定：5 epochs、4 minibatches、clip `0.2`、clipped value loss、value coefficient `1.0`、learning rate `3e-4` adaptive、desired KL `0.01`、gradient norm `1.0`、gamma `0.999`、lambda `0.95`、entropy coefficient `0.005`。
- 诊断至少输出：categorical entropy、weighted Beta entropy、joint entropy、joint KL、clip fraction、value loss、surrogate loss、gradient norm、mode probabilities、各非 HOLD 模式的 alpha 均值和标准差、全部有限性状态。

- [ ] **步骤 1：写真终止不 bootstrap、人为中断 bootstrap、序列掩码和混合策略损失测试**

```python
def test_true_task_terminal_has_zero_bootstrap():
    returns = compute_fixture_returns(terminated=True, sampler_interrupted=False, next_value=9.0)
    assert returns[-1].item() == pytest.approx(REWARD_LAST)

def test_sampler_interruption_bootstraps():
    returns = compute_fixture_returns(terminated=False, sampler_interrupted=True, next_value=9.0)
    assert returns[-1].item() == pytest.approx(REWARD_LAST + 0.999 * 9.0)

def test_recurrent_batches_preserve_environment_time_order(storage_fixture):
    for batch in storage_fixture.recurrent_batches(num_mini_batches=4):
        assert_sequence_ids_are_monotonic_within_each_environment(batch)
        assert_reset_mask_starts_new_sequences(batch)
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_ppo.py -q
```

- [ ] **步骤 3：依据 Gate 0 的实际源码哈希实现最小 PPO 适配**

实现必须在启动时比对依赖审计记录的 RSL-RL 版本、关键方法签名和源码 SHA256；不一致就以明确错误停止。rollout 边界仅 detach 隐状态，不清零；真实 reset 才清零。联合 PPO ratio 使用联合对数概率，熵系数乘联合熵。梯度裁剪后记录实际范数，并在优化器 step 前后检查所有参数与梯度有限。

- [ ] **步骤 4：运行 PPO、Actor、分布与终止合同测试**

```bash
python -m pytest tests/stomach_coverage/test_task010_ppo.py tests/stomach_coverage/test_task010_actor.py tests/stomach_coverage/test_task010_distribution.py tests/stomach_coverage/test_task010_environment_contract.py -q
```

- [ ] **步骤 5：提交 PPO 接入**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_ppo.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/agents tests/stomach_coverage/test_task010_ppo.py
git commit -m "feat: add recurrent task010 ppo"
```

### 任务 10：实现训练 runner、完整检查点和 JSONL 日志

**文件：**
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_runner.py`
- 新建：`scripts/stomach_coverage/train_task010.py`
- 新建：`tests/stomach_coverage/test_task010_runner.py`

**接口：**
- `Task010OnPolicyRunner` 强制 `init_at_random_ep_len=False`，保持十二环境同步。
- `save(path)` 保存 Actor、Critic、优化器、自适应学习率状态、当前更新、总转移数、Critic normalizer、Python/NumPy/PyTorch CPU/CUDA RNG、配置快照与哈希、Git commit、依赖审计哈希。检查点不保存用于接续未完成回合的 GRU 隐状态。
- `load(path, *, strict=True)` 在配置、任务 ID、观测 schema、动作 schema、权重身份或依赖哈希不一致时拒绝恢复。
- `metrics.jsonl` 每个 update 一行，`events.jsonl` 记录生命周期事件，均 flush 并 `fsync`。

- [ ] **步骤 1：写同步起点、检查点往返、RNG 恢复、配置漂移拒绝和日志 schema 测试**

```python
def test_checkpoint_roundtrip_restores_rng_and_update(tmp_path):
    runner = runner_fixture(seed=991000)
    path = tmp_path / "update_0002.pt"
    runner.save(path)
    expected = draw_all_rngs()
    restored = runner_fixture(seed=1)
    restored.load(path)
    assert restored.current_update == runner.current_update
    assert restored.actor_hidden_state_is_zero()
    assert draw_all_rngs() == expected

def test_resume_rejects_config_hash_mismatch(tmp_path):
    checkpoint = make_checkpoint(tmp_path, config_hash="a")
    with pytest.raises(ValueError, match="config hash mismatch"):
        runner_fixture(config_hash="b").load(checkpoint)
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_runner.py -q
```

- [ ] **步骤 3：实现项目内 runner 与专用训练入口**

训练入口只接受冻结配置、输出目录、最大更新覆盖值、保存间隔、验证开关和恢复检查点。覆盖值只能缩短 Gate 运行，不得改变写入 manifest 的原始开发配置。恢复从完整 rollout 边界开始：加载训练状态后统一 reset 十二环境，清空 GRU 隐状态和上一动作，并记录额外 reset 次数；不声称逐位复现中断轨迹。每步记录奖励四个分量、覆盖、动作模式、alpha、停滞与恢复事件；每次更新记录 PPO 诊断；完整回合记录 `C_0`、`C_120`、离散 `nAUC_120` 与恢复统计。禁止记录原始 RGB 或视频。

- [ ] **步骤 4：运行 runner 测试与两次 CPU 伪环境更新**

```bash
python -m pytest tests/stomach_coverage/test_task010_runner.py tests/stomach_coverage/test_task010_ppo.py -q
python scripts/stomach_coverage/train_task010.py --config configs/task010/cnn_gru_development_v1.json --output-dir /tmp/task010_cpu_contract --max-updates 2 --save-interval 1 --validation disabled --backend fake
```

- [ ] **步骤 5：提交 runner 和训练入口**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_runner.py scripts/stomach_coverage/train_task010.py tests/stomach_coverage/test_task010_runner.py
git commit -m "feat: add task010 runner and training entry"
```

### 任务 11：实现固定二十位姿的确定性开发验证接口

**文件：**
- 新建：`scripts/stomach_coverage/validate_task010_checkpoint.py`
- 新建：`scripts/stomach_coverage/summarize_task010_validation.py`
- 新建：`tests/stomach_coverage/test_task010_validation.py`

**接口：**
- 验证位姿固定为设计文档列出的二十个 `validation-*` ID，禁止替换、重采样或访问训练、测试 split。
- 十二环境批次执行前 12 个位姿，第二批只启用前 8 行执行剩余位姿；汇总必须恰好二十条唯一记录。
- 确定性策略使用 mode argmax 与对应 Beta 均值，Critic normalizer 冻结，不更新模型或统计量。
- 输出逐位姿 JSONL 与汇总 JSON，包含覆盖、奖励分量、模式占比、alpha、停滞/恢复和失败原因。

- [ ] **步骤 1：写固定 ID、12+8 分批、确定性动作和无状态更新测试**

```python
def test_validation_uses_exact_twenty_pose_ids():
    assert VALIDATION_POSE_IDS == (
        "validation-0006", "validation-0011", "validation-0015", "validation-0017",
        "validation-0019", "validation-0035", "validation-0040", "validation-0042",
        "validation-0045", "validation-0046", "validation-0051", "validation-0058",
        "validation-0060", "validation-0063", "validation-0067", "validation-0068",
        "validation-0069", "validation-0092", "validation-0095", "validation-0097",
    )
    assert validation_batches(VALIDATION_POSE_IDS, 12) == (VALIDATION_POSE_IDS[:12], VALIDATION_POSE_IDS[12:])
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_validation.py -q
```

- [ ] **步骤 3：实现验证与离线汇总，严格拒绝缺失或重复记录**

验证脚本必须在每批 reset 前注入显式验证 pose ID，验证实际返回的 pose ID 与请求一致，并将第二批未使用的四行排除在指标之外。汇总器必须验证 checkpoint hash、配置 hash、二十个 pose ID、每回合 1200 步和所有有限性字段。

- [ ] **步骤 4：运行验证合同测试**

```bash
python -m pytest tests/stomach_coverage/test_task010_validation.py tests/stomach_coverage/test_entry_pose_library.py -q
```

- [ ] **步骤 5：提交验证入口**

```bash
git add scripts/stomach_coverage/validate_task010_checkpoint.py scripts/stomach_coverage/summarize_task010_validation.py tests/stomach_coverage/test_task010_validation.py
git commit -m "feat: add task010 development validation"
```

### 任务 12：实现后台训练监督器与错误完整留存

**文件：**
- 新建：`scripts/stomach_coverage/task010_training_supervisor.py`
- 新建：`tests/fixtures/task010_failure_worker.py`
- 新建：`tests/stomach_coverage/test_task010_supervisor.py`

**接口：**
- 公开命令：`start`、`status`、`resume`；内部命令：`_worker`。
- `start` 创建唯一运行目录，写 `launch_manifest.json`，使用新会话启动 `_worker`，立即返回 run ID、PID 和目录，不等待训练结束。
- `_worker` 将 stdout 与 stderr 同时追加到 `console.log`，保留完整 traceback；维护 `metrics.jsonl`、`events.jsonl`、原子 `status.json`，心跳间隔不超过 60 秒。
- `status` 只读，不向工作进程发信号；必须区分 `starting|running|completed|failed|interrupted|stale|unknown`。
- `resume` 必须从已校验 checkpoint 启动新进程并在 manifest 中记录 parent run、checkpoint hash 与恢复更新。

- [ ] **步骤 1：写后台立即返回、心跳、并发互斥、恢复和受控失败测试**

```python
def test_start_returns_before_worker_finishes(tmp_path):
    result = run_supervisor("start", tmp_path, worker_args=["--sleep", "3"])
    assert result.elapsed_s < 1.5
    assert (result.run_dir / "launch_manifest.json").is_file()

def test_failure_preserves_exit_code_stderr_and_traceback(tmp_path):
    result = run_failure_fixture(tmp_path, exit_code=23)
    status = wait_for_terminal_status(result.run_dir)
    log = (result.run_dir / "console.log").read_text()
    assert status["state"] == "failed"
    assert status["exit_code"] == 23
    assert "TASK010_CONTROLLED_FAILURE" in log
    assert "Traceback" in log
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_supervisor.py -q
```

- [ ] **步骤 3：实现跨 shell 安全的后台监督器**

启动必须使用 `sys.executable` 与参数数组，不拼接 shell 字符串；Linux 使用 `start_new_session=True`。所有 JSON 先写同目录临时文件、flush、`fsync` 后 `os.replace`。PID 存活不能单独证明训练健康，心跳超过配置阈值时状态为 `stale`。同一输出目录已有活动 worker 时拒绝重复启动。不得自动重启无限循环。

- [ ] **步骤 4：运行监督器测试并人工核对运行目录文件**

```bash
python -m pytest tests/stomach_coverage/test_task010_supervisor.py -q
python scripts/stomach_coverage/task010_training_supervisor.py start --config configs/task010/cnn_gru_development_v1.json --output-root /tmp/task010_supervisor_contract --worker-command tests/fixtures/task010_failure_worker.py --worker-arg=--exit-code=23
python scripts/stomach_coverage/task010_training_supervisor.py status --run-dir /tmp/task010_supervisor_contract/latest
```

- [ ] **步骤 5：提交监督器**

```bash
git add scripts/stomach_coverage/task010_training_supervisor.py tests/fixtures/task010_failure_worker.py tests/stomach_coverage/test_task010_supervisor.py
git commit -m "feat: add task010 detached training supervisor"
```

### 任务 13：执行 Gate 1 代码与配置门禁

**文件：**
- 新建：`handoffs/reports/TASK-010-cnn-gru-development-seed-report.md`

**门禁：**
- 全部 TASK-010 CPU 测试通过。
- 全部已有 `tests/stomach_coverage` 回归通过。
- 配置、观测和动作 schema 哈希稳定，项目内无系统包修改。
- 静态检查确认 Actor 源路径不引用任何特权字段，rollout 不存储 RGB。

- [ ] **步骤 1：新建报告骨架并记录精确提交、环境版本与命令**

报告结论表只允许使用 `通过`、`失败`、`未执行`、`未验证`。每个通过项必须链接到 artifact 相对路径并记录 SHA256；不得把计划值写成实测值。

- [ ] **步骤 2：运行完整 CPU 门禁并保存原始输出**

```bash
mkdir -p /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1
python -m pytest tests/stomach_coverage -q --junitxml=/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/pytest.xml 2>&1 | tee /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/pytest.log
python -m compileall source/robotarm_magnetic_lab/robotarm_magnetic_lab scripts/stomach_coverage 2>&1 | tee /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/compileall.log
git diff --check
```

- [ ] **步骤 3：运行 Actor 信息边界和冻结参数审计**

```bash
python scripts/stomach_coverage/inspect_task010_prerequisites.py --config configs/task010/cnn_gru_development_v1.json --output /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/final_prerequisites.json
rg -n "privileged|coverage_grid|remaining_time|recovery_phase|pose_id|split" source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_actor.py
```

任何命中都必须逐条解释为类型名、拒绝检查或测试；若 Actor 前向读取这些字段，Gate 1 失败。

- [ ] **步骤 4：将 Gate 1 结果写入报告并提交**

```bash
git add handoffs/reports/TASK-010-cnn-gru-development-seed-report.md
git commit -m "test: record task010 gate1 evidence"
```

### 任务 14：执行 Gate 2 十二环境 GPU 集成门禁

**文件：**
- 新建：`scripts/stomach_coverage/validate_task010_gpu_integration.py`
- 新建：`tests/stomach_coverage/test_task010_gpu_integration_protocol.py`
- 修改：`handoffs/reports/TASK-010-cnn-gru-development-seed-report.md`

**门禁：**
- 十二环境实际 RGB 输入、512 特征、519 Actor 观测、65 Critic 观测、动作、奖励和 reset 掩码形状正确且有限。
- 同帧只执行一次视觉编码，各环境缓存、覆盖、恢复状态和 GRU 隐状态隔离。
- 连续采集至少三个完整的 64 步序列，证明 rollout 边界只 detach、不清空 GRU；同步真实 reset 时十二行隐状态全部清零。另用不触发模拟器 reset 的合成布尔掩码单元测试证明循环模块只清空掩码指定行。同时证明每动作 24 个物理子步且 RGB 与覆盖同帧。
- 集成运行前后 ResNet18 参数与 BatchNorm 统计逐位一致。
- 至少一个受控零可达 `C_0` 位姿在 raw 覆盖为正时不重采样，并完成有效 reset。
- 1200 步终点为真终止且无 timeout bootstrap。

- [ ] **步骤 1：先写协议测试，要求脚本暴露精确断言与 JSON schema**

```bash
python -m pytest tests/stomach_coverage/test_task010_gpu_integration_protocol.py -q
```

- [ ] **步骤 2：实现集成脚本并限制运行规模**

脚本先运行十二环境十个 HOLD 稳定边界，再连续运行至少三个 64 步序列，其中包含六种模式的固定动作序列和逐行扰动动作，验证环境隔离与 GRU 边界合同；随后只为终止语义运行到 1200 步并执行十二环境同步 reset。脚本不得向同步环境注入单行 reset。脚本在运行前后分别哈希 ResNet18 参数和 BatchNorm buffers。脚本不得训练、不得写 RGB，仅输出小型 JSONL 与汇总 JSON。

- [ ] **步骤 3：在实际 Isaac Lab GPU 环境运行并保存日志**

```bash
python scripts/stomach_coverage/validate_task010_gpu_integration.py --task Template-Robotarm-Magnetic-Task010-CNN-GRU-Coverage-Lab-v0 --num-envs 12 --config configs/task010/cnn_gru_development_v1.json --output-dir /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate2 2>&1 | tee /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate2/console.log
```

- [ ] **步骤 4：离线校验 Gate 2 汇总并回归协议测试**

```bash
python -m pytest tests/stomach_coverage/test_task010_gpu_integration_protocol.py -q
```

- [ ] **步骤 5：记录 Gate 2 结果并提交**

```bash
git add scripts/stomach_coverage/validate_task010_gpu_integration.py tests/stomach_coverage/test_task010_gpu_integration_protocol.py handoffs/reports/TASK-010-cnn-gru-development-seed-report.md
git commit -m "test: pass task010 twelve-env integration gate"
```

### 任务 15：执行 Gate 3 短时学习门禁

**文件：**
- 新建：`scripts/stomach_coverage/validate_task010_short_learning.py`
- 新建：`tests/stomach_coverage/test_task010_short_learning_protocol.py`
- 修改：`handoffs/reports/TASK-010-cnn-gru-development-seed-report.md`

**门禁：**
- 使用开发配置种子派生的门禁种子 `991010`，十二环境，64 步 rollout，执行 8 次 PPO 更新。
- 至少完成采样、GAE、循环 mini-batch、反向传播、优化、检查点保存与严格恢复。
- 所有损失、梯度、参数、联合 KL、熵、动作与价值有限；至少一个 Actor 参数和一个 Critic 参数在优化后发生有限非零变化。
- 该门禁只证明学习链路工作，不声称覆盖改善、收敛或超过随机策略。

- [ ] **步骤 1：先写短时学习汇总 schema 与通过条件测试**

```python
def test_short_learning_requires_real_updates(summary):
    assert summary["seed"] == 991010
    assert summary["num_envs"] == 12
    assert summary["updates_completed"] == 8
    assert summary["actor_parameter_delta_l2"] > 0
    assert summary["critic_parameter_delta_l2"] > 0
    assert summary["all_finite"] is True
    assert summary["resume_verified"] is True
```

- [ ] **步骤 2：运行协议测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_short_learning_protocol.py -q
```

- [ ] **步骤 3：实现短时学习包装脚本**

脚本先运行 4 次更新并保存，再从该检查点恢复运行至第 8 次更新。恢复后必须核对更新号、配置哈希、RNG 状态存在、Critic normalizer 与 GRU 状态可加载。汇总从 `metrics.jsonl` 和 checkpoints 计算，不以控制台文本推测。

- [ ] **步骤 4：在实际 GPU 环境运行 Gate 3**

```bash
python scripts/stomach_coverage/validate_task010_short_learning.py --config configs/task010/cnn_gru_development_v1.json --seed 991010 --num-envs 12 --updates 8 --split-update 4 --output-dir /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate3 2>&1 | tee /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate3/console.log
python -m pytest tests/stomach_coverage/test_task010_short_learning_protocol.py -q
```

- [ ] **步骤 5：记录 Gate 3 结果并提交**

```bash
git add scripts/stomach_coverage/validate_task010_short_learning.py tests/stomach_coverage/test_task010_short_learning_protocol.py handoffs/reports/TASK-010-cnn-gru-development-seed-report.md
git commit -m "test: pass task010 short-learning gate"
```

### 任务 16：执行 Gate 4 后台入口、恢复与失败留痕门禁并收尾

**文件：**
- 修改：`handoffs/reports/TASK-010-cnn-gru-development-seed-report.md`

**门禁：**
- 正常 smoke 使用 `991010`，只执行 2 次更新，`save_interval=1`，验证关闭；`start` 必须立即返回。
- `status` 必须读到终态和 update 2 checkpoint；`resume` 从 update 2 再运行 1 次更新并生成 update 3 checkpoint。
- 受控失败 fixture 以退出码 23 结束，`console.log`、`events.jsonl` 和 `status.json` 完整保留错误标记、stderr、traceback、退出码和最后心跳。
- 不启动 `991000` 的 1000 更新完整开发种子。

- [ ] **步骤 1：启动两次更新的后台 smoke，不在当前命令中等待训练完成**

```bash
python scripts/stomach_coverage/task010_training_supervisor.py start --config configs/task010/cnn_gru_development_v1.json --output-root /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke --seed 991010 --max-updates 2 --save-interval 1 --validation disabled
```

- [ ] **步骤 2：在训练自然结束后执行一次状态查询并核对 update 2 checkpoint**

```bash
python scripts/stomach_coverage/task010_training_supervisor.py status --run-dir /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke/latest
```

若状态仍为 `running`，Linux 执行者可稍后再运行同一只读命令；Codex 不保持长时等待。只有状态为 `completed` 且 checkpoint 哈希可验证时才能进入恢复测试。

- [ ] **步骤 3：从 update 2 后台恢复一轮并查询 update 3 终态**

```bash
python scripts/stomach_coverage/task010_training_supervisor.py resume --run-dir /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke/latest --checkpoint update_0002.pt --additional-updates 1
python scripts/stomach_coverage/task010_training_supervisor.py status --run-dir /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke/latest-resume
```

- [ ] **步骤 4：运行退出码 23 的受控失败 fixture 并核验错误文件**

```bash
python scripts/stomach_coverage/task010_training_supervisor.py start --config configs/task010/cnn_gru_development_v1.json --output-root /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/failure --worker-command tests/fixtures/task010_failure_worker.py --worker-arg=--exit-code=23
python scripts/stomach_coverage/task010_training_supervisor.py status --run-dir /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/failure/latest
python -m pytest tests/stomach_coverage/test_task010_supervisor.py -q
```

- [ ] **步骤 5：完成报告但明确完整开发种子未执行**

报告必须写明：TASK-010 授权范围可在 Gate 1 至 Gate 4 全部通过后标为 `complete`；`991000`、1000 更新、768000 转移及四次固定验证均为 `未执行、未验证`；三个正式训练种子 `991001`、`991002`、`991003` 不属于本报告；七个随机策略没有升级为论文基线或比较门禁。

- [ ] **步骤 6：运行最终回归与仓库卫生检查**

```bash
python -m pytest tests/stomach_coverage -q
git diff --check
git status --short
git ls-files | rg "artifacts/task010|\.pt$|console\.log$|metrics\.jsonl$|events\.jsonl$|status\.json$"
```

最后一个命令必须无输出；若有输出，先移除误加入索引的大型或运行时工件，但不得删除 artifact 目录中的原始证据。

- [ ] **步骤 7：提交最终报告并推送实施分支**

```bash
git add handoffs/reports/TASK-010-cnn-gru-development-seed-report.md
git commit -m "docs: complete task010 implementation gates"
git push -u origin feature/TASK-010-cnn-gru-development-seed
```

---

## Gate 4 之后的独立运行入口

完整开发种子不属于上述实施门禁。Gate 1 至 Gate 4 通过并由用户审阅返回报告后，用户可在独立 Linux 会话中使用同一监督器启动冻结配置的 `991000`、1000 更新训练；该运行的 `start` 命令应立即返回，后续只用 `status` 查询。250、500、750 与 1000 更新的 checkpoint 验证由 worker 或独立验证进程按配置执行，不能改写训练 checkpoint。

完整开发种子返回后，下一独立决策门禁只根据实测墙钟时间、峰值显存、吞吐、有限性、策略坍缩迹象、覆盖轨迹和二十固定验证位姿结果决定正式三种子的预算。正式种子预留为 `991001`、`991002`、`991003`，但不在 TASK-010 实施报告中宣称已经启动或完成。
