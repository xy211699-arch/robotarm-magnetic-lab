# TASK-010 CNN+GRU 视觉依赖性验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Linux 执行端若没有这些可选技能，不构成阻塞；必须按相同顺序逐项实施、测试、提交并保留门禁证据。

**Goal:** 在不改变 TASK-010 环境、奖励、PPO、模型参数量和正式训练预算的前提下，实现 Blind-GRU 训练、B0 正常视觉与跨回合错配及首帧冻结推理、配对统计和可由用户人工启动的后台监督流程，用可证伪实验检验 CNN+GRU 是否持续利用与当前状态对齐的 RGB 信息。

**Architecture:** 新增独立的视觉依赖性实验配置和 512 维特征干预层。训练入口只允许 `normal` 与 `blind`，其中 `blind` 仍执行 RGB 采集、预处理和冻结 ResNet18 前向，再在可训练视觉投影前置零。验证入口在现有 519 维 Actor 观测的前 512 维实施 `normal`、`blind`、`donor` 或 `first_frame` 干预，后 7 维始终保留目标环境的上一实际动作。监督器串行调度三次 B1 训练、update 750 主矩阵、update 1000 敏感性矩阵和分层配对汇总，但正式运行只能由用户在 Codex 外执行 `start`。

**Tech Stack:** Python 3.12、PyTorch 2.11、torchvision、Isaac Lab 3.0、Isaac Sim 6.0.0.1、RSL-RL 5.4.1、NumPy、pytest；实际环境与版本继续以现有 TASK-010 依赖审计为准，禁止在本任务内升级或重装依赖。

**Spec:** `docs/superpowers/specs/2026-09-03-task010-visual-dependence-design.md`

## 全局约束

- Linux 实施基线固定为 `origin/feature/TASK-010-three-formal-seed-supervisor` 的提交 `026947fac14368266fde4185091dc0142c0ea905`，新分支固定为 `feature/TASK-010-visual-dependence-validation`。先把本设计与计划提交移植到该分支，不得在 Windows workflow 分支编写实现代码。
- B0 只复用已完成的正式种子 `991001`、`991002`、`991003`。B1 使用同样三个条件隔离后的种子，从 update 0 训练到 1000，并永久保存 update 250、500、750、1000。
- update 750 是唯一主比较检查点；update 1000 只用于预先声明的 B0 对 B1 敏感性分析。不得跨种子或跨条件挑选最佳检查点。
- 训练、验证、正式测试三个 split 保持互斥。本任务实现和 V3 人工运行只读取冻结的 20 个 validation 位姿及其既有单扰动；不得读取 test split、生成 test 结果或据此修改方案。
- B0、B1、I1、I2 保持 120 秒、1200 个 10 Hz 动作和包含 `C0` 的 1201 个覆盖点。评估动作采用最大概率模式和条件 Beta 均值，不调用 Critic，不更新参数或归一化器。
- B1 必须与 B0 使用完全相同的 Actor、Critic、PPO、奖励、并行数、初始化和可训练参数清单。每帧仍执行真实 RGB、预处理和 ResNet18；只允许在 512 维特征进入 `Task010Actor.visual_projection` 前置零。
- I1 与 I2 只使用 B0 检查点。I1 的供体特征必须离线生成并按冻结 validation 顺序循环错排，任何位姿不得接收自身序列；I2 只重复目标回合自己的首帧特征。
- 视觉干预不得覆盖 Actor 观测末尾 7 维目标上一实际动作，不得改变目标环境状态、GRU reset、动作执行、覆盖计算或扰动。
- R3 只引用 TASK-009C 已有的七种随机策略结果作为描述性环境参照。本任务不新增随机实验、不改变随机策略，也不把它纳入两个确认性比较。
- 算法独立单位是三个训练种子。20 个 validation 位姿是种子内配对重复测量，1201 个时间点是回合内纵向记录；不得将 60 个回合或时间点当作独立训练样本。
- 唯一确认性指标为 `nAUC_120`，唯一确认性效应为 `B0-B1` 与 `B0-I1`。`C30`、`C60`、`C120`、阈值时间、I2 和 update 1000 均为次要或敏感性结果。
- 大型检查点、特征库、逐回合数据和日志只能写入 `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/`，不得提交 Git。Git 只接收代码、测试、小型冻结配置、文档和文本报告。
- 每个任务都必须先加入失败测试，再做最小实现，再运行聚焦与相关回归，最后独立提交。不得用删除困难位姿、放宽门槛、自动重试、跳过失败种子或修改配置换取通过。
- 本计划只授权实现、测试、短时冒烟和人工启动入口验收。不得由 Codex、Linux 代理或测试流程启动三次 B1 正式训练；返回状态必须是“实现完成，正式 V3 等待用户人工启动”。

---

## 文件与数据流

`configs/task010/visual_dependence_v1.json` 是本实验唯一冻结的编排配置，只引用既有 `cnn_gru_development_v1.json`，不改写旧配置及其哈希。新配置固定实验条件、三种子、两个检查点、20 个 validation 位姿的顺序、循环供体映射、回合预算、bootstrap 次数与输出完整性要求。

`runtime/task010_visual_intervention.py` 实现纯张量干预和逐环境首帧状态；`mdp/task010_terms.py` 在训练环境观测处应用 `normal` 或 `blind`。`task010_runner.py` 和 `train_task010.py` 把条件身份写入检查点，防止正常与 blind 检查点混用。

`validate_task010_checkpoint.py` 保存或读取每个位姿的离线特征库，并在调用 Actor 前只替换 519 维观测的视觉切片。`summarize_task010_visual_dependence.py` 从逐回合原始工件重算指标和分层 bootstrap。`task010_visual_dependence_supervisor.py` 只负责编排、持久状态、心跳、故障暂停和人工继续，不包含训练或统计公式的第二份实现。

---

### 任务 1：建立实施分支并冻结视觉依赖性配置

**文件：**
- 新建：`configs/task010/visual_dependence_v1.json`
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_visual_dependence_config.py`
- 新建：`tests/stomach_coverage/test_task010_visual_dependence_config.py`

**接口：**
- `VisualDependenceConfig`
- `load_visual_dependence_config(path: Path) -> VisualDependenceConfig`
- `canonical_visual_dependence_sha256(config: VisualDependenceConfig) -> str`
- `stamp_visual_dependence_config(path: Path) -> str`

- [ ] **步骤 1：从精确 Linux 基线创建功能分支并纳入设计与计划**

```bash
git fetch origin
test "$(git rev-parse origin/feature/TASK-010-three-formal-seed-supervisor)" = "026947fac14368266fde4185091dc0142c0ea905"
git switch -c feature/TASK-010-visual-dependence-validation 026947fac14368266fde4185091dc0142c0ea905
git cherry-pick 8ea16be
git cherry-pick origin/workflow/TASK-010-cnn-gru-development-seed
test -f docs/superpowers/specs/2026-09-03-task010-visual-dependence-design.md
test -f docs/superpowers/plans/2026-09-03-task010-visual-dependence.md
git status --short --branch
```

第二次 `cherry-pick` 的目标必须是包含本计划的 workflow 远端 HEAD；该提交只增加实施计划与交接合同，其父提交是已经单独移植的设计提交 `8ea16be`。不得重复移植提交，也不得用 merge 把不相关 workflow 历史带入 Linux 功能分支。

- [ ] **步骤 2：先写严格配置的失败测试**

```python
def test_visual_dependence_matrix_is_frozen():
    cfg = load_visual_dependence_config(CONFIG_PATH)
    assert cfg.formal_seeds == (991001, 991002, 991003)
    assert cfg.validation_pose_ids == load_task010_config(BASE_CONFIG).validation.pose_ids
    assert cfg.primary_update == 750
    assert cfg.sensitivity_update == 1000
    assert cfg.training_conditions == ("blind",)
    assert cfg.primary_conditions == ("normal", "blind", "donor", "first_frame")
    assert cfg.sensitivity_conditions == ("normal", "blind")
    assert cfg.episode_steps == 1200
    assert cfg.coverage_points == 1201
    assert cfg.bootstrap_replicates == 10000

def test_donor_mapping_is_derangement_in_frozen_order():
    cfg = load_visual_dependence_config(CONFIG_PATH)
    expected = dict(zip(cfg.validation_pose_ids, cfg.validation_pose_ids[1:] + cfg.validation_pose_ids[:1]))
    assert cfg.donor_pose_by_target == expected
    assert all(target != donor for target, donor in expected.items())

def test_changed_or_unknown_config_field_is_rejected(tmp_path):
    raw = json.loads(CONFIG_PATH.read_text())
    raw["test_pose_ids"] = ["forbidden"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unknown field"):
        load_visual_dependence_config(path)
```

- [ ] **步骤 3：运行测试并确认因模块与配置尚不存在而失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_dependence_config.py -q
```

- [ ] **步骤 4：实现严格加载器和可复现盖章命令**

配置必须包含 `schema_version=1`、旧配置相对路径及旧配置 SHA256、三个种子、20 个 pose ID、循环供体映射、`1200/1201` 预算、条件矩阵、主次检查点、`bootstrap_seed=20260903`、`bootstrap_replicates=10000`、训练停滞阈值 300 秒与失败阈值 900 秒。普通加载要求 `config_sha256` 与去掉该字段后的规范 JSON 哈希一致；只有 `stamp_visual_dependence_config` 可以在首次创建时原子写入实际哈希。

- [ ] **步骤 5：盖章配置并运行配置测试与旧配置回归**

```bash
python -m robotarm_magnetic_lab.runtime.task010_visual_dependence_config --stamp configs/task010/visual_dependence_v1.json
python -m pytest tests/stomach_coverage/test_task010_visual_dependence_config.py tests/stomach_coverage/test_task010_config.py -q
```

预期结果是全部通过，且第二次 `--stamp` 不改变文件字节。

- [ ] **步骤 6：提交冻结配置**

```bash
git add configs/task010/visual_dependence_v1.json source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_visual_dependence_config.py tests/stomach_coverage/test_task010_visual_dependence_config.py
git commit -m "feat: freeze task010 visual dependence matrix"
```

---

### 任务 2：实现 B1 特征置零与干预状态机

**文件：**
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_visual_intervention.py`
- 修改：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/task010_terms.py`
- 新建：`tests/stomach_coverage/test_task010_visual_intervention.py`
- 修改：`tests/stomach_coverage/test_task010_visual_encoder.py`

**接口：**
- `VALID_TRAINING_VISUAL_CONDITIONS = ("normal", "blind")`
- `VALID_EVALUATION_VISUAL_CONDITIONS = ("normal", "blind", "donor", "first_frame")`
- `replace_actor_visual_features(actor_observation, replacement) -> Tensor`
- `Task010VisualIntervention.apply(features, *, env_ids=None, donor_features=None) -> Tensor`
- `Task010VisualIntervention.reset(env_ids: Tensor | None = None) -> None`

- [ ] **步骤 1：先写唯一变量、形状和 reset 失败测试**

```python
def test_blind_zeroes_only_visual_slice_after_encoder_forward(fake_env):
    fake_env.cfg.task010_visual_condition = "blind"
    observation = task010_actor_observation(fake_env)
    assert fake_env.encoder.forward_image_count == fake_env.num_envs
    assert torch.equal(observation[:, :512], torch.zeros_like(observation[:, :512]))
    assert torch.equal(observation[:, 512:], fake_env.previous_action_features)

def test_replace_actor_visual_features_preserves_target_previous_action():
    target = torch.randn(3, 519)
    donor = torch.randn(3, 512)
    changed = replace_actor_visual_features(target, donor)
    assert torch.equal(changed[:, :512], donor)
    assert torch.equal(changed[:, 512:], target[:, 512:])

def test_first_frame_is_per_environment_and_resettable():
    state = Task010VisualIntervention("first_frame", num_envs=2, feature_dim=512)
    first = torch.stack((torch.ones(512), torch.full((512,), 2.0)))
    assert torch.equal(state.apply(first), first)
    assert torch.equal(state.apply(torch.full_like(first, 9.0)), first)
    state.reset(torch.tensor([1]))
    next_features = torch.stack((torch.full((512,), 8.0), torch.full((512,), 3.0)))
    output = state.apply(next_features)
    assert torch.equal(output[0], first[0])
    assert torch.equal(output[1], next_features[1])
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_intervention.py tests/stomach_coverage/test_task010_visual_encoder.py -q
```

- [ ] **步骤 3：实现无参数、无梯度、严格形状的干预层**

`task010_actor_observation` 必须先调用现有 `task010_visual_encoder(env)(rgb, frame_ids)`，再读取 `env.cfg.task010_visual_condition`，默认值为 `normal`。训练环境只接受 `normal` 或 `blind`；若出现 `donor` 或 `first_frame`，立即报错并提示这两者只能由验证入口实施。所有替换必须返回新张量，不能原地修改编码器缓存。

- [ ] **步骤 4：证明 B0 与 B1 参数清单和 ResNet 调用数完全一致**

```python
def test_normal_and_blind_have_identical_trainable_parameter_manifest(actor_factory):
    normal = actor_factory("normal")
    blind = actor_factory("blind")
    assert trainable_parameter_manifest(normal) == trainable_parameter_manifest(blind)

def test_normal_and_blind_both_execute_resnet(fake_env_factory):
    counts = []
    for condition in ("normal", "blind"):
        env = fake_env_factory(condition)
        task010_actor_observation(env)
        counts.append(env.encoder.forward_image_count)
    assert counts == [12, 12]
```

- [ ] **步骤 5：运行视觉、Actor 和观测项回归**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_intervention.py tests/stomach_coverage/test_task010_visual_encoder.py tests/stomach_coverage/test_task010_actor.py tests/stomach_coverage/test_task010_environment_contract.py -q
```

- [ ] **步骤 6：提交特征干预实现**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_visual_intervention.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/task010_terms.py tests/stomach_coverage/test_task010_visual_intervention.py tests/stomach_coverage/test_task010_visual_encoder.py
git commit -m "feat: add task010 visual feature interventions"
```

---

### 任务 3：把条件身份写入训练入口和检查点

**文件：**
- 修改：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_runner.py`
- 修改：`scripts/stomach_coverage/train_task010.py`
- 修改：`tests/stomach_coverage/test_task010_runner.py`
- 新建：`tests/stomach_coverage/test_task010_visual_condition_training.py`

**接口：**
- `Task010OnPolicyRunner` 新增仅关键字参数 `experiment_metadata: Mapping[str, object] | None = None`，其余既有参数顺序和默认值不变
- 训练参数：`--visual-condition {normal,blind}`，默认 `normal`
- 检查点字段：`experiment_metadata.visual_condition`、`visual_dependence_config_sha256`、`base_config_sha256`

- [ ] **步骤 1：先写 CLI、检查点身份和兼容性失败测试**

```python
def test_training_cli_accepts_only_normal_and_blind():
    assert parse_args(["--visual-condition", "blind", *required_args()]).visual_condition == "blind"
    with pytest.raises(SystemExit):
        parse_args(["--visual-condition", "donor", *required_args()])

def test_runner_round_trip_preserves_visual_condition(tmp_path, runner_factory):
    runner = runner_factory(experiment_metadata={"visual_condition": "blind", "visual_dependence_config_sha256": "a" * 64})
    path = tmp_path / "model_0001.pt"
    runner.save(path)
    record = torch.load(path, map_location="cpu", weights_only=False)
    assert record["experiment_metadata"]["visual_condition"] == "blind"

def test_blind_resume_rejects_normal_checkpoint(tmp_path, runner_factory):
    normal = runner_factory(experiment_metadata={"visual_condition": "normal"})
    path = tmp_path / "normal.pt"
    normal.save(path)
    blind = runner_factory(experiment_metadata={"visual_condition": "blind"})
    with pytest.raises(RuntimeError, match="visual condition mismatch"):
        blind.load(path, strict=True)
```

- [ ] **步骤 2：运行测试并确认新参数和元数据尚不存在**

```bash
python -m pytest tests/stomach_coverage/test_task010_runner.py tests/stomach_coverage/test_task010_visual_condition_training.py -q
```

- [ ] **步骤 3：实现向后兼容的检查点元数据**

旧 B0 检查点允许缺少 `experiment_metadata`，但只能被明确解释为 `normal`。任何新 B1 检查点必须含 `visual_condition=blind` 和视觉依赖性配置哈希。严格恢复要求当前期望条件与检查点条件一致；不得通过 `strict=False` 恢复正式 B1。

- [ ] **步骤 4：在线程创建环境前注入条件，并保持 fake backend 可测**

Isaac 路径在 `gym.make` 前设置 `cfg.task010_visual_condition=args.visual_condition`。fake backend 不加载 Isaac，但必须把同样的条件与配置哈希写入 manifest、事件和检查点。`--visual-condition` 默认 `normal`，因此既有训练命令和 B0 回归不改变行为。

- [ ] **步骤 5：运行 runner、训练 CLI 与监督器回归**

```bash
python -m pytest tests/stomach_coverage/test_task010_runner.py tests/stomach_coverage/test_task010_visual_condition_training.py tests/stomach_coverage/test_task010_supervisor.py tests/stomach_coverage/test_task010_formal_seed_supervisor.py -q
```

- [ ] **步骤 6：提交条件身份实现**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning/task010_runner.py scripts/stomach_coverage/train_task010.py tests/stomach_coverage/test_task010_runner.py tests/stomach_coverage/test_task010_visual_condition_training.py
git commit -m "feat: bind task010 checkpoints to visual condition"
```

---

### 任务 4：扩展验证器以生成供体库和实施 I1、I2

**文件：**
- 修改：`scripts/stomach_coverage/validate_task010_checkpoint.py`
- 新建：`source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_feature_bank.py`
- 修改：`tests/stomach_coverage/test_task010_validation.py`
- 新建：`tests/stomach_coverage/test_task010_feature_bank.py`

**接口：**
- 验证参数：`--visual-condition {normal,blind,donor,first_frame}`
- 验证参数：`--experiment-config PATH`
- 验证参数：`--save-feature-bank DIR`，只允许 `normal`
- 验证参数：`--donor-bank DIR`，只允许且必须用于 `donor`
- `save_pose_feature_sequence(root, metadata, features) -> Path`
- `load_pose_feature_sequence(root, pose_id, expected_metadata) -> Tensor`

- [ ] **步骤 1：先写特征库完整性和干预时序失败测试**

```python
def test_feature_bank_round_trip_requires_1200_by_512(tmp_path):
    features = torch.randn(1200, 512)
    path = save_pose_feature_sequence(tmp_path, metadata_for("pose-01"), features)
    assert torch.equal(load_pose_feature_sequence(tmp_path, "pose-01", expected_metadata()), features)
    with pytest.raises(ValueError, match="1200, 512"):
        save_pose_feature_sequence(tmp_path, metadata_for("bad"), features[:-1])

def test_donor_validation_uses_cyclic_other_pose_and_target_actions(fake_validation):
    result = fake_validation.run(condition="donor")
    for record in result.records:
        assert record["donor_pose_id"] == result.mapping[record["pose_id"]]
        assert record["donor_pose_id"] != record["pose_id"]
        assert record["previous_action_source"] == "target_environment"

def test_first_frame_repeats_t0_for_exactly_1200_decisions(fake_validation):
    observed = fake_validation.actor_visual_inputs(condition="first_frame")
    assert observed.shape == (1200, 512)
    assert torch.equal(observed, observed[0].expand_as(observed))
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_validation.py tests/stomach_coverage/test_task010_feature_bank.py -q
```

- [ ] **步骤 3：实现带哈希的二进制特征库**

每个位姿保存一个 PyTorch 文件，内容只含 `[1200,512]` float32 CPU 特征和元数据；另存 `manifest.json`。元数据必须包含源 pose ID、训练种子、update、B0 检查点 SHA256、旧配置 SHA256、视觉依赖性配置 SHA256、特征步数、维度和文件 SHA256。加载时任何字段、形状、dtype、非有限值或哈希不一致都立即失败。不得使用 JSON 存储 512 维浮点序列，也不得默认保存原始 RGB。

- [ ] **步骤 4：在 Actor 调用前实施四种条件**

正常条件先从 `observations["policy"][:, :512]` 保存真实特征，再原样调用 Actor。blind 条件将前 512 维置零。donor 条件按当前 `step_index` 读取冻结映射中另一个 pose 的离线特征。first_frame 条件在 reset 后缓存每个目标环境的首个真实特征并在 1200 次决策中重复。四种条件均调用 `replace_actor_visual_features`，因此观测末 7 维只能来自目标环境当前观测。

- [ ] **步骤 5：扩充逐回合审计字段**

每条记录新增 `visual_condition`、`training_seed`、`checkpoint_update`、`donor_pose_id` 或 null、`previous_action_source=target_environment`、`feature_bank_manifest_sha256` 或 null、`experiment_config_sha256`。覆盖曲线继续保存 1201 点并通过范围、有限性和单调性检查。验证器仍只读取冻结 20 个 validation 位姿，不能新增 `--test` 入口。

- [ ] **步骤 6：运行验证器、Actor 和旧摘要回归**

```bash
python -m pytest tests/stomach_coverage/test_task010_validation.py tests/stomach_coverage/test_task010_feature_bank.py tests/stomach_coverage/test_task010_actor.py -q
```

- [ ] **步骤 7：提交推理干预与特征库**

```bash
git add scripts/stomach_coverage/validate_task010_checkpoint.py source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_feature_bank.py tests/stomach_coverage/test_task010_validation.py tests/stomach_coverage/test_task010_feature_bank.py
git commit -m "feat: validate task010 visual alignment interventions"
```

---

### 任务 5：实现配对指标和分层 bootstrap 汇总

**文件：**
- 新建：`scripts/stomach_coverage/summarize_task010_visual_dependence.py`
- 新建：`tests/stomach_coverage/test_task010_visual_dependence_summary.py`

**接口：**
- `episode_metrics(coverage: Sequence[float]) -> Mapping[str, float | None]`
- `hierarchical_paired_bootstrap(rows, *, seed: int, replicates: int) -> Mapping[str, float]`
- `summarize_visual_dependence(run_dir: Path, config: VisualDependenceConfig) -> Mapping[str, object]`

- [ ] **步骤 1：先写 nAUC、未达阈值和配对层级失败测试**

```python
def test_episode_metrics_use_1200_post_action_points():
    curve = np.linspace(0.0, 1.0, 1201)
    metrics = episode_metrics(curve)
    expected_nauc = np.trapezoid(curve, dx=0.1) / 120.0
    assert metrics["nAUC_120"] == pytest.approx(expected_nauc)
    assert metrics["C30"] == pytest.approx(curve[300])
    assert metrics["C60"] == pytest.approx(curve[600])
    assert metrics["C120"] == pytest.approx(curve[1200])

def test_unreached_threshold_is_retained():
    metrics = episode_metrics(np.linspace(0.0, 0.79, 1201))
    assert metrics["time_to_80"] is None
    assert metrics["reached_80"] is False

def test_hierarchical_bootstrap_pairs_pose_within_seed(synthetic_rows):
    effect = hierarchical_paired_bootstrap(synthetic_rows, seed=20260903, replicates=10000)
    assert effect["independent_seed_count"] == 3
    assert effect["paired_pose_count_per_seed"] == 20
    assert effect["ci95_low"] > 0.0
```

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_dependence_summary.py -q
```

- [ ] **步骤 3：实现从原始 1201 点曲线重算指标**

汇总器不得信任验证器预填的派生指标。它必须按 `np.trapezoid(curve, dx=0.1) / 120.0` 从每回合 0 至 120 秒的 1201 点曲线重算 `nAUC_120`，并重算 `C30`、`C60`、`C120`、80/90/95% 达标时间与达标标志。它还必须拒绝重复或缺失的 `(condition, seed, pose_id, update)`。主矩阵必须恰有 `4×3×20=240` 回合，敏感性矩阵必须恰有 `2×3×20=120` 回合。

- [ ] **步骤 4：实现种子外层、配对位姿内层的 10000 次 bootstrap**

每个 bootstrap 重复先有放回抽取三个 seed，再对每个抽中 seed 的 20 个 pose ID 有放回抽取，且同一 pose 的两个条件必须共同抽取。输出三个逐种子均值、总体均值、样本标准差、2.5% 和 97.5% 分位数。只对 `B0-B1` 和 `B0-I1` 生成 `confirmatory=true` 的效应记录，不把 60 个回合标成独立样本。

- [ ] **步骤 5：实现不由单一位姿独占的可执行判据**

对两个确认性效应分别删除一个 pose ID 后重算 20 组 leave-one-pose-out 总体均值。门禁要求三个逐种子均值全部大于零、95% CI 下界大于零、20 个 leave-one-pose-out 均值全部大于零。输出 `all_seed_directions_positive`、`ci_excludes_zero`、`all_leave_one_pose_out_positive` 和总的 `claim_gate_passed`。这只是预注册门禁，不把 `n=3` 的不确定性改写成高把握显著性。

- [ ] **步骤 6：生成稳定的机器可读与论文绘图输入**

必须输出 `condition_metrics.csv`、`paired_episode_differences.csv`、`confirmatory_effects.json`、`mean_curves_10hz.csv` 和 `artifact_audit.json`。I2 与 update 1000 明确标记 `secondary`；未达阈值保留空时间与单独达标比例，不删除记录。

- [ ] **步骤 7：运行统计测试并提交**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_dependence_summary.py -q
git add scripts/stomach_coverage/summarize_task010_visual_dependence.py tests/stomach_coverage/test_task010_visual_dependence_summary.py
git commit -m "feat: summarize paired task010 visual dependence effects"
```

---

### 任务 6：实现人工启动的后台监督器

**文件：**
- 新建：`scripts/stomach_coverage/task010_visual_dependence_supervisor.py`
- 新建：`tests/fixtures/task010_visual_dependence_fake_stage.py`
- 新建：`tests/stomach_coverage/test_task010_visual_dependence_supervisor.py`

**公开命令：**
- `start --config PATH --b0-run-dir PATH [--artifact-root PATH]`
- `status [--run-dir PATH]`
- `watch [--run-dir PATH] [--interval 60]`
- `continue --run-dir PATH`
- 内部命令 `_worker --run-dir PATH`

- [ ] **步骤 1：先写阶段顺序、脱离终端和错误暂停失败测试**

```python
def test_fake_pipeline_has_exact_frozen_stage_order(fake_supervisor):
    run = fake_supervisor.complete()
    assert run.stage_names == expected_stage_names(
        seeds=(991001, 991002, 991003),
        primary=("normal", "blind", "donor", "first_frame"),
        sensitivity=("normal", "blind"),
    )

def test_start_returns_while_worker_remains_alive(fake_supervisor):
    started = fake_supervisor.start(stage_delay=2.0)
    assert started.command_returned
    assert process_is_alive(started.worker_pid)

def test_failure_pauses_without_retry_or_next_stage(fake_supervisor):
    run = fake_supervisor.fail(stage="train_blind_seed_991002")
    assert run.status == "paused_on_error"
    assert run.attempt_count("train_blind_seed_991002") == 1
    assert not run.was_started("train_blind_seed_991003")
```

- [ ] **步骤 2：运行测试并确认监督器尚不存在**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_dependence_supervisor.py -q
```

- [ ] **步骤 3：实现启动预检查和不可变 manifest**

`start` 必须检查工作树无跟踪修改、当前提交、视觉配置与旧配置哈希、B0 三种子的 update 750 和 1000 检查点及 SHA256、磁盘空间、依赖审计、无活动 run，并要求显式 `--b0-run-dir`。运行目录使用 UTC 时间和随机后缀，绝不覆盖已有目录。manifest 保存所有命令、环境、Git HEAD、配置快照、B0 工件清单、种子和阶段图。创建成功后还要原子更新 `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/latest_run_path.txt`，内容为本次绝对运行目录，供后续人工 `continue` 使用。

- [ ] **步骤 4：实现固定串行阶段图**

阶段顺序固定为三个 `train_blind_seed_*`，随后三个 seed 的 update 750 B0 normal 并保存各自供体库，再执行相同 seed 的 B1 blind、B0 donor、B0 first_frame，随后三个 seed 的 update 1000 B0 normal 与 B1 blind，最后运行汇总和工件审计。供体库只能被同一 seed、同一 update 750、同一 B0 检查点的 donor 条件使用。

B1 训练命令模板固定为：

```bash
python3 scripts/stomach_coverage/train_task010.py \
  --config configs/task010/cnn_gru_development_v1.json \
  --visual-condition blind \
  --seed 991001 \
  --max-updates 1000 \
  --save-interval 50 \
  --validation disabled \
  --output-dir "${TASK010V_RUN_DIR}/training/blind/seed_991001"
```

真实监督器对三个种子展开该模板，不允许接受命令行种子覆盖或缩短更新数。测试只能把阶段执行器替换为 fake fixture，不能运行这条正式命令。

- [ ] **步骤 5：实现原子状态、心跳、停滞与人工继续**

后台 worker 每 5 秒原子更新 `status.json`，并追加 `events.jsonl`。状态至少包含阶段、条件、seed、update、pose 进度、运行时间、动态 ETA、worker PID、child PID、心跳年龄、指标年龄和错误摘要。300 秒无新指标标记 `suspected_stall`；900 秒无新指标禁止进入下一阶段但不自动杀死仍存活子进程。PID 消失、非零退出、CUDA OOM、非有限指标、哈希变化或工件缺失进入 `paused_on_error`。

`continue` 只允许用户对 `paused_on_error` 的现有目录显式执行。它复核代码、配置和已完成工件哈希，从最后完整 rollout 或尚未开始的阶段继续；不自动重试、不跳过阶段、不换种子。代码、奖励、模型、配置或非有限训练错误修复后，受影响的 B1 seed 必须从 update 0 重训。

- [ ] **步骤 6：证明 `watch` 是可中断的只读观察器**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_dependence_supervisor.py -q
```

测试必须覆盖 `start` 立即返回、worker 存活、原子写入、五分钟警告、十五分钟暂停、非零退出、`continue`、watch 被 SIGINT 后 worker 仍存活、完成时工件审计失败不能写 `completed`。

- [ ] **步骤 7：运行既有两个监督器回归并提交**

```bash
python -m pytest tests/stomach_coverage/test_task010_visual_dependence_supervisor.py tests/stomach_coverage/test_task010_formal_seed_supervisor.py tests/stomach_coverage/test_task010_supervisor.py -q
git add scripts/stomach_coverage/task010_visual_dependence_supervisor.py tests/fixtures/task010_visual_dependence_fake_stage.py tests/stomach_coverage/test_task010_visual_dependence_supervisor.py
git commit -m "feat: supervise task010 visual dependence experiment"
```

---

### 任务 7：执行 V0 至 V2 短时门禁和全仓回归

**文件：**
- 新建：`scripts/stomach_coverage/validate_task010_visual_dependence_gate.py`
- 新建：`tests/stomach_coverage/test_task010_visual_dependence_gate.py`

- [ ] **步骤 1：先写门禁聚合器失败测试**

```python
def test_gate_rejects_any_missing_evidence(tmp_path):
    evidence = complete_fake_evidence()
    del evidence["v0"]["critic_isolation"]
    with pytest.raises(RuntimeError, match="critic_isolation"):
        validate_gate_evidence(evidence)

def test_gate_cannot_mark_formal_v3_complete():
    evidence = complete_fake_evidence()
    evidence["v3"]["status"] = "completed"
    with pytest.raises(RuntimeError, match="must await manual start"):
        validate_gate_evidence(evidence)
```

- [ ] **步骤 2：实现 V0、V1、V2 的机器可读证据检查**

V0 必须记录 B0/B1 Actor 与 Critic 可训练参数清单、参数总数、ResNet 前向计数、B1 投影输入全零、Actor 观测 schema 和 Critic 隔离。V1 使用不属于正式种子的 `990999` 完成最短 blind 前向、反向、保存与恢复冒烟。V2 用至少两个 validation 位姿的短回合证明循环 donor 映射、first-frame 重复、目标上一动作未变、各条件唯一变量和曲线长度；短回合不得写成正式结果。

- [ ] **步骤 3：运行 CPU 测试和 Linux GPU 短时门禁**

```bash
python -m pytest tests/stomach_coverage -q
python scripts/stomach_coverage/validate_task010_visual_dependence_gate.py \
  --config configs/task010/visual_dependence_v1.json \
  --output /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/gates/gate_report.json
```

预期结果为全仓测试零失败，V0、V1、V2 通过，V3 状态固定为 `awaiting_manual_start`。若 GPU 门禁失败，保存报告并停止，不得进入任务 8 的交付完成状态。

- [ ] **步骤 4：核验工作树只含预期文件并提交**

```bash
git status --short
git diff --check
git add scripts/stomach_coverage/validate_task010_visual_dependence_gate.py tests/stomach_coverage/test_task010_visual_dependence_gate.py
git commit -m "test: gate task010 visual dependence implementation"
```

---

### 任务 8：编写 Linux 回传报告并推送实施分支

**文件：**
- 新建：`docs/TASK010_VISUAL_DEPENDENCE_AUTOMATION.md`
- 新建：`handoffs/reports/TASK-010-visual-dependence-implementation-report.md`
- 修改：`README.md`

- [ ] **步骤 1：写人工操作文档，明确不会自动开始正式实验**

文档必须给出唯一正式启动命令：

```bash
python3 scripts/stomach_coverage/task010_visual_dependence_supervisor.py start \
  --config configs/task010/visual_dependence_v1.json \
  --b0-run-dir /mnt/isaac-linux/robotarm_magnetic_lab_task010/artifacts/task010_cnn_gru/formal_seeds/20260830T124744.667141Z-fcf8b406
```

文档必须给出只读状态、观察和人工恢复命令：

```bash
python3 scripts/stomach_coverage/task010_visual_dependence_supervisor.py status
python3 scripts/stomach_coverage/task010_visual_dependence_supervisor.py watch --interval 60
TASK010V_RUN_DIR="$(cat /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/latest_run_path.txt)"
python3 scripts/stomach_coverage/task010_visual_dependence_supervisor.py continue --run-dir "${TASK010V_RUN_DIR}"
```

- [ ] **步骤 2：写实际实现报告，不填写未经执行的结果**

报告必须记录实施分支基线与最终 HEAD、所有提交、实际依赖版本、测试命令与通过计数、V0 至 V2 工件绝对路径/字节数/SHA256、B0 检查点审计、监督器 fake 故障演练、已知限制和偏离项。状态原文必须是“实现完成，正式 V3 等待用户人工启动”；B1 训练耗时、240 个主矩阵回合、120 个敏感性回合和统计效应均标记为“未执行、无结果”，不能填预计值冒充实测。

- [ ] **步骤 3：运行最终审计**

```bash
python -m pytest tests/stomach_coverage -q
git diff --check
git status --short
```

预期结果是测试零失败、无空白错误、只有文档的预期未提交修改，且 `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/` 下的大型工件未进入 Git。

- [ ] **步骤 4：提交并推送 Linux 实施分支**

```bash
git add docs/TASK010_VISUAL_DEPENDENCE_AUTOMATION.md handoffs/reports/TASK-010-visual-dependence-implementation-report.md README.md
git commit -m "docs: hand off task010 visual dependence launcher"
git push -u origin feature/TASK-010-visual-dependence-validation
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feature/TASK-010-visual-dependence-validation)"
git status --short --branch
```

- [ ] **步骤 5：向 Windows 规划端回传精确证据**

回传内容必须包含本地与远端完整 HEAD、提交列表、测试通过计数、门禁报告 SHA256、人工启动命令、预估 21 至 23 小时、正式实验尚未启动的明确声明，以及任何偏离或阻塞。只有远端 HEAD 一致且回传报告存在时，实施任务才可标记完成；不得把实施完成写成视觉依赖性实验完成。

---

## 实施完成判据

实施完成要求配置、干预层、条件化检查点、供体库、四条件验证、分层汇总、监督器、门禁、测试、文档和远端功能分支全部就绪，同时正式三种子仍未启动。任何正式 B1 seed、任一预定条件、任一 pose、任一原始曲线或哈希审计缺失时，都不能把后续 V3 状态写成 `completed`。

论文结论不由代码验收决定。用户人工运行结束后，只有 `B0-B1` 和 `B0-I1` 的三个逐种子方向均为正、分层 95% CI 均不跨零、全部 leave-one-pose-out 效应仍为正，才允许采用设计文档中的强视觉依赖性主张；否则必须按设计文档的降级结论解释。
