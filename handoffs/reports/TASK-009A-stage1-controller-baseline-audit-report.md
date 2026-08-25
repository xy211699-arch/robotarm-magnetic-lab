# TASK-009A 第一阶段控制器基线发布与审计报告

## 1. 结论

- 状态：`complete`
- Windows 规划分支：`workflow/TASK-009A-stage1-controller-baseline-audit`
- Windows 规划提交：`58e92c43e442caaec65fced7d488818dba1b0afc`
- Linux 发布分支：`feature/TASK-009-stage1-controller-baseline`
- Linux 控制器提交：`335c5f563da51c50656729db86a7872809c58ada`
- Linux 返回分支：`feature/TASK-009A-stage1-controller-baseline-audit`
- 控制器来源提交：`68586d720752f7499d17e97210178f8267f66164`
- 结果：已发布并验证 240 Hz 物理、10 Hz 动作边界、六模式参数化力控制器；未开始覆盖率、VLM、GRU、PPO、奖励或随机策略工作。

## 2. 合同读取

已完整读取：

- `handoffs/active/TASK-009A-stage1-controller-baseline-audit.md`
- `docs/design/2026-08-25-vlm-gastric-coverage-research-contract-v1.md`

规划分支仅作为合同来源，没有被当作控制器代码基线。

## 3. 执行前取证

实际运行目录为 `/tmp/robotarm-task008-retry`。整理前的完整命令输出如下：

```text
$ git status --short --branch
## HEAD (no branch)
 M docs/PROJECT_RUN_LOG.md
 M scripts/action_layer/inspect_p0_coverage_prerequisites.py
 M scripts/dynamic_force/teleop_dynamic_force_stomach.py
 M scripts/dynamic_force/validate_dynamic_force_stomach.py
 M scripts/dynamic_force_macro/calibrate_validate_table.py
 M scripts/ideal_surface/inspect_ideal_surface_prerequisites.py
 M source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/__init__.py
 M source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py
 M source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/__init__.py
 M source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py
 M source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/dynamic_force_macro_action.py
 M source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/ideal_surface_action.py
 M source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py
 M tests/coverage/test_prerequisite_report.py
 M tests/dynamic_force_macro/test_action_term.py
 M tests/dynamic_force_macro/test_contract.py
 M tests/dynamic_force_macro/test_keyboard.py
 M tests/dynamic_force_macro/test_task_cfg.py
 M tests/ideal_surface/test_preflight_schema.py
?? scripts/parameterized_force/
?? source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/move_displacement.py
?? source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/quaternion_conventions.py
?? source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/parameterized_force.py
?? source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/parameterized_force_action.py
?? source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_parameterized_force_table_env_cfg.py
?? source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/parameterized_force_keyboard.py

$ git rev-parse HEAD
68586d720752f7499d17e97210178f8267f66164

$ git log -1 --oneline --decorate
68586d7 (HEAD, origin/feature/TASK-008-six-action-dynamic-force-controller,
feature/TASK-009-stage1-controller-baseline,
feature/TASK-008-six-action-dynamic-force-controller)
feat: expand dynamic force controller to fourteen levels

$ git diff --stat
 docs/PROJECT_RUN_LOG.md                            | 203 +++++++++++++++++++++
 .../inspect_p0_coverage_prerequisites.py           |  12 +-
 .../dynamic_force/teleop_dynamic_force_stomach.py  |   2 +-
 .../validate_dynamic_force_stomach.py              |  10 +-
 .../calibrate_validate_table.py                    |   3 +-
 .../inspect_ideal_surface_prerequisites.py         |  26 +--
 .../robotarm_magnetic_lab/runtime/__init__.py      |  17 +-
 .../robotarm_magnetic_lab/__init__.py              |  12 ++
 .../robotarm_magnetic_lab/controllers/__init__.py  |  18 ++
 .../robotarm_magnetic_lab/mdp/__init__.py          |   5 +
 .../mdp/dynamic_force_macro_action.py              |   7 +-
 .../mdp/ideal_surface_action.py                    |  10 +-
 .../robotarm_magnetic_lab/teleop/__init__.py       |   8 +
 tests/coverage/test_prerequisite_report.py         |   2 +-
 tests/dynamic_force_macro/test_action_term.py      |  44 +++++
 tests/dynamic_force_macro/test_contract.py         |  35 ++++
 tests/dynamic_force_macro/test_keyboard.py         |  59 +++++-
 tests/dynamic_force_macro/test_task_cfg.py         |  12 ++
 tests/ideal_surface/test_preflight_schema.py       |   2 +-
 19 files changed, 451 insertions(+), 36 deletions(-)

$ git diff --name-only
docs/PROJECT_RUN_LOG.md
scripts/action_layer/inspect_p0_coverage_prerequisites.py
scripts/dynamic_force/teleop_dynamic_force_stomach.py
scripts/dynamic_force/validate_dynamic_force_stomach.py
scripts/dynamic_force_macro/calibrate_validate_table.py
scripts/ideal_surface/inspect_ideal_surface_prerequisites.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/dynamic_force_macro_action.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/ideal_surface_action.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py
tests/coverage/test_prerequisite_report.py
tests/dynamic_force_macro/test_action_term.py
tests/dynamic_force_macro/test_contract.py
tests/dynamic_force_macro/test_keyboard.py
tests/dynamic_force_macro/test_task_cfg.py
tests/ideal_surface/test_preflight_schema.py
```

原工作区同时含其他任务改动。没有 reset、checkout、还原或清理该目录；在独立 worktree
`/tmp/robotarm-task009-baseline` 中，以来源提交为基底，只整理高频参数化控制器的直接文件。
原混合工作区保持原样。

## 4. 发布文件

发布提交包含以下 23 个文件：

```text
scripts/parameterized_force/README.md
scripts/parameterized_force/calibrate_move_displacement.py
scripts/parameterized_force/teleop_table_10hz.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/move_displacement.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/quaternion_conventions.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/parameterized_force.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/parameterized_force_action.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_parameterized_force_table_env_cfg.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/parameterized_force_keyboard.py
tests/dynamic_force_macro/test_action_term.py
tests/dynamic_force_macro/test_contract.py
tests/dynamic_force_macro/test_keyboard.py
tests/dynamic_force_macro/test_task_cfg.py
tests/parameterized_force/conftest.py
tests/parameterized_force/test_baseline_audit.py
tests/runtime/conftest.py
tests/runtime/test_move_displacement.py
tests/runtime/test_quaternion_conventions.py
```

未纳入日志数据、图片、视频、缓存、USD 资产或其他任务代码。

## 5. 动作接口与频率

动作张量固定为 `[mode_id, alpha]`，`alpha` 的有效范围为 `[0, 1]`。离散枚举严格为：

| ID | 模式 |
|---:|---|
| 0 | `HOLD` |
| 1 | `MOVE_POS` |
| 2 | `MOVE_NEG` |
| 3 | `VIEW_POS` |
| 4 | `VIEW_NEG` |
| 5 | `UP` |

- 物理频率：240 Hz。
- Actor/环境动作频率：10 Hz。
- 每动作周期：0.1 s。
- 每动作周期物理子步：24。
- 非 HOLD：24/24 子步持续施力，无等待窗。
- HOLD：24/24 子步零 Actor 力。
- 相邻周期由下一条命令直接覆盖，不插入零力帧。

## 6. 力度范围

实时胶囊质量为 `0.005735 kg`，采用 `g=9.81 m/s²`，所以
`mg=0.05626035 N`。当前配置为：

| 模式 | 范围（mg） | 总力范围（N） | 单端分配 |
|---|---:|---:|---|
| MOVE | 0.70--1.20 | 0.039382245--0.067512420 | 两端各 0.019691123--0.033756210 N |
| VIEW | 0.30--0.90 | 0.016878105--0.050634315 | 仅相机端 |
| UP | 0.70--1.00 | 0.039382245--0.056260350 | 仅相机端 |

MOVE 的 `0.70--1.20 mg` 是用户在本任务前明确冻结的范围。VIEW 与 UP 仍为当前基线范围，
本任务不把一次烟雾测试当作最终物理效果标定。

## 7. 力学实现审计

- MOVE：根据当前胶囊长轴，每个 240 Hz 子步重算水平侧向方向；相机端与另一端施加同向、
  等大小力，各占总力一半。
- VIEW：同一水平侧向定义，只在相机端球心施力。
- UP：只在相机端球心施加世界 `+Z` 点力，非相机端为零。
- HOLD：立即清空永久 Actor wrench。
- MOVE/VIEW 实际路径：端点力经 `equivalent_com_wrench` 变换为严格等价的 COM 合力和力矩，
  再通过 `permanent_wrench_composer.set_forces_and_torques_index` 写入一次。
- UP 实际路径：同一 composer 以 `positions=相机端球心`、`torques=None` 写入单点力，由
  PhysX 根据作用点自然产生力矩。
- 纯函数和源码断言确认不存在“点力 + 等价 COM wrench”重复提交。

## 8. Actor 观测泄漏审计

- 配置静态扫描：Actor policy group 只注册 `rgb`。
- 运行时实际键：`["policy.rgb"]`。
- 未出现真实位姿、速度、接触、胃壁法向、覆盖率、覆盖掩码或 privileged 状态。
- 本基线尚未把上一动作或循环隐状态加入 Actor 观测；二者属于合同允许但非本阶段必需输入。

## 9. 自动化验证

### 9.1 Python 编译

```bash
./run_isaaclab.sh -p -m compileall -q \
  scripts/parameterized_force \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/parameterized_force.py \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/parameterized_force_action.py \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/parameterized_force_keyboard.py
```

结果：通过，退出码 0。

### 9.2 控制器、任务和运行时测试

```bash
./run_isaaclab.sh -p -m pytest -q \
  tests/dynamic_force_macro/test_action_term.py \
  tests/dynamic_force_macro/test_contract.py \
  tests/dynamic_force_macro/test_keyboard.py \
  tests/dynamic_force_macro/test_task_cfg.py \
  tests/parameterized_force/test_baseline_audit.py \
  tests/runtime/test_quaternion_conventions.py \
  tests/runtime/test_move_displacement.py
```

结果：`57 passed, 49 warnings in 1.81s`。警告均为 Isaac Lab/torch 既有弃用警告。

覆盖项：动作枚举、独立力度映射、MOVE 双端分配、VIEW 单端力、UP 相机端世界向上力、
24 子步、完整周期主动施力、HOLD 零力、无重复 composer 写入、RGB-only Actor 观测、
四元数 `xyzw` 约定及位移统计。

## 10. Live smoke

命令：

```bash
./run_isaaclab.sh -p scripts/parameterized_force/teleop_table_10hz.py \
  --headless --device cpu --no-realtime \
  --scripted_actions HOLD:0.0:1,MOVE_POS:0.0:1,MOVE_POS:0.5:1,MOVE_POS:1.0:1,MOVE_NEG:0.5:1,VIEW_POS:0.5:1,VIEW_NEG:0.5:1,UP:0.0:1,UP:0.5:1,UP:1.0:1,HOLD:0.0:1 \
  --max_cycles 11 \
  --output_directory /tmp/task009a-controller-baseline-smoke
```

逐周期摘要：

| 周期 | 模式 | alpha | mg | N | 主动子步 | RGB帧 | 状态 |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | HOLD | 0.0 | 0.00 | 0.000000 | 0/24 | 1→2 | finite |
| 1 | MOVE_POS | 0.0 | 0.70 | 0.039382 | 24/24 | 2→3 | finite |
| 2 | MOVE_POS | 0.5 | 0.95 | 0.053447 | 24/24 | 3→4 | finite |
| 3 | MOVE_POS | 1.0 | 1.20 | 0.067512 | 24/24 | 4→5 | finite |
| 4 | MOVE_NEG | 0.5 | 0.95 | 0.053447 | 24/24 | 5→6 | finite |
| 5 | VIEW_POS | 0.5 | 0.60 | 0.033756 | 24/24 | 6→7 | finite |
| 6 | VIEW_NEG | 0.5 | 0.60 | 0.033756 | 24/24 | 7→8 | finite |
| 7 | UP | 0.0 | 0.70 | 0.039382 | 24/24 | 8→9 | finite |
| 8 | UP | 0.5 | 0.85 | 0.047821 | 24/24 | 9→10 | finite |
| 9 | UP | 1.0 | 1.00 | 0.056260 | 24/24 | 10→11 | finite |
| 10 | HOLD | 0.0 | 0.00 | 0.000000 | 0/24 | 11→12 | finite |

所有周期模拟时间均为 0.1 s（浮点误差范围内），子步索引均严格为 `0..23`。最后 HOLD
没有残留 Actor 力。运行退出码为 0，结束原因 `max_cycles`。

## 11. 外部证据

生成数据未提交进 Git：

| 绝对路径 | 大小（byte） | SHA-256 |
|---|---:|---|
| `/tmp/task009a-controller-baseline-smoke/20260825_094431_735574Z/control_cycles.jsonl` | 19956 | `2e5573105af719d2a61a1e9e2f9c9ec2ea8353f80a29ec69512609fb5764b594` |
| `/tmp/task009a-controller-baseline-smoke/20260825_094431_735574Z/session_summary.json` | 284 | `59b743085d0fb3e8abb36d2107ca3ae7628c29a3d31c44993ab090383afc17d8` |
| `/mnt/isaac-linux/isaacsim/kit/logs/Kit/IsaacLab/3.0/kit_20260825_174425.log` | 699509 | `4ad71bb97be8725fd5c5a692aceca7328164481ac079aa5808648a339188e104` |

## 12. 已确认事实、未验证项与偏离项

### 已确认事实

- 发布分支和完整提交可由 origin 获取。
- 六模式、连续 alpha、三组独立范围、力学分配和作用点符合合同。
- 动作边界为 10 Hz，每周期恰好 24 个 240 Hz 子步。
- 非 HOLD 全周期持续施力，HOLD 全周期零力，周期切换无旧力残留。
- Actor 运行时观测只有 RGB；短序列所有状态数值有限，RGB 帧持续推进。

### 未验证信息

- 本任务没有把单次位移或倾角作为力度有效性验收，也未冻结 VIEW/UP 的最终实验范围。
- 未在胃部环境验证动作结果分布；该内容属于后续阶段。
- 未实现上一动作、GRU 隐状态、面积覆盖、奖励、VLM 或 PPO。
- live smoke 未设置随机种子，因此证明接口与数值稳定性，不证明轨迹可重复性。

### 偏离项

- 整理前源工作区混有其他任务修改。为避免覆盖、丢弃或错误提交，使用独立 worktree 提取
  可明确归属高频控制器的文件；原工作区未改变。该安全隔离不改变控制律，但应作为来源
  整理过程偏离记录。
- live smoke 使用 CPU 物理设备并启用 RTX 相机；本合同未要求 CUDA 物理性能验收。

## 13. 后续边界

后续面积覆盖、入口区域、同步回合环境和随机策略时间预算实验只能从：

```text
feature/TASK-009-stage1-controller-baseline
335c5f563da51c50656729db86a7872809c58ada
```

继续开发，不应从旧 TASK-008 宏动作分支推测或重写该控制器。
