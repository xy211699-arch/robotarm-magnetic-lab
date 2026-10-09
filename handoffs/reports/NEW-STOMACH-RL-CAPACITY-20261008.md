# 新胃RL容量阶段执行报告：P0参照、P1准备与P2独立向量适配

当前状态：`needs_input`（等待用户启动P2 GPU验收，总阶段尚未完成）。2026-10-09用户同意调整顺序为先P2双环境正确性、再P1优化和P3容量。独立向量任务、逐行绑定、部分reset生命周期及验收入口已实现，218项非仿真回归通过；没有GPU双环境结果，不能称已支持/验收N=2。P0原参照、P1优化前登记保留，不要求用户重复采集同类单环境数据；未启用优化候选或测试容量。Windows小包复核仍待明确确认。

此前SSH443失败属于历史记录，随后已核对远端`4625f09af718bd1a4434e1117b1d2cdd953a2902`。本轮代码和报告的最终远端HEAD由推送后终端单独交付，未核验前不冒称已上传。

## 身份与范围

- 用户批准的规划分支：`workflow/new-stomach-rl-capacity-20261008`，完整提交`12e60923ea6b294459712f568bdddfe9c7efb2e8`。
- 代码基准：`3ff631e08ecffd97a3bba78cfbbc923906d5dc01`；原实验实现审计HEAD：`69f393cb95268eb17986f5ee0a2049386d6745aa`。
- 实施分支：`feature/new-stomach-rl-capacity-20261008`；独立目录`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008`。
- 本报告所审计P0实现HEAD：`9c9b78baf35926406c884495d88dce4067f84473`；随后报告/日志提交产生新HEAD，最终远端HEAD在终端单独交付，不以此实现HEAD冒称最终文档HEAD。
- origin为`git@github.com:xy211699-arch/robotarm-magnetic-lab.git`；upstream push仍为DISABLED。旧工作树的审批说明日志未覆盖/清理。
- 未调用子代理；合同提及的superpowers技能不在本会话可用技能中，未安装，采用顺序门禁/TDD。全部GPU/仿真/训练由用户手动启动；本代理仅运行非仿真测试、CLI解析、哈希/CSV复核。

| 阶段 | 当前结果 |
| --- | --- |
| P0证据工具/小包/Linux复核 | pass；29份工件已复制并核验，仍待Windows端确认 |
| P0固定物理参照 | 用户手动运行Single/Chunk均pass；Linux已逐文件/数组/时钟复核 |
| P1热点优化/等价性/新容差 | deferred；准备工作保留，用户批准P2先行，原链路GPU计时/优化尚未运行 |
| P2多环境/部分reset | needs_input；独立向量实现和非仿真测试通过，GPU两环境隔离及N1/N2配对待人工运行 |
| P3容量1/4/8/12/16/20 | not_run；两环境只用于未来P2隔离门禁 |
| R0—R2训练就绪 | not_run；依赖容量回报后冻结批量/预算 |

## 已实施内容与真实结果

1. 新增`export_review_bundle.py`及`runtime/new_stomach_rl_review_bundle.py`：拒绝缺文件、大小/SHA不符、错误实现HEAD、覆盖既有目录、路径穿越或漏项；携带原实验身份与来源路径。Windows复核只读取相对包内文件，Linux元数据明确按POSIX解析，不需要原绝对目录。
2. 导出29份小工件，加原始inventory与bundle manifest，共851541字节（附件父目录的Git属性文件另计）。包含最终各Gate摘要及失败历史、envs_1/envs_2、原计时CSV、129项原回归日志、执行命令、外部依赖/代码哈希清单。不复制大轨迹或checkpoint，不重跑旧224MB实验、不修改旧数据。
3. 从交付CSV直接重算：64个计时步墙钟235.7066897302284秒、平均3.6829170270348186秒、稳态0.27152390147793193 transitions/s；208条总记录的提交100%、缩小投影4.8076923%、HOLD 0%，各timer均与原摘要相等。768000同等采样约785.688966小时，仍仅是**旧D单环境稳态**。旧测量未包含reset成本，新合同`q_valid`尚未测得，其他组/并行数均不推断同速。
4. 新增人工`capture_capacity_reference.py`与只读`MagneticInputTape`：固定train-0003及9D/36D两套20条动作，每模式两个独立reset重放；原模型调用前抓实际pose/速度/先前滤波与coupling时间，原调用后抓raw/filtered wrench，严格240次。另保存真实动力学/关节参考及完整逐10Hz/1Hz可见/累计集合，不从物理终点反推模型输入。没有新磁场模型或优化，没有胶囊主动施力/传送（原reset除外）。启动前核对原30份已执行代码快照、库/资产/掩码；实际GPU PhysX和张量设备必须由运行路径验证，不只看CLI参数。
5. CPU USD构造一套/两套完全相同机构：旧`ball_envelope.audit_stage`一套通过，两套触发原`single-environment rotating geometry audit failed`。这是作用域边界证据，不是GPU双环境隔离通过。
6. TDD：工具缺失时collection错误；新增CSV审计缺失时collection错误；采集fixture最后段与首段相同曾1 failed/3 passed，修正固定测试fixture后通过，未改执行器预算；CLI required参数预解析造成help失败，1 failed/1 passed复现后改为解析完成再验证必填，未启动Kit。最终本轮**145 passed、65 warnings、8.07秒、退出0**；旧129项全部仍通过。

附件路径（已进feature分支，Windows可读取）：

`handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/preflight_review_v1/`

其中`bundle_manifest.json`为23420字节，SHA-256 `670c639a285d3a3414f7203eabced70f74dc68edd775542fd8cd0beb8573a7fe`；`source_inventory.json`保持原53515字节与SHA `6ce46685d8b92950ae0db7a54b741823bdb168d0534d15fb31efefd6e4f98c2f`。原路径、大小、SHA都由manifest直接给出。

初次差异检查把原CSV的CRLF视为trailing whitespace，退出2；没有改CSV。仅对新附件目录设置`-text whitespace=cr-at-eol`，避免Windows自动换行损坏原证据SHA；随后从规划起点检查全部差异退出0。被通用.log忽略的193字节原回归日志显式加入Git；提交前从Git索引读取30份原身份工件（29份+source_inventory），大小/SHA全部匹配。不改原USDA/LFS规则。

## 此前请求的最小共享核心授权（本轮用户已批准实施）

当前复用入口`ActuatorVectorAction._geometry()`依次调用全Stage静态网格审计、Ball包络审计、`NativeMeshClearance`。前者无env作用域；Ball审计要求全场景仅2个旋转mesh/3个joint；原生窄相遍历全Stage机器人，并要求全场景只有一份ASM base_link。新RL逐行对象不能仅通过改N或传不同row索引使这些共享函数正确处理副本，不能简单删除数量断言。

为保留原安全算法而非另写/复制一套，建议只授权以下三个共享接口增加可选`env_root=None`：

```text
TASK/controllers/ball_envelope.py
  audit_stage(stage, env_root=None)
TASK/controllers/mesh_cover_audit.py
  audit_stage_static_meshes(..., env_root=None)
TASK/controllers/native_mesh_clearance.py
  NativeMeshClearance(..., env_root=None)
```

TASK为`source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab`。默认None保持旧全Stage行为及其拒绝规则；新向量适配显式传`/World/envs/env_i`，仅筛选该子树，仍严格核查该行数量、body/joint绑定、闭合网格、球包络、反向包含检查、数值cushion及5mm余量。不得改kernel、网格、参数、停止路径或原单环境assert，不授权旧训练代码变化。

此前P0交付没有修改这些共享文件。本轮用户批准后，仅实施上述三个接口的可选作用域及关联绑定验证；正反边界、env_1/env_10、缺mesh、置换、实例副本及单环境默认等价回归均通过。仍不承诺这三处已足够完成全部P2；若发现其他共享核心需求继续报告，GPU隔离由用户手动验收。

## 实际文件

新增脚本两份：`scripts/new_stomach_rl/export_review_bundle.py`、`capture_capacity_reference.py`；新增runtime两份：`new_stomach_rl_review_bundle.py`、`new_stomach_rl_capacity_reference.py`；新增测试四份：`test_review_bundle.py`、`test_capacity_reference.py`、`test_capacity_reference_cli.py`、`test_capacity_scope_boundary.py`，均位于对应`runtime/`及`tests/new_stomach_rl/`。

P0新增上述小证据目录及其父目录`.gitattributes`、本报告、`docs/NEW_STOMACH_RL_CAPACITY_P0_GUIDE.md`，既有`docs/PROJECT_RUN_LOG.md`追加。本轮仅额外修改已批准的三个几何审计文件，新增`tests/new_stomach_rl/test_scoped_geometry_audits.py`。未修改磁场模型、动作执行器、配置、资产、位姿库、覆盖算法、奖励权重或原预检报告，未安装环境或测试框架。

## 复现命令及未运行内容

工作目录为上述capacity工作树。

```bash
python scripts/new_stomach_rl/export_review_bundle.py export \
  --source_inventory /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008/artifacts/new_stomach_rl/evidence/final_artifact_inventory.json \
  --output_directory handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/preflight_review_v1
# export已运行，目录存在时拒绝覆盖，不再次运行。
python scripts/new_stomach_rl/export_review_bundle.py verify --output_directory handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/preflight_review_v1
python scripts/new_stomach_rl/export_review_bundle.py audit --output_directory handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/preflight_review_v1

env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl tests/stomach_migration tests/magnetic_following \
  tests/stomach_coverage/test_task010_recovery.py -q --disable-warnings

./run_isaaclab.sh -p scripts/new_stomach_rl/capture_capacity_reference.py --help
git diff origin/workflow/new-stomach-rl-capacity-20261008 --check
```

完整人工启动指令、参数/输出说明、pass与interrupted判读在`docs/NEW_STOMACH_RL_CAPACITY_P0_GUIDE.md`。`--help`已实际通过，GPU采集命令仅是实现后的待人工验收入口；不声称运行成功或模型等价容差已测定。当前Python/Isaac环境沿用上轮，没有GPU版本重验或升级。旧图像/物理证据仍是上轮日期，不将本次导出时间当作实验时间。

## 本轮外部工件

完整日志/红灯/复核清单（原失败不删除）：

`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p0_evidence_inventory.json`

4387字节，SHA-256 `a02203a972f551c583a2ec1fcde3e87039f57512c134f4acd1ee219dca81fca0`。

| 完整绝对路径 | 字节 | SHA-256 |
| --- | ---: | --- |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p0_regression_delivery.log` | 273 | `a7d097ef0cce51f370234253ea506570c0e518d1588f04e5da8139ecec52c0d4` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p0_csv_audit.json` | 702 | `63bbe2bc980926332cc9734599e66ee75ead9e7899d224a7550eebbe8b7bf397` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p0_original_sources_check.log` | 2171 | `7b25188af8c913fea2d08903ba9199161c8e0f1a9e2aa019f1838063669666db` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p0_capture_help_final.log` | 3787 | `175a6e13b0e284029ca6ddad4eb82e014dde194524c391edf446e6266b3ca173` |

## 2026-10-08用户授权后的实际实施与回归

本轮起点为`c68dc686dfd1ab9766299a81ba604ace8d0bd098`，审计实现HEAD为`e56a1481612644f83b3f8e768480a734a9d87c7d`，实施分支及工作目录不变。报告/日志提交会再次产生新HEAD，终端单独交付最终哈希。

三处接口增加末尾可选参数`env_root=None`。None仍使用原全Stage遍历，不删除原单副本数量保护；显式作用域使用USD子树遍历（含instance proxies）和`Sdf.Path.HasPrefix`边界，不用字符串前缀误选env_10。非法、缺失或未激活根直接拒绝；Ball joint不能跨环境绑定，静态/原生mesh不能归属作用域外RigidBody。网格证书、闭合性、公共Ball支点、反向包含、数值cushion、停止路径和5mm余量不变。未改旧调用点，未将旧单环境任务伪装成向量任务。

TDD新增测试首先复现24 failed/1 passed（尚不支持env_root），实现后25 passed。扩展到32项时31 passed/1 failed，直接原因为受限环境中Warp CPU内核缓存尝试写只读`/home/multirobo/.cache/warp`；仅为测试进程设置`WARP_CACHE_PATH=/tmp/new-stomach-scope-warp-cache`，没有改内核/算法、安装或调整模拟参数。最终作用域测试32 passed/14 warnings/3.52秒；完整回归**177 passed/65 warnings/8.47秒，退出0**，包含上一轮145项。默认与显式作用域单副本结果、sphere repair结果及实际CPU有符号距离逐元素一致。实例副本、行插入顺序、缺网格/关节、跨行关系、不同pivot、非三角面、非闭合网格的拒绝规则均有测试。旧29份证据包复核仍通过，`git diff --check`通过。

实际完整回归命令（只运行测试，不启动Isaac SimulationApp或GPU任务）：

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  WARP_CACHE_PATH=/tmp/new-stomach-scope-warp-cache \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl tests/stomach_migration tests/magnetic_following \
  tests/stomach_coverage/test_task010_recovery.py -q --disable-warnings
```

作用域测试使用相同环境变量，末尾替换为`tests/new_stomach_rl/test_scoped_geometry_audits.py -q --disable-warnings`。原红灯与缓存失败均保留，不用后续绿灯覆盖。

最后两份通过日志另以原字节提交至`handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/scoped_audit_v1/`（文件名与下表相同），便于Windows端复核；不修改先前冻结的`preflight_review_v1`包及其manifest。

| 完整绝对路径 | 字节 | SHA-256 |
| --- | ---: | --- |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/scoped_audit_20261008/scoped_geometry_red_20261008.log` | 32669 | `cc37da86c5413732ef5283cb4e6bd38dc45e24027dcd28da3199f9d0eb5da972` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/scoped_audit_20261008/scoped_geometry_green_20261008.log` | 112 | `1b899c330f8f17e5cc549f69b1c467335b83f8428c53ae47d568af21ee9b110e` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/scoped_audit_20261008/scoped_geometry_extended_20261008.log` | 3632 | `e38f2d7c6c0874bb50f8485064b865307bc6fbb6d0bfc3195a1ffca98d8c2148` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/scoped_audit_20261008/scoped_geometry_final_20261008.log` | 112 | `ca4b4f100963057df1575960ae34cabfe10f0e297ccba7a2ee392fdaad6f491e` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/scoped_audit_20261008/scoped_geometry_regression_20261008.log` | 273 | `6981f43ea298cf564d8feab3ea13cfd0514fe0db21dc31695c9ef95897624bb0` |

前轮下一步（2026-10-08历史）：Windows核验小包；用户手动采集Single/Chunk固定参照，再依门禁继续P1/P2。共享接口授权已解除，不重复审批。该时点未有真实基线/GPU隔离/容量结果，177项纯回归不能替代这些验收。2026-10-09实际进展以下节为准。

## 2026-10-09：用户参照复核与P1计时交付

以上为前轮历史。此轮起点`c1ef70a9c1593c182df9353a4086bc16ff6f1de3`，审计实现HEAD为`1e0f600384d97ff5651fa7d996e31485cf62e057`（报告提交后的最终远端HEAD另在终端交付）。原用户工作树日志追加已保留。没有启动SimulationApp、GPU实验或训练；仅离线NumPy审计、CPU/fake-worker回归和CLI解析。

| 原始运行 | 模式 | 每次有效物理步/磁力调用 | 两次最大数值差 | C0 | 20秒C10/C1 |
| --- | --- | ---: | ---: | ---: | --- |
| `20261008T152609.305712Z` | Single | 4800 / 4800 | 0 | 0.11854450205701249 | 0.26208506519281605 / 0.2600531414753366 |
| `20261008T153020.049473Z` | Chunk | 4800 / 4800 | 0 | 0.11854450205701249 | 0.26341721943301133 / 0.2612071897943429 |

两次原始summary记录环境、PhysX、相机为cuda:0且GPU Dynamics=true；原磁场计算仍是原有限尺寸Magpylib模型，不宣称全部数值计算在GPU。每模式两个独立reset，各20个真实动作、201 C10、21 RGB和222覆盖集合；共19200条有效物理/磁力记录，不把reset HOLD计入策略预算。每份声明工件大小/SHA匹配，50列物理/33列模型输入/25列输出均有限，物理中的applied wrench与模型输出一致。相同模式两次物理/模型数值、可见/累计mask、C10/奖励、指令历史均精确一致；累计集合为逐帧visible的精确并集，时钟无跳帧。

这些是固定参照的覆盖数值，不是学习性能或A/B/C/D研究结果。Single和Chunk的动作轨迹不同，不能以两行结果声称某种策略优越。

优化前已登记物理、模型输入/输出的atol=0/rtol=0，mask、C10/奖励、指令历史及RGB帧号严格一致，不允许见优化结果后放宽。每模式两次21张RGB哈希中相同0张：哈希不能提供像素误差大小，且P0未保存原像素，**没有登记像素容差/宣称RGB等价**。P1入口额外保存原始RGB供原链路像素重复性复核，不改变相机采集频率或新增渲染。

### 新增实现及没有实施的内容

- `scripts/new_stomach_rl/profile_capacity.py`：audit离线读取两份原参照、拒绝覆盖登记目录；run复用原`capture_capacity_reference.main()`，不复制或替换场景/动作/磁场。原文件未改，只在新入口内临时挂接计时和像素记录，退出恢复方法。
- `runtime/new_stomach_rl_reference_audit.py`：验证工件身份、240/10/1Hz计数、有限性、applied wrench、覆盖并集、指令与奖励重复性；只读取原参照，在未读取优化结果时登记容差。
- `runtime/new_stomach_rl_capacity_profile.py`：开关默认off，inclusive/exclusive CPU标签及真实父关系；外层CUDA完成同步，内层无额外计时同步；reset不计样本，trace步排除稳态，CUDA完成失败不计有效样本。模拟器/模型接口、参数、安全策略均未修改。
- 测试四份：`test_capacity_profile.py`、`test_magnetic_equivalence.py`、`test_capacity_profile_cli.py`、`test_capacity_profile_worker.py`，新增20项；文件名中的equivalence当前只验证原参照登记，不冒称已比较优化模型。
- 中文说明：`docs/NEW_STOMACH_RL_CAPACITY_P1_GUIDE.md`及P0指南衔接；本报告与PROJECT_RUN_LOG追加。

P1 run记录规划、碰撞、磁场总跨度/两方向FT、物理、scene update、10Hz精确覆盖/奖励、第一交点查询、pose-only forward、1Hz RGB/冻结视觉、guide与日志。模型、碰撞等嵌套跨度不再次加到父项；标签只代表CPU跨度，不把异步CUDA排队时间当独立GPU耗时。可选torch trace只采首个真实动作，包含CPU/CUDA事件和拷贝/同步详情，不能将嵌套事件时长相加。完整轨迹/像素/trace有记录开销；吞吐字段明确标诊断范围和reset/记录成本，不能充当生产性能或P3容量。

此轮没有选择或接入优化候选，没有新增磁场近似/改变限幅、阻尼、更新频率、5mm间距或库；不能宣称任何提速收益或模型等价通过。P2/P3及训练就绪阶段仍not_run。下一步是用户按P1指南运行两种原链路热点；计时返回后才决定实际热点的保持语义优化。

### 执行命令与测试证据

实际离线登记命令：

```bash
python scripts/new_stomach_rl/profile_capacity.py audit \
  --single_reference artifacts/new_stomach_rl_capacity/reference/20261008T152609.305712Z \
  --chunk_reference artifacts/new_stomach_rl_capacity/reference/20261008T153020.049473Z \
  --output_directory artifacts/new_stomach_rl_capacity/evidence/p1_preregistration_20261009
```

实际回归沿用前节env命令（`WARP_CACHE_PATH=/tmp/new-stomach-scope-warp-cache`），测试集合仍为`tests/new_stomach_rl tests/stomach_migration tests/magnetic_following tests/stomach_coverage/test_task010_recovery.py -q --disable-warnings`。最终**197 passed、65 warnings、9.58秒、退出0**；CLI help/缺参不启动Kit，伪worker验证两个repeat挂接/恢复、239步故障中止、trace导出故障和伪CUDA完成失败不得通过，不调用GPU。

红灯先复现缺失两个runtime模块及缺CLI，保留collection/入口失败日志。中间11 pass/1 fail为测试fixture改变raw wrench却没有同步applied对应列，修正fixture后工具回归通过，未更改力学模型或放宽断言。最终旧177项仍通过；没有旧控制器/配置/资产变化，`git diff --check`通过。

原始运行完整路径及summary身份：

| 完整绝对路径 | 字节 | SHA-256 |
| --- | ---: | --- |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/reference/20261008T152609.305712Z/summary.json` | 12026 | `674661cfbdfd8024cfd60c4c50f9aad230f20ce658b63c8504d931b26d23cfd6` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/reference/20261008T153020.049473Z/summary.json` | 12024 | `8d9bca307f4907134d9118fdc752b1b8095c64626561cc6af806b4d810a7f680` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p1_preregistration_20261009/tolerance_registration.json` | 19409 | `d27ce726268013163b66bb4d088e2acfa79755c151ac4cbb0d9ce93ed7c6667b` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p1_implementation_20261009/p1_regression_accepted.log` | 273 | `79db1ed5b11cc56826270338c995a807469d6fe7a983df1ebf0fcb976ee78aad` |

登记文件中含全部18份原工件的完整绝对路径、字节及SHA，物理NPZ/JSONL均保留Git外、没有再生成或替换；运行报告中的数据采集HEAD仍为c1ef70a，不以新工具HEAD改写旧实验身份。两份summary、登记文件与最终回归日志的原字节另交付于`handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/p1_reference_v1/`（Windows可读，-text禁用换行转换；先前preflight小包不改）。

全部本轮红灯/绿灯日志在`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p1_implementation_20261009/`，未删除。人工GPU命令及输出目录见P1中文指南，当前尚未运行。Windows旧小包人工确认仍pending，不能标记整个P0/P1或容量任务complete。

上述目录的完整日志清单为`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p1_implementation_20261009/log_inventory.json`，1784字节，SHA-256 `c18d0f3b4c9ec778a0339bdec614ce995439679168574c6d064b739d97995f45`；记录12份日志共8926字节、绝对根目录/完整文件名/每份大小与SHA，已逐条复核。清单的原字节也复制至上述p1_reference_v1附件。

## 2026-10-09用户批准P2先行：独立向量实现交付

本轮基线为`4625f09af718bd1a4434e1117b1d2cdd953a2902`；主适配提交为`2e67b40ddf00eaaa0b09380e9cb735bd68ea4116`，本节审计的最终实现HEAD为`98b84fcdc7d2424b669dd940b861fb352777e790`，位于既有独立feature分支。后续报告/日志提交将产生新HEAD，最终完整远端HEAD在交付终端给出。

**明确的合同顺序偏差**：用户先问为何未到多环境测试，Linux解释P0参照与P1热点的用途，并提出先做P2正确性、后优化；用户本轮回复“可以”批准该顺序调整。因此暂缓P1人工热点采集，保留既有优化前登记，不修改Windows原方案文件的历史表述。P3固定阶梯、全部物理/安全/观察/奖励约束与人工GPU启动边界仍不变。

已实现的新任务为`Template-Robotarm-Magnetic-New-Stomach-RL-Vector-v0`。本轮只新增独立适配/测试并更新文档，**没有再次改任何旧共享模块**：

```text
runtime/new_stomach_rl_vector_rows.py          真实逐行读写、相机绑定、位姿平移
runtime/new_stomach_rl_vector_lifecycle.py     WARMUP/ACTIVE与零C0兼容基线
runtime/new_stomach_rl_vector_runtime.py       每行C10/C1、奖励、视觉缓存
TASK/mdp/new_stomach_rl_vector_action.py       每行原Single/Chunk执行器
TASK/mdp/new_stomach_rl_vector_magnetic.py     每行原有限磁体物理桥
TASK/mdp/new_stomach_rl_vector_terms.py        每行观察与原arm/world安全判定
TASK/new_stomach_rl_vector_env.py              共享物理批次、局部reset、终点接口
TASK/robotarm_magnetic_new_stomach_rl_vector_env_cfg.py
scripts/new_stomach_rl/validate_vector_isolation.py
```

runtime相对PKG、TASK的定义沿用原合同。新增七份测试为`test_vector_action.py`、`test_vector_runtime.py`、`test_vector_isolation.py`、`test_vector_step.py`、`test_vector_magnetic.py`、`test_vector_cli.py`、`test_vector_worker.py`，均在`tests/new_stomach_rl/`。既有217项阶段结果后来增加一个“20秒配对失败立即停机”用例，最终为218项；不以早先结果覆盖最终结果。

实现方式是每个旧单行控制器只接收真实的一行tensor视图，所有关节/wrench写入强制映射到对应父场景env_ids；没有给完整N行数据伪报N=1，也没有删除旧guard。旧0索引仅指已隔离的一行。轨迹/伺服/限位/停止路径直接继承原方法，几何绑定仅用此前批准的env_root接口；每行都有独立模型可变对象、滤波、coupling elapsed、几何checker及停止缓存。磁体双向force/torque计算、限幅、drag、release/ramp及float32滤波全部调用原`LegacyMagneticCollisionBridge.physics_step`，并非复制成新磁力模型或给胶囊直接施力。

胃壁审计根据实际clone平移更新世界绑定；覆盖在原canonical网格/哈希/BVH上查询该行相机的canonical坐标，不改拓扑、掩码或面积权重，不把平移后的新哈希伪称原哈希。原位姿库逐行独立RNG，显式canonical world→local→clone world，四元数仍为XYZW；partial reset不重新播全局seed。

每批只有240次正常物理推进。reset只写入选中行的合法库位姿并清空该行状态，下一正常批次该行执行HOLD、建立C0且`valid_transition=False`；其他ACTIVE行正常执行并产生一个样本。TIMEOUT先保留旧`final_obs/final_mask`，之后进入WARMUP，不额外调用全局step。安全终止直接报错，不用自动reset掩盖。新适配允许合法零C0，仅建立奖励基线；旧正C0规则和四项奖励/5秒状态机不改。

相机按每个共享一秒边界只采集一个完整batch，再关联到各行独立的回合RGB时钟；避免逐行补采反复渲染整批。视觉骨干权重可只读共享，但frame/feature cache独立；编码缓存构造在inference_mode(False)下，避免跨上下文reset产生inference tensor写入错误。环境提供有效样本/回合起点/旧终点掩码，但GRU/PPO训练器消费这些接口属于R0，尚未实施或验收。

### 本轮实际运行与证据

实际执行的完整回归命令（没有创建SimulationApp或GPU仿真）：

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  WARP_CACHE_PATH=/tmp/new-stomach-scope-warp-cache \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl tests/stomach_migration tests/magnetic_following \
  tests/stomach_coverage/test_task010_recovery.py -q --disable-warnings
```

最终`218 passed, 65 warnings in 10.73s`，退出0，较旧197项新增21项。真实旧Single/Chunk schedule/servo方法在假资产和已知安全几何fixture中完成240步，逐行写入/局部reset不污染另一行。磁桥测试使用**假有限模型SI输出**验证实际继承的原函数限幅/drag/ramp/filter和双向调用完全相同，不冒称已做真实磁体积分或GPU动力学等价。完整伪worker跑通两个121批次、240步/逐10Hz集合记录、终点后WARMUP和输出保存；239步或前20秒配对不符时停止、不启动下一段、不调整容差，保留失败摘要/部分轨迹。

初次新增fixture分别因SDK stub装载顺序、缺少假omni.usd、缺少假C1 accumulator.mask出现失败；修正的是CPU fixture，未改变旧安全/物理实现。真实Kit内部mask接口由本机源代码核对为Warp bool数组，新的逐行相机绑定按此使用，不用Torch索引冒充mask。仍须用户GPU执行复核SDK接线，不能用伪worker代替。

实际只读输入核对命令：

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh \
  scripts/new_stomach_rl/validate_vector_isolation.py --check_inputs \
  --pose_manifest /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-coverage-regenerate/configs/new_stomach_v1/entry/pose_library_manifest_v1.json \
  --mask /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/new_stomach_coverage/tube_candidate_full_right_appendage_v1/candidate_face_mask.npz \
  --device cuda:0 --viz none
```

结果`inputs_verified`、`gpu_simulation: not_run`、退出0。原参照两模式、登记文件、资产、1200位姿库、完整右管排除掩码与七份外部参数/模型/URDF/XRDF身份均通过只读校验；没有替换旧数据或生成新库。CLI help/错误环境数可在启动Kit前正常退出；`git diff --check`通过。

| 完整绝对路径 | 字节 | SHA-256 |
| --- | ---: | --- |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p2_implementation_20261009/regression_accepted.log` | 354 | `13572291a6e3a9fe7e3cde79c44733d629f7f6f4e9e137bd6e5e6be67647de00` |
| `/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/evidence/p2_implementation_20261009/input_audit.log` | 4827 | `deeb2b60fd2e73d2b780261c0053bb72dbd0dfe3d7b65345ff392626a1b7c5cd` |

### 待用户运行的真实P2入口与未验证边界

完整命令和输出说明已置于现有中文`docs/NEW_STOMACH_RL_CAPACITY_P1_GUIDE.md`最上方“当前下一步：先验收P2双环境”，历史P1命令保留但暂缓。先Single、pass后再Chunk；各模式一个GPU进程完成对照/反复reset行0两段120秒任务，另有终点后的正常WARMUP，合计244个共享一秒批次；墙钟尚未实测。源骨干CPU有限磁模型依旧逐行调用，不能据此承诺N倍加速。

输出默认在`artifacts/new_stomach_rl_capacity/isolation/<UTC运行ID>/`，每段有`boundaries.jsonl`和`isolation_tape.npz`，最终summary记录代码HEAD、完整命令、资产/输入清单、实际GPU PhysX/环境/相机设备、有效样本、TIMEOUT、19项配对比较和每个外部工件路径/大小/SHA。动态轨迹、覆盖集合仍在Git外。前20秒N1/N2物理/模型配对失败会提前终止，记录原预登记容差与真实误差；clone世界坐标的float32平移差异不能被自动放宽。正常行C10/C1集合、四项奖励、历史和帧号仍须严格相符；原RGB像素哈希不同属于已知renderer重复性，既不把它当作力学故障，也不据此声称像素等价。

**尚未验证**：真实GPU两环境构造/窄相、N1/N2动力学/覆盖等价、反复partial reset长期隔离、相机帧/图像等价及实际GRU/rollout接入；更没有P1优化收益、P3容量或R0—R2结果。当前仅可认为独立适配的软件路径交付，整体`needs_input`。遇到真实安全/同步/等价失败保留日志并报告，不擅自调参、更换库、跳过样本或开始训练。
