# 新胃RL容量阶段执行报告：P0交付与已批准的逐环境几何审计

当前状态：`partial`。用户本轮明确“同意改动”，已实施此前请求的三处可选`env_root`接口；共享审计授权阻塞已解除。P0软件/小证据包已交付，真实配对参照及Windows确认仍待人工执行。尚未开始热点优化、完整向量任务或容量GPU实验，不将作用域CPU回归或旧吞吐当作GPU多环境能力。

上一轮远端交付状态：本地实施提交已生成，但SSH443推送origin两次均返回`Connection closed by 198.18.0.5 port 443`和Git退出128，未更换凭据或改推upstream。本轮最终远端交付结果及完整HEAD在终端单独给出；未确认推送成功前，本文的“交付”仅指本地已提交工件。

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
| P0固定物理参照 | not_run；采集入口已实现，GPU实际运行未验证 |
| P1热点优化/等价性/新容差 | not_run；不在缺少配对参照时试调参数或宣称提速 |
| P2多环境/部分reset | not_run；三处作用域接口已批准并通过CPU回归，原guard仍保留；完整向量隔离未实现/未运行 |
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

下一步：Windows核验小包；用户按中文指南手动采集Single/Chunk固定参照并返回摘要，然后依门禁继续P1/P2。共享接口授权已解除，不再要求重复审批同一改动。当前仍为partial：没有真实基线/GPU隔离/容量结果，不能以177项纯回归替代这些验收或启动正式训练。
