# TASK-010 CNN+GRU 视觉依赖性验证 Linux 实现报告

## 结论

- 状态：实现完成，正式 V3 等待用户人工启动。
- 实施基线：`origin/feature/TASK-010-three-formal-seed-supervisor` 的
  `026947fac14368266fde4185091dc0142c0ea905`。
- 实施分支：`feature/TASK-010-visual-dependence-validation`。
- 原视觉依赖性实现 HEAD：`8533fde`（提交 `test: add cpu gate self-check`）；本次续接改动的
  完整提交哈希以交付终端信息为准。
- 正式 B1 训练已由独立 V2 流程完成；本监督器现支持审计并复用该结果。update 750 主矩阵、
  update 1000 敏感性矩阵和确认性统计尚未执行、无结果。

## 实现提交

```text
6b3a2bb docs: design TASK-010 visual dependence validation
79f9587 docs: plan TASK-010 visual dependence validation
51493d9 feat: freeze task010 visual dependence matrix
7c69b0f feat: add task010 visual feature interventions
04f0c4f feat: bind task010 checkpoints to visual condition
a563feb feat: validate task010 visual alignment interventions
16534b1 feat: summarize paired task010 visual dependence effects
ec8abc3 feat: supervise task010 visual dependence experiment
62ce21d test: gate task010 visual dependence implementation
8533fde test: add cpu gate self-check
```

## 测试证据

### 聚焦回归

- 配置与旧配置回归：`6 passed`
- 视觉干预、编码器、Actor、环境合同：`22 passed`
- runner、训练 CLI、两个既有监督器：`27 passed, 1 skipped`
- 验证器、特征库、Actor：`17 passed`
- 视觉依赖性汇总：`4 passed`
- 视觉依赖性监督器及既有监督器：`22 passed`
- 门禁校验：`3 passed`

### 全仓回归

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest tests/stomach_coverage -q
```

本次续接后观测结果：`297 passed, 1 skipped, 49 warnings`，退出码 0。与视觉依赖性验证直接
相关的聚焦回归为 `49 passed, 49 warnings`，退出码 0。

## V0-V2 短时门禁

运行命令：

```bash
PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh \
  scripts/stomach_coverage/validate_task010_visual_dependence_gate.py \
  --config configs/task010/visual_dependence_v1.json \
  --output /tmp/task010_visual_dependence_gate_report.json \
  --self-check
```

结果：`status=passed`，V3 状态为 `awaiting_manual_start`。

- 报告路径：`/tmp/task010_visual_dependence_gate_report_20260907.json`
- 字节数：`578`
- SHA-256：`ef3174b9b1bdbcb7afe27553800029b2eface786dd197678c3ca387493a52847`

说明：本门禁是 CPU/实现级 self-check，不是 Isaac Lab GPU 的 V1/V2 长时短冒烟；当前环境
`nvidia-smi` 无法通信，未启动 GPU 验证。

## B0 检查点审计

| seed | update 750 SHA-256 | update 1000 SHA-256 |
|---|---|---|
| 991001 | `d621f1673b3dcb9e114b0510068270131d55bf1a9c36919d63f08c407e4a199a` | `51c7d56f19a81db949c2ee03df3146b2223cabca9dd31d9ae00a71eb6f571587` |
| 991002 | `37a7af729cae2f72a3ad241ceeb1c181d0612ff56d58caaa416183a30c1899a8` | `1b8d5636ce6bc7b0c0163866305bc787cc0c23cdcf9c91883aacdef86231c839` |
| 991003 | `58cae4d113264213d6497184fe6516c14f0498db91443fe9c4714af25d7447f4` | `5d4c48a300910fa3b65b4d82a5fde0116c421f31fc2e5961832bd90b16c12136` |

所有六个检查点均存在，SHA-256 与既有正式工件状态一致。

## 已完成 B1 的复用与续接

- B1 来源运行：
  `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_blind_training_v2/20260905T055703.090461Z-ccadb75b`
- 来源状态：`completed_training`；三个种子的完成审计均为 update 1000。
- 新增 `--b1-run-dir`：启动时同时冻结来源 `status.json` 和 update 750/1000 检查点哈希，后台
  worker 再次校验后将三个训练阶段标记为 `completed/reused`，直接执行 B0 推理。
- 验证阶段显式启用 `cuda:0` 和完整逐边界遥测；`status` 同时报告当前条件、种子和已完成位姿数。

| seed | B1 update 750 SHA-256 | B1 update 1000 SHA-256 |
|---|---|---|
| 991001 | `aa05402a6fddc81643aa666990ee90616c7e5101c48cb4ddf420572083cc848d` | `eb52710a1f01ccd55c018f11ba652bce43a4f1b53b1038b71803f69adea3af40` |
| 991002 | `70209023bd12bafc9ce88e46af80d38b2866905896d543badbb0b0a5e70d619e` | `26e2b177fa7a64e89142476b0540143bf4fc13558bf4680b5790885f9a230dc2` |
| 991003 | `eead286924f1c3bd85de7d060e263fb80c63f4d8fd0367cc2c51228da5539d76` | `9678d875e19ff7e0e3135c9b258080b9047e3aa4b4c93c7f8a4d823dbb7789f2` |

## 偏差与已知限制

- GPU 短时门禁未在 Isaac Lab/GPU 下运行，仅完成 CPU self-check。
- 正式 V3 推理矩阵和统计汇总尚未执行，报告中不填写预计结果；B1 训练结果来自上列已完成、
  已审计的独立运行，不会由本入口重复训练。
- 工作树仍保留用户既有未提交内容：`docs/PROJECT_RUN_LOG.md` 的修改和未跟踪脚本
  `scripts/stomach_coverage/capture_task010_selected_pose_overlays.py`；本任务未覆盖或提交它们。
- `tests` 目录被 `.gitignore` 忽略，新增测试文件通过 `git add -f` 纳入 Git。

## 人工正式启动命令

见 `docs/TASK010_VISUAL_DEPENDENCE_AUTOMATION.md`。
