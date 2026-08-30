# TASK-010 开发种子训练与冻结验证报告

## 结论与边界

- 开发种子 `991000` 已完成：`1000/1000` updates、`768000` transitions、12个并行GPU环境，进程正常退出（exit code 0）。
- update 250、500、750、1000均已在冻结的20个validation位姿上完成120秒确定性终点验证；四份汇总均为20/20且数值有限。
- update 1000当前最好：平均最终可达面积加权覆盖率`97.1051%`，平均总奖励`9.703866`。
- 正式种子`991001/991002/991003`未启动。
- 后加的逐10 Hz覆盖曲线补采尚未完成，不得声称已有四检查点时序曲线；本报告只把已完成的终点验证写为事实。

## 执行身份

- 实施分支：`feature/TASK-010-cnn-gru-development-seed`
- 训练启动时干净HEAD：`7e6533d814eb05c176aba35d507e9c7ebd5acf71`
- 本报告生成前实现HEAD：`29578e9d14a3ce8590e8caa96032b1c173207494`
- 运行ID：`20260829T100131.927476Z-a4f7d6fd`
- 种子：`991000`
- 环境数：12
- 训练设备：`cuda:0`，NVIDIA GeForce RTX 5090，驱动595.84
- Python 3.12.13；PyTorch 2.11.0+cu128；torchvision 0.26.0+cu128；Isaac Lab 12.0.0；RSL-RL 5.4.1
- 配置声明哈希：`c497fc430af729e7e6fb09849330de23fdea45e568b657a766cc5e030e1d19d4`
- 配置文件字节哈希：`88e8b43d905d72a50e1b78eddee94943d4b7809ea5432f8c82acaede918bf65e`

## 训练结果

- 实际运行时长：`19894.586 s`（约5小时31分35秒）。
- 1000/1000条update记录`all_finite=true`；每50 updates保存一次检查点，最终检查点为`update_1000.pt`。
- 首100 updates → 末100 updates：
  - joint entropy：`1.124727 → 0.355123`；策略探索明显收敛但未变为零。
  - value loss：`0.041215 → 0.006724`。
  - value explained variance：`0.883807 → 0.988213`。
  - joint KL：`0.012347 → 0.011659`，保持在目标0.01附近。
  - 吞吐：`38.240 → 38.795 transitions/s`。
- 首120条episode记录 → 末120条episode记录：
  - 平均C0：`7.4355% → 7.5791%`。
  - 平均C120：`92.6065% → 94.0220%`。
  - 平均NAUC120：`88.1635% → 94.0074%`。
  - 平均总奖励：`9.255886 → 9.398597`。
- 末120条动作比例（HOLD/MOVE+/MOVE-/VIEW+/VIEW-/UP）：`3.14% / 15.51% / 19.43% / 4.11% / 15.62% / 42.20%`；没有单一动作达到100%，但UP占比最高。

## 冻结20位姿验证

验证使用同一覆盖定义和120秒/1200控制边界，但Actor采用确定性输出、模型不更新、位姿来自未参与训练的冻结validation集合。

| 检查点 | 位姿 | 数值有限 | 平均最终覆盖率 | 平均总奖励 | 检查点SHA-256 |
|---|---:|---|---:|---:|---|
| update 250 | 20 | 是 | 72.3484% | 7.143058 | `99c6669b13edc8163db7392305a373f7dfca2f3a905e33767c26aaf12f8d851c` |
| update 500 | 20 | 是 | 75.3390% | 7.394642 | `5b8365e4077750cfeb8375e18bb70f247a64ecc88eea9acb09f9eef54148ec80` |
| update 750 | 20 | 是 | 96.2026% | 9.582875 | `8b7b23837f41c1c3bd9e50f3ccb7dfcc61fd1bcf8fc286d8cab838f53b25aaca` |
| update 1000 | 20 | 是 | 97.1051% | 9.703866 | `434eba6171cad0c0bb2c0bf500db677390cc0071cfc5b57b1a1cf37c723c0289` |

验证性能随检查点推进提高，未观察到update 1000相对update 750的终点验证退化。该结论只适用于此开发种子和这20个冻结位姿；尚不能代替三个正式训练种子的统计结论。

## 已发现并修复的验证问题

首次验证把`env.step()`置于`torch.inference_mode()`，第一批12个位姿后生成不可原地reset的覆盖状态，导致第二批8个位姿失败。提交`a8e744d05abfe82cf12a2a066e6951137f3a331e`把无梯度范围缩小到Actor前向，182项胃部回归通过；随后四个检查点均完成20/20验证。

为了绘制四条0–120秒曲线，提交`29578e9d14a3ce8590e8caa96032b1c173207494`新增逐10 Hz轨迹记录与绘图程序。曲线补采进程后来退出，未生成完整轨迹，故图、SVG和CSV均不列为已完成证据。补采启动时清空了update 250的逐位姿JSONL；其已完成的`summary.json`仍保留且哈希如下。update 500/750/1000逐位姿记录仍在。该缺口不改变已保存的四份终点汇总，但若要求逐位姿审计update 250或绘制时序图，必须重新运行该检查点。

## 外部工件清单

大型运行工件不加入Git；交接报告只记录绝对路径、字节数和SHA-256。根目录：

`/mnt/isaac-linux/robotarm_magnetic_lab_task010/artifacts/task010_cnn_gru/development_seed_991000/20260829T100131.927476Z-a4f7d6fd`

| 工件（相对上述根目录） | 字节 | SHA-256 |
|---|---:|---|
| `status.json` | 720 | `ad80f7b808954979aff30848b2e44f97c1b9e3ea40a0f3331d5c916eb297d5ad` |
| `launch_manifest.json` | 2146 | `a9f549eb994981da651cc17c9d76868ba50c04025adcb2d60b23268a3156ef8b` |
| `metrics.jsonl` | 671289 | `dd745d5dd1806bf46c26af7073d23b61e6d598ae21eb6e48cae2032dac2d34e7` |
| `episodes.jsonl` | 367767 | `31bf7b9b99f315fa19f9a2fa42ed9775d6de634fd7584c15e1db7e7ea38aef53` |
| `boundaries.jsonl` | 115019707 | `ceee4febaae36d1e1628e695deb4a5b82cc7920acb96258e7f9ba4aae57bee91` |
| `tensorboard/events.out.tfevents.1787997698.multirobo-System-Product-Name.462595.0` | 925142 | `7e7ed606102a19ab3862a8f50142700ce96e351016d7e5fc69a198782e427d55` |
| `checkpoints/update_0250.pt` | 9102085 | `99c6669b13edc8163db7392305a373f7dfca2f3a905e33767c26aaf12f8d851c` |
| `checkpoints/update_0500.pt` | 9102085 | `5b8365e4077750cfeb8375e18bb70f247a64ecc88eea9acb09f9eef54148ec80` |
| `checkpoints/update_0750.pt` | 9102085 | `8b7b23837f41c1c3bd9e50f3ccb7dfcc61fd1bcf8fc286d8cab838f53b25aaca` |
| `checkpoints/update_1000.pt` | 9102085 | `434eba6171cad0c0bb2c0bf500db677390cc0071cfc5b57b1a1cf37c723c0289` |
| `validation/update_0250/summary.json` | 892 | `dcf4d0a625618e485f241a00b8d27a6dab375722baf573e51a1597c36d14c7ae` |
| `validation/update_0500/summary.json` | 893 | `b24dcea9afbd28d33108a3e3092ee7d3484147286be3a934e9697db62ec914de` |
| `validation/update_0750/summary.json` | 892 | `792f5f561a0c0319da6094b4d89b3bfa79ed8ca936bdfb95e103911e7642f5ec` |
| `validation/update_1000/summary.json` | 892 | `e9133996e0f08bc8f6a74f7006addfe700d0a275b9335b729976f2f3944bbbf5` |
| `validation/update_0500/pose_records.jsonl` | 10778 | `b6675ac62a16fed27826356a8458cb7730fab544a9adac76626a8383cce56572` |
| `validation/update_0750/pose_records.jsonl` | 10723 | `c26e00ff621b45123fd96d441004d0c4cf341100afd0f855078014b0e42e8e3c` |
| `validation/update_1000/pose_records.jsonl` | 10590 | `aa50f96ab2674bccea07d60314d9a8b7e86ee3bebd8a1305cd7656a230c4669b` |
| `validation/coverage_curve_collection.log`（未完成补采） | 23209 | `58d7fd21ceb83af42d52cc7787ed32f9e6d01fad2fb9ada8e5ff25d94d152679` |

## 状态与下一步

- 开发种子训练：`complete`。
- 四检查点20位姿终点验证：`complete`。
- 四检查点逐10 Hz覆盖曲线：`partial`，需要在普通终端重跑，不能由终点值插值。
- 正式多种子训练：`not_started`。

进入正式种子前，建议先补齐四检查点覆盖时序图，并将update 1000与TASK-009C在相同120秒统计口径下比较；不要仅以97.1051%的终点覆盖率宣称策略优于随机基线。
