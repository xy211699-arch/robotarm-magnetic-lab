# TASK-010 开发种子补充验证数据

本目录保存 `seed=991000` 开发种子在冻结20个验证位姿上的四检查点覆盖率补测工件。
本次仅运行验证，没有重新训练或修改检查点。

## 来源

- 运行ID：`20260829T100131.927476Z-a4f7d6fd`
- 补测ID：`validation_curve_final_20260830_082425`
- 验证修复提交：`847fa1f9ac723ef40da04bbdafca7b91759e26fa`
- 数据登记前HEAD：`8d20d8863849ccf83f82d68f3d101080127efb33`
- 每个检查点：20个唯一验证位姿
- 每条轨迹：1201个覆盖率样本（0至120秒，10 Hz）

## 验证结果

| 检查点 | 最终平均覆盖率 | 非单调轨迹 |
|---|---:|---:|
| update 0250 | 62.302258% | 0 |
| update 0500 | 66.807640% | 0 |
| update 0750 | 97.601593% | 0 |
| update 1000 | 92.009726% | 0 |

## 文件说明

- `update_*/coverage_trajectories.jsonl`：逐位姿1201点累计覆盖率轨迹。
- `update_*/pose_records.jsonl`：逐位姿终点覆盖率、奖励、动作统计及配置哈希。
- `update_*/summary.json`：各检查点验证摘要。
- `summary/checkpoint_mean_coverage.csv`：四检查点平均覆盖率时序。
- `summary/checkpoint_mean_coverage.png`、`.svg`：四曲线汇总图。
- `collection.log`：完整补测终端日志。
- `SHA256SUMS`：本目录数据文件的完整性清单。

`update_1000`低于`update_0750`是冻结验证集上的真实性能回落，不是轨迹缺失或单调性错误。
