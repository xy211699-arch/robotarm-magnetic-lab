# TASK-010视觉依赖性实验交接包

本目录对应正式运行`20260907T043738.212135Z-47fe3440`。运行已完成23/23阶段，包含
update-750主矩阵240回合和update-1000敏感性矩阵120回合。

## 可直接分析的文件

- `summary/condition_metrics.csv`：360个回合的C30、C60、C120、nAUC和阈值时间。
- `summary/mean_curves_10hz.csv`：各条件、种子的10 Hz平均覆盖率曲线。
- `summary/paired_episode_differences.csv`：确认性配对差异。
- `summary/confirmatory_effects.json`：分层bootstrap、逐种子和leave-one-pose-out结果。
- `summary/aggregate_metrics.csv`、`per_seed_metrics.csv`：Linux端复算的聚合表。
- `summary/RESULT_ANALYSIS.md`：中文结果解释。

## 原始结果压缩包

- `validation_records.tar.gz`：360条逐位姿1201点覆盖率曲线、回合摘要和最终覆盖mask。
- `telemetry_10hz.tar.gz`：360个回合的完整10 Hz状态、动作、奖励和覆盖率遥测。
- `feature_bank_manifests.tar.gz`：60个跨回合错配供体特征的身份、形状和SHA-256清单。

压缩包在Linux或Windows的Git Bash/WSL中可使用：

```bash
tar -xzf validation_records.tar.gz
tar -xzf telemetry_10hz.tar.gz
tar -xzf feature_bank_manifests.tar.gz
```

## 未纳入Git的内容

60个供体特征`.pt`张量合计约1.5 GB，是执行donor推理时产生的可再生成中间缓存，不是结果
复算所需数据，因此未写入Git历史。其逐文件SHA-256和元数据均保存在
`feature_bank_manifests.tar.gz`中。B0/B1模型检查点同样未重复上传，其冻结路径和SHA-256保存在
`run_metadata/manifest.json`中。

所有交接文件的哈希见`SHA256SUMS`。
