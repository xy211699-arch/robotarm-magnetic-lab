# TASK-010视觉依赖性正式结果交接报告

## 结论

- 正式运行`20260907T043738.212135Z-47fe3440`已完成，23/23阶段成功且无错误。
- update-750主矩阵为4条件×3种子×20位姿，共240回合。
- update-1000敏感性矩阵为2条件×3种子×20位姿，共120回合。
- 360条覆盖曲线均为包含C0的1201个10 Hz点；完整遥测与最终mask均为360份。
- B0−B1的nAUC效应为`+0.13222`，95% CI `[+0.00352,+0.29713]`，单项门禁通过。
- B0−I1效应为`+0.05452`，95% CI `[-0.03661,+0.14213]`，门禁未通过。
- 因两项确认性门禁没有同时通过，不支持“模型已被证明持续依赖当前状态对齐RGB”的强结论。

## Git交接内容

交接目录：

```text
handoffs/artifacts/TASK-010-visual-dependence-results-20260907/
```

目录包含直接分析CSV、逐种子聚合CSV、确认性统计、完整逐位姿覆盖记录、最终mask、完整10 Hz
物理遥测、运行状态/配置/事件和完整SHA-256清单。压缩后约54 MB。

## 外部原始工件

```text
/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_visual_dependence/20260907T043738.212135Z-47fe3440
```

外部目录约1.6 GB，其中约1.5 GB为60个可再生成的供体特征张量。交接包保留其manifest和逐文件
哈希，不提交张量本体；模型检查点也不重复上传。该取舍不会影响覆盖曲线、动作、姿态、奖励、
最终mask或确认性统计的复算。

## 注意事项

`run_metadata/gate_report.json`中的`V3=awaiting_manual_start`来自实现级self-check，不是正式运行
终态。正式完成状态以`run_metadata/status.json`和`artifact_audit.json`为准。
