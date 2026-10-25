# Handoff report contract

## 当前报告导航

新胃主线依次查阅 [容量报告](NEW-STOMACH-RL-CAPACITY-20261008.md)、[训练就绪](NEW-STOMACH-RL-TRAINING-READINESS-20261008.md) 与 [性能短测](NEW-STOMACH-RL-PERFORMANCE-500X32-20261010.md)，后续预算修订见 [R2指南](../../docs/NEW_STOMACH_RL_R2_GUIDE.md) 最后“最终确认”及运行日志。当前证据摘要见 [STATUS](../../docs/STATUS.md)，不以旧报告中暂时状态推断现在正在运行的作业。

新报告可使用 [中文模板](../../docs/templates/report.md)，完整日志及数据留Git外，精选附件继续保存在handoffs/attachments。下面的证据字段合同保持有效，状态仅适用于实际报告范围。

Every report must use exactly one status: `complete`, `partial`, `needs_decision`, or `blocked`.

`complete` means every authorized requirement was executed and verified. `partial` means useful authorized work was completed but one or more requirements remain. `needs_decision` means a design-changing ambiguity requires Windows-side approval. `blocked` means no further authorized progress is possible because of an external dependency or access failure.

Every report must include task ID, status, base commit, HEAD immediately before the report commit, branch, environment identity, commands and exit codes, observed results, deviations, unverified claims, and artifact paths. The final report commit hash is verified from the pushed remote branch rather than embedded recursively in its own content. Credentials, private keys, access tokens, large logs, datasets, recordings, generated assets, and checkpoints must never be committed.
