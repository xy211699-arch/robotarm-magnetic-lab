# 当前状态

## 快照身份

本页于2026-10-25依据远端 main 的0552cc5ca7bc98b97ac0a414b3e4945b34b2f7ff及其中代码、配置和报告整理。实验记录主要截至2026-10-10，本次未连接Linux重跑、核验实时作业或全部外部工件。机器入口为 [研究登记](research_state.json)，登记日期不是实验测量日期。

## 容量与训练

最新记录已包含多环境适配，不能再用旧预验证“最大一环境”描述当前主线。[容量报告](../handoffs/reports/NEW-STOMACH-RL-CAPACITY-20261008.md) 与 [运行日志](PROJECT_RUN_LOG.md) 记录D组短容量测量至N20，用户冻结N8用于开发；这不证明所有组长期稳定或扩展效率充分，Chunk跨P0参照的姿态等价仍有暂缓项。

长程collector、TIMEOUT GAE、GRU掩码、边界checkpoint、监督器及人工恢复已实现。日志记录最新314项CPU回归通过，这是Linux历史报告，本次没有独立重跑这314项。最新开发预算为A/B/C/D各200 updates、rollout8、N8、保存50、累计总墙钟43200秒，含初始化、回放与退出，依据 [R2指南最终确认](NEW_STOMACH_RL_R2_GUIDE.md) 和日志末段。恢复不重新领取12小时，预算中止不算完成。

真实完整GPU训练仍无通过证据。旧A运行停止于0更新；优化后900秒短测为12个有效边界、96样本、0更新，详见 [性能报告](../handoffs/reports/NEW-STOMACH-RL-PERFORMANCE-500X32-20261010.md)。8步rollout真实PPO、长期学习稳定性和正式多种子性能仍未验收。已有人工开发启动授权保留，本次不发起训练；不能据旧停止报告推断服务器此刻空闲。

## 历史版本

[A版发布](RELEASE_A_NEW_STOMACH_MAGNETIC.md) 是首秒执行、后三秒预览的归档身份，[新胃预验证](../handoffs/reports/NEW-STOMACH-RL-PREFLIGHT-20261008.md) 是早期单环境证据。两者不替代后续RL四段实际执行的语义。旧TASK-010直接施力、十二环境和正式种子结果属于旧胃，不能外推到新胃磁控。

## 科研协作设施

本批增加导航、AI约定、架构图谱、登记、模板和标准库检查，实际验证见 [维护记录](../version_history/2026-10-25-ai-research-workflow.md)。结构检查只覆盖显式管理范围，不意味着历史自由文本已全面审计；文件完整性通过不验证科学结论。
