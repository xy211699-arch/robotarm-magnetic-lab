# 研究记录入口

## 单一证据位置

本仓库继续用 [活动合同](../handoffs/active/README.md)、[返回报告](../handoffs/reports/README.md) 和handoffs/attachments保存精选证据，experiments不复制第二套报告。当前研究身份见 [登记](../docs/research_state.json)，最新结论见 [状态](../docs/STATUS.md)。

## 新记录

[实验模板](../docs/templates/experiment.md) 规定假设、配置、划分、种子、命令、指标与限制。可用research_workflow new-run初始化Git外的唯一记录，再登记工件大小与哈希；它不启动原监督器，也不证明实验成功。原Isaac作业目录及证据保持原样，详见 [工作流指南](../docs/guides/ai_research_workflow.md)。

## 证据等级

本次执行结果与历史报告分开；CPU回归、GPU工程验收、训练完成和研究效果依次需要各自证据。正式种子与test协议另行冻结，旧TASK-010、新胃A版和新胃RL不能混为同一实验。
