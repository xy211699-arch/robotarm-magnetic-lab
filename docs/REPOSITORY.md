# 仓库结构与维护规则

## 入口和事实来源

README为入口，AGENTS为协作规则，STATUS维护当前结论，TODO维护下一步，research_state.json提供任务身份。repository_layout.json列出受检查的路径和文档，不是执行配置；新文档须显式登记才进入链接检查。

## 实现层

source/robotarm_magnetic_lab/robotarm_magnetic_lab保留Isaac扩展布局。tasks/manager_based是环境与执行器接线，runtime负责观察、奖励、覆盖同步和生命周期，learning负责Actor/PPO/collector/checkpoint，coverage和geometry定义几何，baselines保留旧控制器。scripts按任务保存入口，tests保存行为回归；本次不复制或重排源码。

## 文档层

docs/architecture解释架构与决策，guides提供操作，plans保存新方案，templates提供记录格式，archive保存旧入口。已签发docs/superpowers计划保持原位并允许跟踪，不强制迁移。PROJECT_STATE转为兼容导航，旧文保留归档。

## 合同与实验

handoffs/active保存合同及标为draft的草稿，reports保存返回报告，attachments保留精选证据；experiments只提供索引与规范，不复制第二套结果。version_history记录每批变更、验证与恢复。数据、权重和完整录像进入Git外artifacts，详见 [存储](guides/storage.md)。

## 兼容性与来源

旧代码、配置、资产和结果路径保持不变。原Git整体忽略tests和docs/superpowers，已跟踪文件仍在历史，但新文件易遗漏；本批解除整体忽略并继续忽略缓存。工具不安装根pyproject的仿真依赖，不成为训练器隐式门禁。结构借鉴来源见 [设计说明](plans/2026-10-25-ai-research-workflow-design.md)，没有迁入其云盘、数据或模型约定。
