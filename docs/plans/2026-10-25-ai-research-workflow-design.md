# AI 科研仓库改造设计

## 目标和授权

本次按用户“开始实施”的要求改造科研协作设施，让新 AI 能从有限上下文找到当前任务、实现入口、冻结约束与证据。改造不授予新的训练、GPU实验、硬件操作、数据重建或算法变化权限，已有明确开发授权保留。当前副本为robotarm-magnetic-lab-handoff-20261008，用户随后指定公开远端main，最终基准为0552cc5ca7bc98b97ac0a414b3e4945b34b2f7ff；旧planning副本和目录外材料保持原样。

## 依据与取舍

参考来源为用户指定的 Hierarchy_Tactile_WAM，本次 GitHub 连接器返回 404，公开网页亦未能读取。实际阅读的是本机 C:/Users/35542/Desktop/Tactile_WAM/Hierarchy_Tactile_WAM，HEAD 为 498c2bb078cd3cab1f2026eff9e70789611cc926；未核实它与 GitHub main 同步。借鉴其 AGENTS.md、docs/README.md、STATUS.md、TODO.md、REPOSITORY.md、repository_layout.json、architecture/decision_log.md 和 version_history/ 的职责分离方式，不复制其模型、路径、实验结论或操作授权。

整体移动现有代码会破坏 Isaac 扩展导入、启动脚本及历史报告的路径合同；仅添加说明又不能检测文档漂移。因此采用增量改造：源码保留原位，增加规范入口、研究状态登记和轻量命令行工具。新增架构图谱说明现有模块依赖，不把文档分层冒称为运行时解耦已经完成。

## 文档和事实来源

README 是简短入口，docs/README.md 分类导航；docs/STATUS.md 给出带证据边界的本地快照，TODO.md 记录有前置条件的下一步。docs/research_state.json 只登记研究任务及来源，不承担运行作业监控或执行授权。docs/architecture/ 解释现有系统及决策，docs/guides/ 说明操作流程，docs/templates/ 保存模板，version_history/ 保存每批维护记录。历史报告、合同、运行日志与已跟踪计划保持原位，PROJECT_STATE.md 改为兼容入口并保留旧文本为归档。

远端main已包含容量与训练就绪报告，D组短容量至N20，开发冻结N8，最新预算为200更新、rollout8、每组累计12小时；真实完整GPU更新与学习稳定性仍缺证据。这些是主要截至2026-10-10的仓库记录，不是2026-10-25的Linux实时状态，初始旧副本的单环境判断已被替代。旧 TASK-010 的十二环境及正式训练结果不外推到新胃。状态只在核实新报告后更新，不能因待办文字或代码存在而宣布验收。

## 工具接口与数据边界

scripts/research_workflow.py 仅用 Python 标准库，提供 check、context、new-task、new-run、verify-run。check 检查登记文档、相对链接、架构路径及状态引用；context 输出最小阅读上下文。new-task 只生成 draft 合同；new-run 只初始化独立记录目录，记录 Git 身份与 UTC 时间。verify-run 验证显式列出的相对工件路径、字节数及 SHA-256，不将哈希通过解释为科学结论或独立复现。

输出禁止覆盖现有文件；任务 ID 拒绝路径穿越，工件真实路径不得逃出运行目录。生成操作不执行 shell 命令、不安装依赖、不探测远端、不载入模型。凭据、原始数据和 checkpoint 不进入普通 Git 历史。现有工件存储策略保留，新记录默认写入被忽略的 artifacts/research/。

## 验收及恢复

先写工具行为测试并观察失败，再实现工具与文档。验收覆盖缺失路径、坏链接、损坏登记、草稿语义、重复 ID、越界路径、空证据和哈希不匹配；再执行真实仓库检查、临时 Git 仓库端到端及 diff 检查。只声明本次实际通过的检查，不运行全仓仿真回归，也不声称重现历史 GPU 结果。用户明确要求直接main，本批验证和评审后提交并正常快进推送，不强制PR、不强推；恢复时只撤回本批新增及修改文件，不删除用户已有文件。
