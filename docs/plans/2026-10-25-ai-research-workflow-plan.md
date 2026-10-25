# AI 科研工作流实施计划

**目标：**在保留现有运行接口的前提下，建立可交接、可查证的科研协作入口。

**设计依据：**[改造设计](2026-10-25-ai-research-workflow-design.md)。技术栈为 Markdown、JSON 和 Python 标准库；用户已要求直接实施，不额外设置设计审批。

## 任务一：工作流工具与回归

创建 scripts/research_workflow.py 和 tests/research_workflow/test_workflow.py。先在临时目录测试检查器能拒绝断链和损坏登记，生成器能拒绝覆盖和路径穿越，证据核验能拒绝缺失与篡改；记录首次失败，再实现 CLI。check 返回非零表示结构错误，context 不做写入，new-task 始终生成 draft，new-run 始终 initialized。测试采用 unittest，不依赖 torch、pytest 插件或 Isaac。

## 任务二：研究入口与文档分层

重写 README、AGENTS.md、PROJECT_STATE.md，新增 docs/README.md、STATUS.md、TODO.md、REPOSITORY.md、research_state.json、repository_layout.json。新增 architecture、guides、templates 和 version_history 的内容，补充 handoffs 两级导航；旧状态页归档，历史实验原文不改。新增文档使用中文段落、标题及必要命令，避免嵌套清单。

## 任务三：交付检查

解除 .gitignore 对测试和计划源码的整体忽略，仅保留缓存/工件忽略；为标准库检查增加独立 CI，不自动安装根 pyproject 的大型仿真依赖。运行工具单测、真实仓库检查和临时 Git 仓库端到端；审阅范围确保 source、configs、assets、历史结果和其他副本无改动。维护记录区分本地检查与历史报告，给出限制及恢复方式。

## 执行记录

2026-10-25：已完成参考和本地状态检查，建立 feature/ai-research-workflow-20261025。当前目录为独立克隆，采用当前副本的独立分支交付可审阅改动，不新建不可见工作副本；既有两个 .local-preflight 目录未改。工具与文档的共享接口为 repository_layout.json 与 research_state.json，先固定结构再做最终自检。实施和验证结果将在本文件末尾追加。

## 范围修订和进度

用户后续明确要求直接修改公开远端main。已fetch并核对0552cc5，切换main，读取更新后的容量报告、训练入口与200×8/N8/12小时预算；不沿用旧本地一环境判断，不混入远端已有算法变更。当前副本原为稀疏检出，现已补齐登记需要的目录；Git LFS shell在沙箱不可用，单次Git检查禁用过滤器，只检出既有指针，不下载或改变资产。

任务一已完成：先观察导入缺失失败，再实现工具；Windows默认临时盘权限导致夹具失败后，改为仓库内artifacts/workflow-tests。最终21项中20通过、1符号链接用例因权限跳过，退出0；包含CLI端到端和篡改拒绝。任务二已完成文档与JSON登记，真实check通过。任务三完成独立审阅及修复复核、CI结构解析与diff检查，发布前核对仅修改文档和工作流；远端发布身份在推送后单独核验。详见本批version_history记录，未运行仿真或训练。
