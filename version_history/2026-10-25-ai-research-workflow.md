# 2026-10-25 AI科研工作流改造

## 范围和来源

基准为公开目标仓库main的0552cc5ca7bc98b97ac0a414b3e4945b34b2f7ff。用户明确要求直接修改main；现有两份.local-preflight目录及旧planning副本保留。参考项目GitHub读取返回404，实际参考本机Hierarchy_Tactile_WAM副本498c2bb078cd3cab1f2026eff9e70789611cc926，没有核实与其GitHub main同步。

## 最终行为

补齐README、AGENTS、文档导航、STATUS/TODO、机器可读研究登记、系统图谱、决策、操作指南、模板和本维护记录。复用handoffs单一证据位置，旧PROJECT_STATE归档并保留兼容入口，历史实验和实现不改。新增research_workflow提供check/context/new-task/new-run/verify-run，不启动实验；解除tests和docs/superpowers的整体忽略，新增独立标准库CI。

## 验证

Windows Python3.13.5实际执行python -m unittest discover -s tests/research_workflow -v，共21项，20通过、1项符号链接测试因本机权限跳过，退出0；包含CLI草稿、运行初始化、空证据拒绝、真实文件哈希通过及篡改拒绝的端到端。python scripts/research_workflow.py check通过；CI配置的双平台矩阵与只读权限本地解析通过；git diff --check通过。远端Actions本记录编写时尚未运行，不把本地结果冒称CI通过。

首次测试在工具不存在时因导入缺失失败；默认Windows临时盘被沙箱拒绝后，改用仓库内artifacts/workflow-tests夹具。独立审阅发现已登记ID但合同文件名不同会重复建草稿，补回归复现FAIL后修复并复核通过；另验证子目录不会冒用父Git仓库身份。符号链接越界用例保留供Linux CI实际执行，本机跳过如实记录。

## 研究状态与限制

main已包含N8和200×8/12小时修订，不能沿用本地旧副本“最大一环境”结论。314项历史CPU回归及GPU短测均来自Linux旧记录，本次未重跑。没有修改source/configs/assets，没有启动仿真或训练，也未复核全部外部数据。check只覆盖登记文档链接和路径，不检查锚点、网络及研究语义；verify-run只核验列出文件，不证明结果完整或科学有效。

## 恢复

本批按用户要求在main准备正常快进提交，实际发布身份以远端核验为准。恢复时以本批提交为边界创建revert或修正提交，不硬重置或强推。原PROJECT_STATE保留归档，原运行接口、合同、报告和数据位置保持原样。Git外初始化记录不得与真实作业混淆，不删除用户工件。
