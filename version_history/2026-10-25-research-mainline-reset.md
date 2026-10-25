# 2026-10-25研究主线重设

## 依据和计划

用户明确旧四组只是可行性验证，要求主线收敛为项目介绍、已有实现、下一阶段任务与基础入口。基准46d01fbb9bedd4b1fe8babed089a2919e39e407b，沿用本会话直接main授权。用documents技能核对261017任务.docx正文、表格和四张图，依brainstorming及writing-plans梳理范围，将实施计划合并本记录以免再增加多层流程文档。

顺序为核对材料和代码、精简当前文档、建立四类入口、修正AI与机器上下文、检查审阅发布。材料是暂定方向，不是执行指令；正文GRU与图中Transformer冲突保留待决，不复制DOCX、私有路径或人员分工到公开仓库。

## 最终行为

README和docs收敛为PROJECT、IMPLEMENTED、NEXT、ENTRYPOINTS。旧状态、训练待办及重复流程说明移出当前路径，固定Git历史可回查；source、configs、assets、既有回归及工件不变。handoffs索引明确旧任务不活动，当前登记去掉旧预算与四组。

context不再硬编码STATUS/TODO而使用reading_order，check核验声明阅读路径。草稿模板引用本地设计、映射时序、实际动作与取消行为，不构建虚假新模型。

## 验证

新增两项阅读顺序测试先FAIL，修复后python -m unittest discover -s tests/research_workflow -v共23项，22通过、1项符号链接用例因本机权限跳过，退出0。python scripts/research_workflow.py check与context通过，后者只显示本地设计接收draft及新的阅读顺序；git diff --cached --check通过。72个旧说明文件退出当前路径，docs保留9个文件，其中4份为主文档。

使用暂存树与基准逐目录比较，source、assets、configs、dependencies、handoffs/artifacts和handoffs/attachments的Git树哈希完全相同；源DOCX哈希与登记一致。没有启动Isaac、训练或硬件，也未独立重跑历史GPU结果。远端CI在本记录编写时尚未运行，推送后再核对，不能冒称本地通过等于CI通过。

## 恢复

旧说明完整保留在46d01fb，按本批范围创建修正或revert，不强推覆盖团队工作。实验数据和本地源DOCX未移动改写，研究架构仍等后续本地方案明确。
