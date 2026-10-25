# AI 科研工作流

## 接手与范围

先阅读AGENTS、STATUS、TODO和活动合同，核对工作树及代码身份。以下命令可从仓库根目录执行；脚本默认根目录由自身位置确定，也支持在子命令前指定--root。context输出任务来源，不读取凭据、不联网，也不把历史报告变成当前远端状态。

```bash
python scripts/research_workflow.py check
python scripts/research_workflow.py context
```

## 任务草稿

生成器使用 [任务模板](../templates/task.md)，拒绝已有ID、非法路径和空标题。产物始终draft，不更新研究登记或自动授权；先补齐需求、允许修改范围和验收，再依据用户明确指示登记为authorized。编号使用大写字母、数字与连字符。

```bash
python scripts/research_workflow.py new-task RESEARCH-001 --title "明确要验证的研究问题"
```

## 独立运行登记

new-run只创建唯一记录目录和manifest.json，不启动训练。默认在本工作副本artifacts/research/，--output-root可显式选择已确认的持久记录盘。时间使用UTC，元数据记录Git HEAD、分支和dirty；非Git副本记录available=false。dirty或缺失身份须在报告中说明，不能当作已冻结可复现版本。

```bash
python scripts/research_workflow.py new-run EXP-001
```

初始化status为initialized，hypothesis、command和config为空，seeds与artifacts为空。实际实验另行执行并按 [实验模板](../templates/experiment.md) 记录，不把目录存在当成运行完成。现有Isaac监督器继续使用自身目录，不将其迁入此登记器，也不复制大文件以凑齐清单。

## 工件核验

在manifest的artifacts中登记每个运行目录内文件的path、bytes和小写64位sha256，再运行下列命令。path必须为相对路径，不得越界或通过符号链接指向目录外；空清单、缺失文件、重复项、错误大小或哈希均返回非零。输出只代表字节完整性，不证明模型效果、统计可靠或GPU门禁。

```bash
python scripts/research_workflow.py verify-run artifacts/research/EXP-001/实际运行ID
```

## 结果与交接

使用 [报告模板](../templates/report.md) 和 [交接模板](../templates/handoff.md)，把假设、实现、实际验证和未验证项分开。报告仍写handoffs/reports，精选证据留attachments，完整工件留Git外。同步STATUS、TODO和research_state.json，追加维护记录后再次check。

check只核验repository_layout声明的文档、相对文件链接、架构路径和任务引用，不检查网络链接、Markdown锚点、代码块示例、研究数值或全部历史文档。新增规范文档应主动加入managed_documents；这项检查不是合并、训练或论文验收的替代。
