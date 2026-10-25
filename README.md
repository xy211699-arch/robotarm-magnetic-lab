# Robotarm Magnetic Lab

本项目基于 Isaac Lab，研究外部磁控胶囊在胃部模型中的运动、视觉覆盖和强化学习控制。仓库同时保留旧胃 TASK-010、新胃磁控发布版和新胃 RL，三者动作语义不同；当前主线为新胃八环境 RL 开发，已实现内容和验收缺口见 [当前状态](docs/STATUS.md)。

## 从这里开始

AI 先读 [协作约定](AGENTS.md)，再经 [文档导航](docs/README.md) 和 [待办](docs/TODO.md) 找到当前任务。实现定位见 [仓库结构](docs/REPOSITORY.md) 与 [系统架构](docs/architecture/system.md)，实验来源见 [研究记录](experiments/README.md)，本批变更见 [维护历史](version_history/README.md)。

## 无仿真依赖的科研工作流

以下命令只使用 Python 标准库，检查文档与任务引用、输出上下文和测试工具本身；不安装 Isaac、不启动 GPU 或训练。任务草稿与运行记录的方法见 [科研工作流指南](docs/guides/ai_research_workflow.md)。

```bash
python scripts/research_workflow.py check
python scripts/research_workflow.py context
python -m unittest discover -s tests/research_workflow -v
```

## 仿真与训练

运行仍依赖外部 Isaac Sim/Isaac Lab、CUDA、PyTorch、磁场 vendor、机器人 USD 与冻结位姿库，不是独立安装包。依赖边界见 [环境指南](docs/guides/development.md)，实际入口见 [训练与验收](docs/guides/training_validation.md)。

A/B/C/D 分别为 Blind+Single、Vision+Single、Blind+Chunk、Vision+Chunk。最新登记开发预算为 N8、200 updates、rollout8、每组累计12小时，依据 [R2 指南](docs/NEW_STOMACH_RL_R2_GUIDE.md) 最后“最终确认”一节；这不是完整 GPU 训练通过，也不是正式多种子结论。旧1000×64和500×32仅保留为历史，远程正在运行的状态需另行核验。

准备和启动仿真前核对授权、代码身份、配置和外部资产。完整日志、权重及视频保存于 Git 外，精选证据沿用 handoffs，详见 [工件存储](docs/guides/storage.md)。
