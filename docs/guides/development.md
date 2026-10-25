# 开发环境与验证

## 两类解释器

科研工作流工具只用标准库，推荐Python3.12及以上；本次Windows使用Python3.13.5，不因此更改仿真Python约束。以下验证不会导入Isaac、Torch或自动加载pytest插件，也不需要pip安装根项目。

```bash
python -m unittest discover -s tests/research_workflow -v
python scripts/research_workflow.py check
```

单测临时夹具位于当前仓库被忽略的artifacts/workflow-tests并自动清理自己的临时子目录，避免Windows受限环境默认Temp权限问题。此测试目录不用于真实训练或存储唯一证据。

## Isaac运行环境

仿真仍使用既有run_isaaclab.sh选择匹配的外部Isaac解释器。根pyproject包含大量上游工作区依赖，不是本工具安装清单，不要为运行文档检查而安装或升级Torch、CUDA和Isaac。机器人USD、磁场vendor、冻结位姿库及掩码依赖在Linux侧定位和核验，普通Windows Python不能证明它们可运行。

## 检查层级

工具单测验证记录行为，结构检查验证受管理的路径和引用。原tests/new_stomach_rl、stomach_migration、magnetic_following及其他仿真回归继续按原环境指南运行；具体历史命令见 [训练就绪报告](../../handoffs/reports/NEW-STOMACH-RL-TRAINING-READINESS-20261008.md)。报告中的314项不是本次本地执行结果。

## CI与源码

research-workflow CI只在Linux和Windows运行上述标准库测试与文档检查，不加载资产、下载模型或启动训练。源码目录和现有CLI不变，新增文件不再被tests整体忽略；缓存和artifacts继续忽略。CI通过不表示历史仿真回归或GPU训练通过。
