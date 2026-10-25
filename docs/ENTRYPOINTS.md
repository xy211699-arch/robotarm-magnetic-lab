# 基础入口

## 代码与文档

README连接四类主文档，AGENTS说明协作边界。实现放source/robotarm_magnetic_lab，脚本放scripts，配置放configs，回归放tests；保持路径和运行接线，不为整理文档改变仿真导入。

## 不启动仿真的检查

下列命令仅需Python标准库与Git，不加载Isaac、模型或硬件。check检查登记路径和相对文件链接，context显示阅读顺序及当前任务；这些不核验研究效果、网络链接、Markdown锚点或远程实时状态。

```bash
python scripts/research_workflow.py check
python scripts/research_workflow.py context
python -m unittest discover -s tests/research_workflow -v
```

## 现有仿真入口

仿真依赖匹配的Isaac Sim/Isaac Lab、CUDA、磁场vendor和外部USD等资产。run_isaaclab.sh仍固定调用/mnt/isaac-linux/IsaacLab/isaaclab.sh，新成员先核对本机路径；仓库不是独立安装包，文档检查不需要安装根pyproject的大型仿真依赖。

下面两条会启动Isaac运行时，不是静态目录扫描；第一条列出注册任务，第二条是已有新胃基础场景100步零动作入口。它们不代表新多模态系统或触觉标定完成，本次未执行。

```bash
./run_isaaclab.sh -p scripts/list_envs.py --keyword Robotarm-Magnetic
./run_isaaclab.sh -p scripts/zero_agent.py --task Template-Robotarm-Magnetic-New-Stomach-Lab-v0 --num_envs 1 --max_steps 100 --headless
```

## 设计和记录

本地方案与任务拆分见 [NEXT](NEXT.md)。任务草稿可用python scripts/research_workflow.py new-task --help了解参数，生成draft不改变活动登记；new-run只初始化独立记录，verify-run只检查列出文件的大小和SHA-256，不接管训练监督器。

完整日志、数据、checkpoint和视频留Git外artifacts，先核对持久化路径。旧脚本沿用scripts/_artifact_paths.py，新增记录默认当前副本artifacts/research，可显式指定已确认位置；不覆盖旧结果。精选证据保留handoffs，旧操作说明见 [历史入口](archive/README.md)。
