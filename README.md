# Robotarm Magnetic Lab

Isaac Lab 仿真项目，用于研究外部磁控胶囊在胃部模型中的运动、覆盖率和强化学习控制。

## 当前主线

`main` 当前对应新胃 RL 容量训练版本，包含四组实验：

| 组 | 视觉 | 动作 |
|---|---|---|
| A | Blind | Single，9 维 |
| B | ResNet18 | Single，9 维 |
| C | Blind | Chunk，36 维 |
| D | ResNet18 | Chunk，36 维 |

训练框架使用 8 个环境、240 Hz 物理更新和 1 Hz 动作边界。训练、检查点和评测结果保存在 Git 仓库外的 `artifacts/` 目录。

## 环境要求

- Isaac Sim / Isaac Lab
- CUDA、PyTorch 和项目磁场 vendor
- 项目依赖的胃部 USD、机器人资产和冻结位姿库

项目不是独立安装包。请先确认外部依赖路径，再使用项目启动器运行命令。

## 安装与基础检查

```bash
./run_isaaclab.sh -p -m pip install -e source/robotarm_magnetic_lab
./run_isaaclab.sh -p scripts/list_envs.py --keyword Robotarm-Magnetic
```

运行基础任务冒烟测试：

```bash
./run_isaaclab.sh -p scripts/zero_agent.py \
  --task Template-Robotarm-Magnetic-New-Stomach-Lab-v0 \
  --num_envs 1 --max_steps 100
```

## 训练入口

训练监督器支持启动、状态查询、停止和从边界检查点恢复：

```bash
./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_training.py status
./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_training.py start \
  --config artifacts/new_stomach_rl_capacity/training_readiness/configs/<CONFIG>.json
```

训练配置、检查点和实验约束见 [`docs/NEW_STOMACH_RL_R2_GUIDE.md`](docs/NEW_STOMACH_RL_R2_GUIDE.md)。

## 相关文档

- [容量与训练指南](docs/NEW_STOMACH_RL_CAPACITY_P1_GUIDE.md)
- [项目运行日志](docs/PROJECT_RUN_LOG.md)
- [历史交接记录](handoffs/reports/NEW-STOMACH-RL-CAPACITY-20261008.md)

大尺寸检查点、视频和仿真日志不提交到 Git；请根据文档中的路径和哈希进行核对。
