# 训练与验收入口

## 最新规则

新胃主线为N8，A=Blind+Single、B=Vision+Single、C=Blind+Chunk、D=Vision+Chunk。最新用户预算为200更新、rollout8、序列8、保存50、每组累计总43200秒，依据 [原指南](../NEW_STOMACH_RL_R2_GUIDE.md) 最后“最终确认”和 [项目日志](../PROJECT_RUN_LOG.md) 末段。原指南前面1000×64、500×32及旧两小时描述已被后续修订取代，不能择取旧规则混用。

## 只读状态

在已配置Linux运行环境，使用已有监督器status查看实际作业。该命令是只读操作；当前Windows文档检查不连接服务器，不能从仓库记录推断实时状态。

```bash
./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_training.py status
```

## 准备和启动

prepare_r2.py --help可查看参数；真正prepare需要明确的位姿库、mask、冻结权重与容量证据，--full_training生成当前开发预算，绑定干净HEAD和输入文件哈希。不能复用旧HEAD配置，也不在文档中猜测新机器路径。具体历史环境路径见原指南，只作为定位线索，执行端应重新核对。

start会创建GPU作业，resume会创建人工续训，stop会中断运行，均按当前用户授权执行。本次仓库改造不执行这些操作，不新增自动四组串行、重试或默认预算。已有人工开发授权保留，正式新胃多种子研究另行定义；--full_training标志本身不等于论文正式实验。

## 完成判读

completed需要实际摘要与合同门禁通过，interrupted、budget_exhausted和0更新不得记成成功。N8、200×8最多12800行样本，WARMUP另扣；总12小时包含启动、采样、PPO、保存、固定回放和退出，resume不能重置已用时限。边界保存会reset并清GRU，不声称物理轨迹连续恢复。

Actor/奖励/覆盖和5 mm安全保持冻结，不为了补足进度改语义。自然TIMEOUT、检查点截断和预算中止分别记录，不能伪造C120。详细尚未验证项见 [当前状态](../STATUS.md)，具体实现见 [架构](../architecture/system.md)。
