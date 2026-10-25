# 系统架构与实现定位

## 阅读范围

本页是现有main实现的图谱，不是尚未存在的新系统。包内路径相对source/robotarm_magnetic_lab/robotarm_magnetic_lab，文件存在性由 [结构登记](../repository_layout.json) 检查。外部Isaac、vendor、USD和位姿库仍需运行端配置。

## 入口与生命周期

scripts/new_stomach_rl/prepare_r2.py准备绑定代码和资产的配置，supervise_training.py管理后台、状态、停止和人工恢复，train.py直接创建NewStomachRLVectorEnv，不需要虚构已注册Gym ID。

tasks/manager_based/robotarm_magnetic_lab/new_stomach_rl_vector_env.py和robotarm_magnetic_new_stomach_rl_vector_env_cfg.py负责N8环境接线。runtime/new_stomach_rl_vector_rows.py管理逐行坐标，new_stomach_rl_vector_lifecycle.py管理reset/WARMUP，new_stomach_rl_vector_runtime.py协调覆盖与边界。局部reset不可额外推进全场景物理。

## 学习与恢复

learning/new_stomach_rl_actor.py、new_stomach_rl_distribution.py和new_stomach_rl_runner.py提供GRU、连续分布与更新。new_stomach_rl_collector.py采集真实env.step，new_stomach_rl_long_rollout.py处理GAE、有效掩码和序列回放，new_stomach_rl_checkpoint.py记录模型、优化器、RNG及身份。GRU跨rollout延续，检查点保存与恢复会开新回合，不保证PhysX轨迹连续。

## 执行与磁作用

mdp/new_stomach_rl_vector_action.py把9D Single或4×9D Chunk送入controllers/actuator_vector.py及网格保护，校核速度、加速度、停止路径和5 mm间距。实际执行历史与网络sample/latent/logp分存，不能以投影动作替换PPO采样概率。

mdp/new_stomach_rl_vector_magnetic.py及legacy_bridge.py每240 Hz物理步提供磁力矩；runtime/finite_magnet_cache.py只复用不变磁体对象和局部积分网格。getFT仍含逐行CPU计算，GPU PhysX并行不代表磁计算已GPU批处理。

## 观察与权限

runtime/new_stomach_rl_rgb.py、new_stomach_rl_vector_rgb.py、new_stomach_rl_optics.py负责1 Hz真RGB、同步与10 Hz精确覆盖；new_stomach_rl_reward.py汇总冻结奖励。Actor仅用允许的视觉和实际指令历史，Critic特权通道隔离；覆盖、pose/split和奖励不可成为Actor输入，安全执行器和记录器的信息权限不自动授予策略。

## 三种版本

旧TASK-010是旧胃六模式加力度、10 Hz直接施力，入口scripts/stomach_coverage/train_task010.py。新胃A版1 Hz的4×9D是四秒预览、只执行第一秒，见 [发布归档](../RELEASE_A_NEW_STOMACH_MAGNETIC.md)。新胃RL的Single执行一秒，Chunk四个0.25秒都执行，三者不可混用。

## 协作层

scripts/research_workflow.py只读登记并管理记录，不导入仿真或学习模块，不控制supervisor，不验证策略性能。CI只运行该层标准库回归，GPU及资产验收按独立合同执行。
