# A组磁控性能优化与500×32短测

状态：优化与预算修改已实施，短测按时结束；长训练与PPO更新性能仍未验证，不能标记完整训练通过。

基线HEAD：093dd55b081b88b6662f0e34887dde3a62440d11；短测实施HEAD：645de715efc15cbdc1d5fe6b02f5eb46ad5fbe68。分支：feature/new-stomach-rl-capacity-20261008。报告提交后的HEAD与上述测量身份区分。

## 实施与边界

用户批准完整训练由1000×64调整为500×32，N8/保存50不变。collector与恢复计数取配置值，序列长度32，采集隐状态跨rollout延续；不改变模型、PPO两轮更新、奖励、磁铁材料/尺寸、有限磁体积分精度、双向反作用、滤波、240Hz磁力/物理、1秒动作边界、10Hz覆盖/1HzRGB或碰撞规则。

新增finite_magnet_cache.py，复用磁体对象及局部积分网格，不缓存时间变化的磁力；向量桥将每tick全行姿态打包一次CUDA→CPU；同tick全相机姿态/Fabric同步复用，reset失效。磁力仍逐行CPU getFT，不宣称已实现GPU磁力批处理。原legacy_bridge默认路径保持原样，只有显式向量host快照路径使用打包输入。

变更文件：scripts/new_stomach_rl/{prepare_r2.py,supervise_training.py,train.py}；runtime/{finite_magnet_cache.py,new_stomach_rl_vector_runtime.py}；mdp/{legacy_bridge.py,new_stomach_rl_vector_magnetic.py}；tests/new_stomach_rl/{test_magnetic_performance.py,test_vector_runtime.py}；操作指南与运行日志。

## 验证事实

CPU回归命令：

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor WARP_CACHE_PATH=/tmp/new-stomach-scope-warp-cache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest tests/new_stomach_rl tests/stomach_migration tests/magnetic_following tests/stomach_coverage/test_task010_recovery.py -q --disable-warnings
```

301 passed，65 warnings，13.95秒，exit0，diff检查通过。12个随机磁体姿态的双向力/矩缓存与原模型逐元素一致。独立CPU双向getFT基准100次：原0.334163秒，缓存0.240134秒，1.392倍；不等于仿真整体速度。

短测使用新准备的A配置，max_updates=500、rollout32、显式wall900/time_limited_probe（仅该短测限时）：

```bash
ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor ./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_training.py start --config artifacts/new_stomach_rl_capacity/training_readiness/configs/optimized_500x32_probe_20261010/A.json
```

run_id=20261010T041557.357525Z；最终interrupted，原因requested short-test duration reached，exit0；878.784秒（14.646分钟）包含启动/退出，监督PID1775454/worker1775455均已退出。GPU仅Awesun575MiB，无遗留训练。12个有效批次/96样本、update0；加1个初始化HOLD共13条边界。原生Kit在限时SIGINT后退出，无summary.json和检查点；已保存边界日志而非严格续训快照。没有自动重跑或进入长训练。

主动边界均值17.823秒、范围15.056–20.638秒、最近6个均值19.458秒。嵌套主机计时均值：规划1.804秒、磁力桥8.133秒（其中getFT6.073秒）；这些不是独立GPU核计时，磁力桥与getFT不能相加。剩余耗时包括PhysX/servo/覆盖/渲染同步等，本次未进一步拆分。

旧A原运行20261010T031537.437429Z的前12个主动边界均值19.402秒，与本次12个原始动作输入完全相同；当前降低8.13%（吞吐约1.089倍）。并非严谨全物理状态配对的长期基准。旧全部57个主动边界均值31.037秒，受轨迹阶段影响，不可与新前12步直接比较为约两倍优化。

## 时间估计与未验证项

500×32=16000批次，最多128000行样本（WARMUP另扣）。按当前平均/最近6步，纯采样约79.21–86.48小时，即3.30–3.60天；加初始化、每50更新reset和固定train/validation各120秒回放，初步按约4天安排。尚未完成首轮32步PPO，优化器开销、后期轨迹/碰撞规划、长回合稳定性均未验证；不能承诺4天内结束，复杂阶段可能更久。减少总样本不是算法加速或等预算比较，四组应统一新预算。

## 不可变外部工件

目录：/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008/artifacts/new_stomach_rl_capacity/training_readiness/jobs/20261010T041557.357525Z/

| 完整文件名（拼接上述绝对目录） | 字节 | SHA-256 |
|---|---:|---|
| frozen_config.json | 69013 | 9c775a521996e624ef9bb12fac841cc64dc65bc6f13341389a22f44fd3652a9b |
| status.json | 1093 | 7c98578406cb47faed812165f9e31eee29fb65d9433cf5a8c6bd138ee4d42fa7 |
| heartbeat.json | 257 | 6052de156dfef0dd956c9f51d95725c82fc65a9a356b3460a342a6d2a040c853 |
| boundaries.jsonl | 1138619 | c2e1c3a877eb75ab3c4af9e4d5ac497088d099f0ff095f44d37b56e7f8f347b5 |
| worker.log | 20475 | 19cfebbd8c98b11801566f81ea511eedf90d4420a2516a3bf3f4eafebd6704c7 |

日志含原始10Hz覆盖与1HzRGB哈希、9维动作、物理计数及分项时间；工件不入Git。正式完整配置另行在最终HEAD准备，不设置900秒限制；本文不授权自动启动完整训练。
