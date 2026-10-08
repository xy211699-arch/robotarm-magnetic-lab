# 新胃连续RL预验证执行报告（进行中）

当前状态：`partial`。Gate 0通过；Gate 1–6未验收。未启动PPO或正式训练。

## 版本及边界

- 发布基准：`1e89c9ccb48545c4513915be94d4c17d9cdebaaa`。
- 规划/实施起点、Gate 0日志记录HEAD：`8a1156a9cf349fdd730cc6d39132628b3f73f641`；测试包含此时的新增未提交代码，不把规划提交误称为实现提交。
- 分支：`feature/new-stomach-rl-preflight-20261008`。
- 工作树：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-preflight-20261008`。
- 新增：`scripts/new_stomach_rl/validate_preflight.py`、`tests/new_stomach_rl/test_preflight.py`、本报告；项目日志更新。未改归档控制器、旧TASK-010、资产、物理参数或位姿库。
- superpowers技能当前不可用，使用逐Gate/TDD流程，未安装额外技能。

## Gate 0观察结果

| 检查 | 结果 |
| --- | --- |
| 发布语义 | 原4秒961点预览、首秒241点提交、单环境断言及5mm余量均保留 |
| 库 | 已确认位置/50mm区域的1200条，train/validation/test=1000/100/100，最终位姿唯一，原生成门槛满足 |
| 几何 | GPU Stage哈希c4028caa5da750d769a6c5e39709bd7caac975339f35eac5d36fc0e48036c72e，与位姿库一致 |
| 掩码 | 429029面重算标签差异0；排除176109、目标252920，目标面积约0.040205301m² |
| 依赖 | 七份外部磁场/配置/XRDF/URDF及原Stage哈希匹配；USD解析19层，无未解析项 |
| GPU回载 | 固定各split20条，60条全部通过；逐条完整环境及滤波重置，240步HOLD后物理/RGB有限、光心在胃腔侧、无方向长轴倾角≥45°、无终止 |
| GPU证据 | env/simulation/RGB=cuda:0；实际/physicsScene的GPU Dynamics=True、broadphase=GPU |
| 测试 | 75 passed、50 warnings，退出码0；git diff --check通过 |

HOLD允许真实磁力自然响应，不声称胶囊完全静止。源锚点本身未确认落稳，但1200个库样本各自通过原生成稳定性条件。test数据仅做资产有效性核查，未用于挑模型或调参。

版本：RTX5090、驱动595.91.07、Python3.12.13、PyTorch2.11.0+cu128；本机Isaac Sim VERSION为6.0.1-rc.7+release.42383.32955d8d.gl，IsaacLab仓库VERSION为3.0.0；运行包isaaclab.__version__实际返回12.0.0，两者分开记录，不擅自解释或重装。

## 实际命令

在上述工作树运行：

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/new_stomach_rl/test_preflight.py tests/stomach_migration tests/magnetic_following \
  -q --disable-warnings

ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/new_stomach_rl/validate_preflight.py \
  --gate assets --runtime --device cuda:0 --viz none

git diff --check
```

## 工件和工具修复

以下相对路径均以工作树绝对路径为前缀；大工件不进入Git。

- 正式摘要：`artifacts/new_stomach_rl/preflight/20261008T054338.801049Z/summary.json`；60条记录为同目录`runtime_reload.jsonl`。摘要assets字段含库/依赖的绝对路径、字节数和SHA。
- 完整工件清单：同目录`artifact_inventory.json`（交付前生成并核查），含全部绝对路径、字节数及SHA。
- 最终日志：`artifacts/new_stomach_rl/evidence/gate0_runtime_final.log`、`gate0_tests_final.log`。
- 初始诊断`gate0_runtime.log`未逐样本清空磁力滤波，且Kit关闭前没有持久化总摘要；不作为Gate通过证据。
- 第二次`gate0_runtime_reset_isolated.log`因新审计工具错误解包最近点函数返回值失败；失败摘要位于`artifacts/new_stomach_rl/preflight/20261008T054059.845404Z/summary.json`。原库和物理门槛未改，修复工具后重跑。
- TDD红灯：上述evidence目录`gate0_shutdown_red.log`、`gate0_closest_red.log`各2 failed，最终修复后通过；初始未实现检查时终端观察5 failed/1 passed。
- 固定日志地址`/mnt/isaac-linux/robotarm_magnetic_lab/logs/runtime.txt`由既有桥追加本轮记录，未改加载/日志路径。

未验证：Single/Chunk安全执行、C10/C1、奖励、Actor隔离、PPO短训、并行容量及正式训练预算；不把Gate 0结果当作全部任务完成。
