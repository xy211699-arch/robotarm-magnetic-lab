# 新胃8环境：R2人工启动有限验收

## 当前用途与边界

这是训练链路验收，不是正式开发种子训练或覆盖率性能比较。工程及277项CPU回归已通过；真实GPU训练、TIMEOUT/回载和固定回放结果尚未运行。旧任务及物理、磁力、5mm安全距离、10Hz面积覆盖、1HzRGB和模型定义不变。

每次只启动一组：A=Single/Blind，B=Single/Visual，C=Chunk/Blind，D=Chunk/Visual。每组8环境、3次更新，每次64个全局一秒策略边界；原烟雾候选超参、完整64步序列、单minibatch及2 epochs仅用于有限验收，不是正式冻结超参。

第5个训练边界后重置行0，不推进额外物理；其WARMUP不计样本。前两次更新跨120秒TIMEOUT，各行均须实际出现TIMEOUT。第二次更新后显式结束剩余回合、全行新reset到零秒边界，保存training-boundary checkpoint；新建runner回载模型/Adam/RNG，再新reset并清GRU，执行第三次更新。此reset截断余下轨迹会记录，不声称PhysX连续恢复。另执行一个冻结train位姿及一个冻结validation位姿各120秒、八行确定性回放，不更新网络。回放不计训练有效样本。

根据N=8短容量测量：192个训练边界纯步进约57分钟，另外240个固定回放约72分钟及初始化约11分钟。完整验收粗估可能超过2小时；这不是实际新训练耗时承诺。**保留7200秒硬上限，含初始化与退出宽限**，不能保证两小时内完成全部内容。超预算暂停并保留数据，不删除/放宽验收，不自动重启或接着运行下一组。完整四组结果目前全部not_run，不能称R2通过。

## 一次性准备并启动D组

先确认没有其他GPU仿真/训练。准备脚本只做文件/哈希检查、不创建Kit；会生成四组有限配置，绑定当前Git HEAD及实际源码/配置/资产/原权重字节。代码提交或文件改变后，旧配置不能再用于启动，须重新准备。不要直接使用`r2/preparation_audit`，那是工程提交时的历史CPU配置审计，不是当前HEAD的启动配置。

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008
export ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor
CONFIG_DIR="$PWD/artifacts/new_stomach_rl_capacity/training_readiness/configs/inputs_$(date +%Y%m%d_%H%M%S)"

./run_isaaclab.sh -p scripts/new_stomach_rl/prepare_r2.py \
  --pose_manifest /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-coverage-regenerate/configs/new_stomach_v1/entry/pose_library_manifest_v1.json \
  --mask /mnt/isaac-linux/robotarm_magnetic_lab/artifacts/new_stomach_coverage/tube_candidate_full_right_appendage_v1/candidate_face_mask.npz \
  --weights /home/multirobo/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  --capacity_summary artifacts/new_stomach_rl_capacity/capacity/20261009T125733.819352Z/summary.json \
  --capacity_boundaries artifacts/new_stomach_rl_capacity/capacity/20261009T125733.819352Z/boundaries.jsonl \
  --output_dir "$CONFIG_DIR" && \
./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_training.py start \
  --config "$CONFIG_DIR/D.json"
```

准备失败时不会启动后面的进程。成功启动立即打印`run_dir`并返回，后台独立运行，不依赖当前终端或Codex持续在线。不要重复start，也不要直接运行内部worker。单任务锁禁止同项目并发；其他项目/旧训练须由使用者先确认未运行。D组结果审阅后，其他组仍由用户分别启动，不提供自动四组串行训练。

## 查询与停止

可另开终端，查询不会修改运行状态：

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008
./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_training.py status
```

只显示最近一次R2任务。查看指定历史任务可加`--run_dir /绝对任务目录`。尚未启动时没有latest，不要把文件不存在当GPU故障。

主动停止：

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008
./run_isaaclab.sh -p scripts/new_stomach_rl/supervise_training.py stop
```

监督器只停止自己启动的进程组，SIGINT宽限20秒，超时KILL兜底。若物理步未及时返回，不能保证写出中途参数快照；已写出的边界日志保留。正常响应时尽量写`interrupted_weights_only.pt`，它不是严格同轨迹续训checkpoint。停止显示interrupted，不当作pass。异常/预算/陈旧心跳为paused_on_error；查看worker.log及summary，不自行跳过故障、放宽指标或自动恢复。

## 如何判断结果

`queued/running`只是运行状态。`stage=sampling`时update长期不变并不代表卡死，应看有效样本、worker心跳、边界日志是否推进；监督器自己的心跳与worker心跳分别记录。启动阈值1909秒、进度阈值181秒来自当前N=8容量证据。

仅status=completed且summary全部通过，才说明该组有限链路完成；不是R2四组完成或策略优于随机/Blind。真实GPU PhysX、相机、覆盖设备必须cuda；每行每边界240次物理/磁力、十条10Hz记录（WARMUP仅C0）、1Hz RGB哈希和TIMEOUT旧终点值均在链路内检查。

## 数据位置

```text
artifacts/new_stomach_rl_capacity/training_readiness/jobs/<UTC运行ID>/
  launch.json / frozen_config.json
  status.json / heartbeat.json
  worker.log / supervisor.log
  boundaries.jsonl             # 原始10Hz/C1、位姿ID、执行历史、旧终点Critic、阶段
  rollout_update_0001.pt       # CPU张量字典：原始采样/latent/logp、有效掩码、GAE取值
  rollout_update_0002.pt
  rollout_update_0003.pt
  boundary_update_0002.pt      # 回合边界模型/优化器/RNG
  interrupted_weights_only.pt # 仅异常时且成功保存才存在
  summary.json                # 设备、更新、样本数、TIMEOUT、回载、固定回放及工件哈希
```

`jobs/latest`仅为最近任务指针。原始日志/张量/checkpoint都留在当前项目artifacts，不入Git。请把实际run_dir及最终summary提供给执行端，随后才能判断R2是否通过；这一步没有训练VLM或正式研究种子。
