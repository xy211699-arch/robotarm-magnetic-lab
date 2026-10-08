# 新胃RL容量阶段：P0操作说明

当前仅交付P0证据和原单环境基线采集工具，**不是已支持多环境的训练入口**。P1—P3尚未完成，正式训练不得启动。

## Windows端核验小工件

获取`feature/new-stomach-rl-capacity-20261008`最新提交，在该分支的独立工作区运行普通Python，无需Isaac Lab：

```bash
python scripts/new_stomach_rl/export_review_bundle.py verify --output_directory handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/preflight_review_v1
python scripts/new_stomach_rl/export_review_bundle.py audit --output_directory handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/preflight_review_v1
```

`verify`复核29份小工件和原始清单，`audit`直接从已交付CSV重算旧D单环境吞吐/计时/投影比例。原始Linux绝对路径仅作来源记录，Windows复核不需要访问这些路径。新目录的Git属性禁用换行转换；不要手工重新格式化CSV、JSON或日志。

两个入口均不启动仿真。请将核验结果交回方案端。当前仅在Linux完成了复核，不能声称Windows已人工确认。

## Linux端人工采集固定参照

先运行Single。确认终端最终`NEW_STOMACH_PREFLIGHT`的`status`为`pass`后，将下面`MODE=single`改为`MODE=chunk`再运行一次。**进程退出码0不替代摘要status**；失败或中断时先反馈，不继续后续优化。

```bash
cd /mnt/isaac-linux/isaacsim/.worktrees/new-stomach-rl-capacity-20261008

MODE=single
POSE_MANIFEST=/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-coverage-regenerate/configs/new_stomach_v1/entry/pose_library_manifest_v1.json
TARGET_MASK=/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/new_stomach_coverage/tube_candidate_full_right_appendage_v1/candidate_face_mask.npz

ROBOTARM_MAGPYLIB_VENDOR=/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/vendor \
./run_isaaclab.sh -p scripts/new_stomach_rl/capture_capacity_reference.py \
  --mode "$MODE" --pose_manifest "$POSE_MANIFEST" --mask "$TARGET_MASK" \
  --device cuda:0 --viz none
```

每种模式使用固定`train-0003`、固定20个原始动作，独立reset重复两次，策略物理时间共40秒，不训练网络。沿用旧任务：每秒240步磁力/物理、10次覆盖、1次RGB，四段Chunk全部执行。墙钟预计每模式约3–6分钟，按旧单环境吞吐加启动成本估计，尚未实测此采集脚本。不要同时开两个GPU实例。

输出在本项目：

```text
artifacts/new_stomach_rl_capacity/reference/<UTC运行ID>/
  summary.json
  raw_actions.json
  repeat_0/
    reference_tape.npz
    policy_1hz.jsonl
    coverage_reward_10hz.jsonl
    rgb_1hz.jsonl
  repeat_1/
    同上
```

每个repeat应有4800物理步/4800原磁场调用，C10 201点、C1 21点，共222份分支覆盖集合。NPZ记录真实逐步动力学、施力**前**的磁场模型输入及先前滤波状态、施力**后**的原始/滤波wrench，不能以积分后的胶囊pose代替模型输入。覆盖集合采用little-endian packbits，解包时按`num_vertices`裁掉填充位。

基线不是性能测量：完整mask/逐步记录会增加开销。脚本未登记优化容差；P1必须先根据原模型重复性登记容差，不能看到优化误差再放宽。`Ctrl+C`会记`interrupted`，不记通过；强杀可能来不及持久化摘要，缺摘要也不能通过。

## 继续条件

P0尚待Windows端复核及上述真实参照采集。用户已批准报告列出的三个可选`env_root`接口，并已完成实施及32项CPU作用域回归（完整回归177项）。旧调用默认仍为全Stage审计，旧单环境断言未删除；这尚不代表向量任务或GPU两环境隔离已完成，无需重复审批同一接口改动。

P2两环境正确性通过后才能测P3固定容量阶梯`1,4,8,12,16,20`。预算/批量未冻结前，不进入R0—R2的GPU接入或正式训练。

2026-10-09更新：Single/Chunk人工采集均已通过Linux离线复核，原数据未修改；下一步使用`docs/NEW_STOMACH_RL_CAPACITY_P1_GUIDE.md`中的热点入口。Windows端小证据复核尚未获得明确确认，不冒称已完成。
