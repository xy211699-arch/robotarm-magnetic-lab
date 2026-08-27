# TASK-009C 同步随机基线预实验

## 目的与边界

TASK-009C 在已冻结的 TASK-009B 胃部覆盖环境上验证单环境 10 Hz 同步回合、七种无观测
随机策略和 HOLD 诊断。它不实现学习模型、奖励、自动调参、多环境并行或覆盖启发式。

物理频率固定为 240 Hz；每个策略动作保持 0.1 秒并推进 24 个物理子步。每个动作边界只
采集一帧新的 720p RGB，Actor 观测与面积覆盖评估必须使用相同帧号和相同图像摘要。

## 冻结输入

- 验证位姿：`validation-0006/0011/0015/0017/0019`。
- 力度：MOVE 0.70--1.40 mg，VIEW 0.20--0.50 mg，UP 0.80--1.05 mg。
- 可见性：70 mm、面积加权、120 度圆形视野、胃腔侧法向和第一命中遮挡。
- 不可达区域：`configs/task009b/unreachable_region_v1.json`。
- 完整策略、种子、执行顺序和曲线样式：
  `configs/task009c/random_baseline_preexperiment_v1.json`。

## 策略

- R1：每个边界独立等概率采样模式与力度。
- R2：随机模式和力度保持 5 个边界。
- R3：随机模式和力度保持 10 个边界。
- R4：0.8 概率保持模式，0.2 概率切换；力度截断随机游走。
- R5：MOVE 偏置，0.9 概率保持模式与力度。
- R6：与 R5 相同，但非 HOLD 力度固定为 0.5。
- R7：MOVE 阶段与 VIEW/UP 观察阶段交替，每阶段保持 5--20 个边界。
- HOLD：始终输出零 Actor 力，只保留重力、接触和摩擦动力学。

策略对象只接收自身随机数生成器和动作历史，不接收 RGB、覆盖率、位姿、速度、接触或胃
壁几何。特权状态仅由运行器记录，不能反馈给策略。

## 运行命令

在本任务实现分支工作树执行：

```bash
cd /tmp/robotarm-task009c

# 五个冻结位姿 reset 验证
./run_isaaclab.sh -p scripts/stomach_coverage/run_random_baseline_preexperiment.py \
  --device cuda:0 --reset_only

# 八个三秒 GPU 冒烟回合
./run_isaaclab.sh -p scripts/stomach_coverage/run_random_baseline_preexperiment.py \
  --device cuda:0 --smoke

./run_isaaclab.sh -p scripts/stomach_coverage/summarize_random_baselines.py \
  --latest_smoke --validate_only

# 三十七个三百秒正式回合；未完成的同配置运行可按哈希恢复
./run_isaaclab.sh -p scripts/stomach_coverage/run_random_baseline_preexperiment.py \
  --device cuda:0 --formal

# 只有全部正式回合通过后才允许汇总和绘图
./run_isaaclab.sh -p scripts/stomach_coverage/summarize_random_baselines.py \
  --latest_formal --write_figures
```

## 数据完整性

每个 300 秒正式回合必须含 3001 条边界记录：`C0` 加 3000 个动作后状态。校验器拒绝缺失、
重复、乱序、非 0.1 秒时间点、非 24 子步、RGB 帧错位、累计覆盖下降、非有限状态和提前
终止；禁止插值、补零、前值填充或平滑修复。

外部工件默认写入：

```text
/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009c_random_baseline_preexperiment/
```

`latest_smoke_manifest.json` 和 `latest_formal_manifest.json` 是稳定指针。读取时必须复核目标
清单的绝对路径、字节数和 SHA-256。正式运行清单只追加；仅当既有回合日志完整、协议通过
且哈希一致时才允许恢复跳过。

## 结果解释

覆盖低、长时间无新增覆盖、胃壁阻挡或动作效果较弱均是有效物理结果，不是运行失败。
TASK-009C 每个策略只有五个位姿和单策略种子，仅用于验证数据链路、建立无学习基线及帮助
选择后续回合长度，不构成论文正式统计证据。
