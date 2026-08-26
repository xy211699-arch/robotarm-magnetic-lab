# TASK-009B 三视图人工验收

## 启动

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/teleop_stomach_coverage.py \
  --device cuda:0 \
  --viz kit
```

程序从Gate 3冻结位姿库中的稳定胃内位姿启动，并同时打开：

1. 主Kit视口：外部场景，60 Hz渲染；
2. `Capsule Camera | Recorded 10 Hz`：训练/记录使用的同一胶囊RGB，不创建额外预览传感器；
3. `P0 Stomach Coverage`：隔离的覆盖工程视图，红色为未覆盖、绿色为历史累计覆盖、蓝色为当前帧可见，HUD显示面积加权覆盖率。

## 按键

按住动作键才持续提交命令；松开后自动回到HOLD。每个命令边界为0.1秒，包含24个240 Hz物理步。

| 按键 | 动作 |
|---|---|
| `D` / `A` | MOVE 正向 / 负向 |
| `E` / `Q` | VIEW 正向 / 负向 |
| `W` | UP |
| `Space` | HOLD |
| `Z` / `X` / `C` | 力度 `alpha=0 / 0.5 / 1` |
| `R` 或 `Backspace` | 恢复同一冻结初始位姿并清空覆盖 |
| `P` | 保存覆盖率快照 |
| `Esc` | 保存最终工件并退出 |

## 验收要点

- 主视口中胶囊能随MOVE/VIEW/UP产生物理响应，画面保持交互；
- 胶囊相机仅以10 Hz刷新，圆形视野和画面方向正确；
- 覆盖视图也只在10 Hz边界更新，蓝色当前区域随相机改变，绿色历史区域只增不减；
- `R`后覆盖图清空，并从首个有效RGB边界重新得到非零初始覆盖；
- 终端连续输出的`rgb_frame`严格递增，且面积覆盖百分比有限；
- 三个窗口来自同一仿真时间线，没有第二个30 Hz胶囊预览传感器。

人工确认前，TASK-009B Gate 5状态保持`needs_input`，不得标记为完成。
