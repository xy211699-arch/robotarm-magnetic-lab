# 10 Hz参数化力控制器平面可视化

该入口只调用新的10 Hz控制器，不调用旧的1秒宏动作控制器。

## 时序合同

- 物理频率：240 Hz；
- 控制频率：10 Hz；
- 每个控制周期：0.1秒、24个物理步；
- 按住动作键：每0.1秒继续发送同一模式和力度；
- 松开动作键：下一个控制边界立即发送HOLD，最坏延迟0.1秒；
- 每个物理步都会根据当前`xyzw`姿态重新计算胶囊长轴、相机端球心及世界施力方向。

## 启动

```bash
cd /tmp/robotarm-task008-retry

./run_isaaclab.sh -p scripts/parameterized_force/teleop_table_10hz.py \
  --viz kit \
  --render_fps 120
```

启动后点击主场景窗口取得键盘焦点。

## 英文键盘映射

| 按键 | 功能 |
|---|---|
| 按住`D` / `A` | MOVE正/负 |
| 按住`E` / `Q` | VIEW正/负 |
| 按住`W` | UP，相机端向世界+Z抬起 |
| `Z` | `alpha=0.0`：MOVE/VIEW/UP为0.70/0.30/0.70 mg |
| `X` | `alpha=0.5`：MOVE/VIEW/UP为0.95/0.60/0.85 mg |
| `C` | `alpha=1.0`：MOVE/VIEW/UP为1.20/0.90/1.00 mg |
| 按住`Space` | 强制HOLD；松开后恢复仍按住的上一个动作键 |
| `R` | 重置场景并释放全部按键状态 |
| `P` | 保存当前胶囊相机图像 |
| `Esc` | 清力并退出 |

若同时按住多个动作键，最后按下的动作生效；松开后恢复到仍按住的前一个动作。单周期只会激活一个模式。

## 肉眼观察要点

- MOVE：胶囊质心沿当前长轴的水平法向移动；正负方向应相反；
- VIEW：相机端受到单点侧向力，长轴观察方向应向正/负两侧转动；允许伴随一定平移；
- UP：必须始终是相机端抬起，不应由非相机端起身；
- 松键：下一0.1秒边界后不再存在Actor外力，但胶囊可继续依靠惯性、重力和接触运动。

## 输出

每轮输出位于：

```text
/tmp/parameterized-force-table-10hz/<UTC时间戳>/
```

- `control_cycles.jsonl`：每个10 Hz周期的控制量、姿态、速度和末物理步遥测；
- `snapshot_*.png`：按`P`保存的胶囊相机画面；
- `session_summary.json`：退出原因、周期数和时钟合同。

## 稳定接触MOVE位移标定

MOVE的单周期效果采用配对HOLD校正：

```text
corrected = dot(active_end - active_start, command_direction)
          - dot(hold_end - hold_start, command_direction)
```

每次恢复姿态后必须先以HOLD重建接触，并连续两个周期满足接触有效、线速度和角速度门槛，
之后才记录0.1秒动作起点。独立扫描入口：

```bash
./run_isaaclab.sh -p scripts/parameterized_force/calibrate_move_displacement.py \
  --headless \
  --device cpu \
  --samples 5 \
  --ratios 0.20,0.25,0.28,0.30,0.32,0.34,0.48,0.49,0.50,0.51,0.52,0.53
```

该入口仅替换当前试验的MOVE测试比例，不会修改正式控制器范围、物理材质或资产。
