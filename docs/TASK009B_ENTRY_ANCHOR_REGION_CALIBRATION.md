# TASK-009B 稳定锚点与胃壁测地区域标定

三维包围盒入口方案已经取消。当前标定顺序为：暂停物理平移 Dynamic 胶囊、自然下落、
确认稳定锚点、寻找最近胃壁三角面、沿共享边图选择测地区域。

## 启动

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/calibrate_entry_anchor_region.py \
  --device cuda:0 \
  --viz kit
```

## 操作顺序

### 1. 调整释放前位置

- `W`、`S`：沿世界坐标 `+Y`、`-Y` 移动。
- `A`、`D`：沿世界坐标 `-X`、`+X` 移动。
- 普通速度：10 mm/s。
- 按住左 `Shift`：2 mm/s精细移动。
- `Enter`：从当前候选位置释放。

此阶段胶囊仍是 Dynamic 刚体，只暂停PhysX。程序只修改世界X/Y，Z和四元数保持不变；
窗口刷新时仿真时间不得推进。

### 2. 自然下落与稳定锚点

释放前程序清除Actor力并把线速度、角速度清零。随后胶囊只受重力、胃壁接触和摩擦影响。

稳定判据为连续0.25秒同时满足：

- 质心速度不超过2 mm/s；
- 角速度不超过5 deg/s。

最长等待2秒。稳定后：

- `Y`：接受并保存锚点；
- `Backspace`：拒绝并恢复到释放前位置。

超时或出现非有限状态时程序自动拒绝并恢复。

### 3. 选择胃壁测地区域

接受锚点后，绿色点显示胶囊质心在胃壁上的真实最近表面点，橙红色区域显示从种子三角面
沿共享边邻接图扩展的测地区域。

- `[`、`]`：在10、15、20、25、30 mm五档半径间切换；
- `Enter`：确认并保存当前区域；
- `Esc`：退出程序。

区域必须只有一个共享边连通分量，否则禁止保存。

## 输出

- `configs/task009b/entry_anchor_v1.json`
- `configs/task009b/entry_region_v1.json`
- `logs/task009b_entry_calibration/<UTC会话>/events.jsonl`

终端出现`TASK009B_ENTRY_REGION_SAVED`且其中`status`为`complete`，代表两份配置均已保存并
精确重新加载。请把`TASK009B_ENTRY_ANCHOR_SAVED`和`TASK009B_ENTRY_REGION_SAVED`两行
完整输出交给执行端继续Gate 3。

