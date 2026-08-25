# TASK-009B 胃入口三维包围盒标定

## 启动

在本机图形会话中执行：

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/calibrate_entrance_region.py \
  --device cuda:0 \
  --viz kit
```

程序只加载 TASK-009B 胃部环境并显示标定几何，不推进物理、不施加胶囊控制力。

## 按键

- `X`、`Y`、`Z`：选择要调整的世界坐标轴。
- `A`、`D`：沿当前轴负向、正向移动包围盒中心。
- `Q`、`E`：缩小、增大包围盒在当前轴上的尺寸。
- `1`、`2`、`3`：把移动和尺寸步长切换为 0.5、2、5 mm。
- `R`：恢复本次启动时的初始包围盒。
- `S`：保存当前配置，但保存本身不代表人工验收完成。
- `Esc`：退出。

窗口和终端会显示包围盒中心、三个方向尺寸、相交胃壁三角面数量、面积和共享边连通
分量数量。橙红色区域是三角形与包围盒真实相交后得到的候选入口表面；蓝色半透明区域
是世界轴对齐包围盒。若连通分量大于 1，程序会给出警告，应继续调整，避免同时选中不连续
的胃壁区域。

## 人工门禁

确认高亮区域仅对应预期胃入口表面后按 `S`。默认生成：

- `configs/stomach_entrance_region_v1.json`
- `configs/stomach_entrance_region_v1.save_summary.txt`

随后把终端的 `TASK009B_ENTRANCE_SAVED` 行和“入口区域确认正确”回复给执行端。入口未被
现场确认前，不得生成最终初始位姿库。

