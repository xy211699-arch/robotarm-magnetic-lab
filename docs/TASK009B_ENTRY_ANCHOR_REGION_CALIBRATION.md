# TASK-009B 稳定锚点与胃壁测地区域标定

三维包围盒和暂停物理平移方案已经取消。当前标定顺序为：使用既有10 Hz参数化力控制器
移动Dynamic胶囊、切换HOLD自然落稳、确认稳定锚点、寻找最近胃壁三角面、沿共享边图
选择测地区域。

## 启动

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/calibrate_entry_anchor_region.py \
  --device cuda:0 \
  --viz kit
```

## 操作顺序

### 1. 使用MOVE、VIEW、UP调整位置和姿态

- 按住`A`、`D`：MOVE负向、正向；
- 按住`Q`、`E`：VIEW负向、正向；
- 按住`W`：UP，相机端向世界`+Z`抬起；
- `Z`、`X`、`C`：选择`alpha=0.0/0.5/1.0`；
- `Space`：HOLD；
- `R`：恢复本次标定启动时的默认状态；
- `Enter`：停止主动控制并开始稳定检测。

胶囊始终是Dynamic刚体，物理仿真正常推进。控制器保持240 Hz物理、10 Hz控制；每个按键命令
固定执行0.1秒（24个物理步），方向在每个物理步根据实时胶囊长轴重新计算。程序不通过
Kinematic或直接写位姿实现人工移动。

### 2. HOLD自然落稳与稳定锚点

按Enter后，控制器在下一个10 Hz边界切换为HOLD并清除主动Actor力，但不清零线速度和角速度。
胶囊随后只受已有惯性、重力、胃壁接触和摩擦影响。

稳定判据为连续0.25秒同时满足：

- 质心速度不超过2 mm/s；
- 角速度不超过5 deg/s。

最长等待2秒。稳定后：

- `Y`：接受并保存锚点；
- `Backspace`：拒绝并从当前状态返回MOVE/VIEW/UP控制。

稳定超时会返回控制阶段；出现非有限状态时恢复默认状态。普通滚动、滑动和碰撞不视为故障。

### 3. 选择胃壁测地区域

接受锚点后，绿色点显示胶囊质心在胃壁上的真实最近表面点，橙红色区域显示从种子三角面
沿共享边邻接图扩展的测地区域。

- `[`、`]`：以5 mm步长在10–80 mm范围内切换；
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

## 使用已确认锚点重新选择半径

如果锚点已经通过动力学定位并确认，只需要扩大或缩小测地区域，不要手工修改JSON中的
`radius_m`。面片、顶点、面积和配置哈希都依赖半径，必须由程序重新计算。

例如直接从已确认锚点预览60 mm区域：

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/calibrate_entry_anchor_region.py \
  --device cuda:0 \
  --viz kit \
  --resume_anchor \
  --initial_radius_mm 60
```

程序会校验锚点哈希和胃壁几何哈希，恢复稳定后位姿并直接进入测地区域界面。确认高亮区域
合理后按`Enter`，程序将重新计算并覆盖`entry_region_v1.json`；锚点文件保持不变。
