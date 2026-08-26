# TASK-009B 不可达胃壁区域标定与覆盖率接入

## 目的与边界

该工具用于在训练或正式评估前，人工冻结物理/解剖意义上的不可达胃壁区域。它不修改胃部
资产、70 mm 可见距离、相机模型、参数化力控制器、入口区域或历史覆盖权重。

不得仅因为当前控制器没有覆盖某处就将其排除。允许排除的典型理由包括模型开口外表面、
被资产拓扑封闭且胶囊物理上无法进入的腔体，或实验装置明确阻断的表面。

## 标定启动

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/calibrate_unreachable_regions.py \
  --viz kit \
  --reason "请在这里填写可复核的物理或解剖原因"
```

若已经存在配置并明确需要重新标定，先保留旧文件，再增加 `--overwrite`。工具不会在未显式
指定该参数时覆盖已有冻结配置。

## 操作

- `W/S`：三维游标沿世界 `+Y/-Y` 移动。
- `A/D`：沿世界 `-X/+X` 移动。
- `Q/E`：沿世界 `+Z/-Z` 移动。
- 左 `Shift`：精细速度 2 mm/s；普通速度 10 mm/s。
- `G`：将游标投影到最近的真实三角面，并添加一个种子。
- `Tab`：切换当前种子。
- `[` / `]`：将当前种子的测地半径在 10–80 mm 间以 5 mm 切换。
- `Backspace`：删除当前种子；`C`：清除全部种子。
- `S`：冻结、保存并重新加载校验；`Esc`：退出且不保存。

红色网格是多个种子测地区域的并集，绿色点为种子，黄色点为三维游标。工具采用三维游标
而不是二维屏幕坐标直接取点，避免薄层胃壁前后表面的深度歧义；每次按 `G` 仍会计算真实
最近三角面点，不使用最近顶点代替。

配置保存到：

```text
configs/task009b/unreachable_region_v1.json
```

其中包括种子世界坐标、面片编号、各自半径、各区域面片、并集面片、剩余可达面片、面积、
排除理由、操作者、胃模型哈希、面积权重哈希和配置哈希。

## 在三视图中启用

```bash
cd /tmp/robotarm-task009b

./run_isaaclab.sh -p scripts/stomach_coverage/teleop_stomach_coverage.py \
  --task Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0 \
  --device cuda:0 \
  --viz kit \
  --unreachable_region configs/task009b/unreachable_region_v1.json
```

覆盖窗口中灰色表示已排除区域，红色表示可达但尚未覆盖，绿色表示可达历史覆盖，蓝色表示
当前帧可见。终端和结构化帧日志同时输出：

- `reachable_coverage_fraction`：按剩余可达三角面重新分配面积权重后的正式指标；
- `raw_coverage_fraction`：原完整 ROI 的覆盖率；
- `excluded_area_fraction`：从原 ROI 排除的面积比例。

原字段 `coverage_fraction` 在启用配置时等同于 `reachable_coverage_fraction`，便于后续训练和
终止条件直接使用正式指标；未提供配置时两种覆盖率完全相同。

## 复标规则

胃模型几何哈希、种子、半径或理由任一变化，都会产生新的配置哈希。正式数据集、训练运行
和论文结果必须记录该哈希。不得在看到策略结果后无版本记录地修改掩膜；若重新标定，应将
不同掩膜下的结果作为不同实验版本报告。
