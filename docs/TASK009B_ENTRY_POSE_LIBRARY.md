# TASK-009B 有效初始位姿库

Gate 3使用已确认的`entry_anchor_v1.json`和`entry_region_v1.json`。生成器按入口面片面积
采样，候选状态在零主动作用力下真实松弛，只有满足冻结稳定、相机位置和长轴方向门限的
最终状态才会写入外部JSONL。

生成：

```bash
cd /tmp/robotarm-task009b
./run_isaaclab.sh -p scripts/stomach_coverage/generate_entry_pose_library.py \
  --device cuda:0
```

固定回载验收：

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/validate_entry_pose_library.py \
  --device cuda:0
```

Git只保存`configs/task009b/pose_library_manifest_v1.json`。1200条数据本体、拒绝记录和
live回载日志保存在清单指向的外部绝对路径；使用前必须同时校验入口配置哈希、胃壁几何
哈希和数据SHA-256。不得移动数据后仅手工修改清单路径，也不得把未通过松弛的候选补入库中。
