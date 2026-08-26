# 生成数据与工件目录

## 默认位置

项目生成的数据统一保存在主仓库下：

```text
/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/
```

当前目录结构：

```text
artifacts/
├── task009b_pose_library/             # Gate 3位姿库、生成摘要、接受种子和拒绝记录
├── task009b_pose_library_validation/  # Gate 3固定回载验证日志
├── task009b_coverage_validation/      # Gate 4覆盖率验证日志
└── task009b_three_view/               # Gate 5现场会话、覆盖图片、mask和轨迹
```

`artifacts/`属于可再生成或体积可能持续增长的运行工件，已被Git忽略，不随源码提交。
Git中的`configs/task009b/*manifest*.json`保存实际数据路径、内容哈希和配置哈希。

## Git worktree处理

TASK脚本可能从`/tmp/robotarm-taskXXX`工作树运行。`scripts/_artifact_paths.py`会通过Git
common-dir自动找到主仓库，因此默认数据仍写入上述主仓库`artifacts/`，不会散落到`/tmp`。

需要临时改到其他磁盘时可设置：

```bash
export ROBOTARM_MAGNETIC_ARTIFACT_ROOT=/自定义/绝对路径
```

命令行显式传入`--output_root`或`--output_directory`时，显式参数优先。

## 历史路径说明

原目录`/mnt/isaac-linux/robotarm_magnetic_lab_artifacts`已整体迁移并删除。部分历史
`generation_summary.json`中的旧绝对路径保留为生成时溯源信息，运行时不再读取这些字段；
当前有效路径以Git清单和本文件为准。
