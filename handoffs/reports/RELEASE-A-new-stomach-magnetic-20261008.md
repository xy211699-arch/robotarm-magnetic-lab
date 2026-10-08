# A版新胃联合磁控上传报告

## 实际状态

用户已选择A版。基于既有新胃联合磁控代码准备独立归档分支，完成源码索引、依赖快照和回归核对；未修改控制器、物理参数或已确认资产，未训练、未运行长时仿真。新胃正式覆盖率仍未启用。本报告不是对历史300秒长测的补充验收。

## 提交边界

- 基线及归档前HEAD：`e4fde7904b04bea4f6f99288b3dcd88546bad80f`。
- 当前分支：`release/new-stomach-magnetic-v1-20261008`。
- origin：`git@github.com:xy211699-arch/robotarm-magnetic-lab.git`。
- upstream推送地址保持`DISABLED`，不推送main。
- 新提交添加README主版本说明、`docs/RELEASE_A_NEW_STOMACH_MAGNETIC.md`、本报告、七份`dependencies/legacy_magnetic_sim/`快照及清单，并记录`docs/PROJECT_RUN_LOG.md`。正式仿真实现、训练配置及源资产无新修改。
- 基线已有的新胃USD/贴图随完整历史上传，不重新导出、还原或LFS化。
- 归档提交HEAD无法在其自身内容中自引用；推送后的完整提交哈希及远端核验结果由终端交付。
- `/mnt/isaac-linux/robotarm_magnetic_lab`的旧胃反光修改、`robotarm_magnetic_lab_task010`的未跟踪实验数据和日志均未合并。

## 本轮直接证据

首轮pytest被本机ROS的自动插件阻断：缺少`lark`；没有安装依赖或修改测试，而是关闭无关插件自动加载后重试。

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH="$PWD/source/robotarm_magnetic_lab" \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/stomach_migration tests/magnetic_following -q
```

结果：65 passed，退出码0；50条兼容性弃用警告，不涉及测试失败。运行时间4.32秒。工作目录：`/mnt/isaac-linux/isaacsim/.worktrees/new-stomach-coverage-regenerate`。这些是代码/几何单测，不是GPU长时动力学验收。

- 六份依赖源用`cmp`确认完全相同，退出码均0；URDF副本统一LF并去除一处导出器注释末尾空格，以ElementTree.canonicalize忽略注释比对标准化XML，结果一致、退出码0。完整路径与源/快照SHA-256见`dependencies/legacy_magnetic_sim/snapshot_manifest.json`。
- `UsdUtils.ComputeAllDependencies`只读解析外部Stage：19个layer、0个未解析依赖。
- `git ls-remote --heads origin`核对归档分支尚不存在；远端已有TASK-010视觉依赖分支，不代表A版已上传。
- `git diff --check`初次检查通过；提交前再次检查。

没有新增外部实验工件。既有候选图像/掩码留在Linux工件目录，不加入本次提交。

## 未验证事项与后续

本次没有重跑新胃GPU仿真、300秒耦合安全、双相机录像或正式覆盖计算，也没有训练。外部机器人USD及第三方vendor不在本次上传中，版本说明明确列出固定路径依赖，不能把本分支称为跨机器零配置安装包。新胃候选不可达区仍需人工确认并正式接线。

本轮不对应新的Windows任务合同；按用户明确指定的A版归档请求执行，且用户明确授权目标fork上传。
