# TASK-010 三正式种子自动训练与验证入口执行报告

## 执行结论

- 状态：`complete`（仅代码与CPU伪进程测试）。
- 正式训练/验证：未启动。
- 基础提交：`c2ca69f254c1dab6b444b541375b5b1b44be5ddc`。
- 已测试实现提交：`e1d29c329d08b29629dcf516174cff40ccd88438`。
- 分支：`feature/TASK-010-three-formal-seed-supervisor`。
- 报告提交后的最终远端HEAD以Linux终端交付信息为准。

## 实现范围

- 新增单次`start`后台入口，固定顺序执行991001、991002、991003的训练、验证与汇总。
- 生产命令固定为12环境、64步rollout、1000 updates、50 update检查点间隔、`cuda:0`顺序执行。
- 新增只读`status`，报告当前种子/阶段、训练update、验证位姿、最近检查点、运行时长、心跳与错误。
- 新增`paused_on_error`和人工`continue`，只重试失败阶段，不自动跳过、替换或恢复故障种子。
- 严格审计每个种子的20个冻结位姿、每条1201点真实可达面积加权累计覆盖曲线、有限性、单调性、配置及检查点哈希。
- 每个种子生成均值曲线；三个种子生成1201行总体均值和种子间总体标准差CSV。
- 新增中文操作文档及CPU伪训练/伪验证驱动。

未修改十二环境、模型、奖励、PPO、训练配置、仿真源代码或已有实验结果。

## 测试证据

执行命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh \
  -m pytest tests/stomach_coverage/test_task010_formal_seed_supervisor.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh \
  -m pytest tests/stomach_coverage -q

/mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m py_compile \
  scripts/stomach_coverage/task010_formal_seed_supervisor.py \
  tests/fixtures/task010_formal_fake_stage.py

git diff --check
```

观测结果：

- 新协调器定向测试：`11 passed`，退出码0。
- `tests/stomach_coverage`完整回归：`197 passed, 49 warnings`，退出码0。
- 警告仅为Isaac Lab配置类和Torch JIT既有弃用警告。
- 语法编译、三个入口帮助与`git diff --check`通过。
- 伪进程测试覆盖固定顺序、立即返回、禁止并发、20×1201原始曲线、单种子均值、三种子总体均值/标准差、三类无效验证、失败暂停、人工继续、陈旧状态和只读状态查询。

完整CPU回归日志：

- 路径：`/tmp/task010_formal_full_pytest_final.log`
- 字节数：2717
- SHA-256：`63c5d2bcb2ba7babbc928b5d80e72a1b39be5029d1e8d7f459feed1e3a120481`

## 正式种子未启动证据

- 991001/991002/991003相关正式训练或验证进程计数：0。
- 默认正式输出根目录`artifacts/task010_cnn_gru/formal_seeds`：不存在。
- 测试输出仅位于pytest临时目录，测试驱动受`TASK010_FORMAL_TEST_MODE=1`保护。

## 偏差与未验证内容

- 合同偏差：无。
- 未验证：真实Isaac Lab/GPU下三个正式种子的训练时长、最终覆盖表现和正式CSV数值；按任务要求本轮不得启动这些任务。
