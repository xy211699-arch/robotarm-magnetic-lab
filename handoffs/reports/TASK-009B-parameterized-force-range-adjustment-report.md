# TASK-009B 参数化力范围实测调整报告

状态：`implemented_live_verified`

日期：2026-08-26

## 调整依据与边界

本次范围由用户实测结果直接指定，只调整10 Hz参数化力控制器的力度映射。以下内容保持
不变：六模式ID、`alpha`定义、240 Hz物理、10 Hz控制、每边界24子步、完整0.1秒持续施力、
每物理步方向重算、MOVE双端均分、VIEW/UP仅相机端施力，以及HOLD零Actor力。

## 当前冻结范围

令`alpha`属于`[0, 1]`，`m`为运行时读取的胶囊质量，`g=9.81 m/s2`：

| 模式 | 新范围 | `alpha=0` | `alpha=0.5` | `alpha=1` | 作用点 |
|---|---:|---:|---:|---:|---|
| MOVE正/负 | 0.70--1.40 mg | 0.70 mg | 1.05 mg | 1.40 mg | 总力由两端均分 |
| VIEW正/负 | 0.20--0.50 mg | 0.20 mg | 0.35 mg | 0.50 mg | 仅相机端 |
| UP | 0.80--1.05 mg | 0.80 mg | 0.925 mg | 1.05 mg | 仅相机端，世界+Z |
| HOLD | 0 | 0 | 0 | 0 | 无Actor力 |

线性映射为：

```text
MOVE = mg * (0.70 + 0.70 * alpha)
VIEW = mg * (0.20 + 0.30 * alpha)
UP   = mg * (0.80 + 0.25 * alpha)
```

## 实现位置

- 纯控制合同：
  `controllers/parameterized_force.py::ParameterizedForceConfig`
- Isaac Lab执行配置：
  `mdp/parameterized_force_action.py::ParameterizedForceActionTermCfg`
- 冻结端点和中点断言：
  `tests/parameterized_force/test_baseline_audit.py`

纯合同与ActionTerm默认值已同步，避免离线规划、测试和仿真执行使用不同范围。

## 数据影响

- Gate 3位姿库只通过HOLD松弛，位姿本体和分组仍有效；
- Gate 4覆盖验证使用冻结位姿和相机几何，不依赖主动力度，结果仍有效；
- 旧范围下产生的MOVE/VIEW/UP响应统计不再代表当前控制器，后续训练数据必须记录本报告
  对应的代码提交和实际`force_ratio`；
- Gate 5三视图接口无需改动，会自动使用本次新范围。

## 验证与结论

自动回归59/59通过。Isaac Lab胃部环境live按HOLD、MOVE正负、VIEW正负、UP依次执行6个
0.1秒边界：全部为24个物理子步，主动模式24/24施力，HOLD 0/24，RGB每边界递增一帧，
状态与图像均有限。运行时中点遥测为MOVE 1.05 mg、VIEW 0.35 mg、UP 0.925 mg，与本报告
线性映射一致。

外部证据：

- 目录：
  `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task009b_force_range_validation/20260826_101504_295337Z`
- `environment_cycles.jsonl`：7787字节，SHA-256
  `d32cb94c4a8852bdb9674e7e51d2c823171abbea0d40a460b253e84a7d872ac2`
- `summary.json`：369字节，SHA-256
  `38121c8ebc191bcf3ede4c2d811e1b18f4e8c98216104d9c89286e30fb8d4992`

本次验证证明接口、实际比例和时序实现正确。最终运动幅度由用户实测范围定义，本报告不把
额外的主观三视图观察写成重复验收。
