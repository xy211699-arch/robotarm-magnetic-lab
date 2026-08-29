# TASK-010 冻结 ResNet18 + GRU 开发种子执行报告

## 执行边界

- Windows 规划提交：`ae086b7e98c0c181c7f0bd5e2870b07aeb9d10e6`
- Linux 实施基线：`1533bfa59f3d2d7b2f1769a9890efb354a5e4de6`
- Linux 实施分支：`feature/TASK-010-cnn-gru-development-seed`
- 冻结配置哈希：`c497fc430af729e7e6fb09849330de23fdea45e568b657a766cc5e030e1d19d4`
- 当前报告只覆盖 Gate 0–4 开发门禁。`seed=991000` 的 1000 次更新、768000 转移和 250/500/750/1000 固定验证均未执行、未验证；正式种子 `991001/991002/991003` 未执行、未验证。

## 实测环境

- Python 3.12.13；PyTorch 2.11.0+cu128；torchvision 0.26.0+cu128；RSL-RL 5.4.1；Isaac Lab 12.0.0。
- GPU：NVIDIA GeForce RTX 5090；CUDA 可用。
- ResNet18 `IMAGENET1K_V1` 权重 SHA-256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。
- RSL-RL 5.4.1 的实际模型接口是 `rsl_rl.models` + `rsl_rl.modules.RNN`，没有规划文本假设的旧 `ActorCritic` 导出；实现使用项目内混合分布、循环 Actor、Critic、PPO 与 runner，不修改系统 RSL-RL。

## 门禁结论

| 门禁 | 状态 | 实测结论 |
|---|---|---|
| Gate 0 依赖审计 | 通过 | 版本、GPU、ResNet 枚举及 RSL-RL 关键源码签名均已记录。 |
| Gate 1 CPU/配置 | 通过 | `tests/stomach_coverage` 共 163 项通过；compileall 通过；Actor 特权字段扫描零命中。 |
| Gate 2 十二环境 GPU 集成 | 通过（有一项实例证据未形成） | 1200 步、24 子步/动作、GPU PhysX/相机/覆盖、RGB 同帧、真终止、无 timeout bootstrap、ResNet 不变均通过。冻结训练批次的零可达 `C0` 实测数量为 0；接受 `reachable=0, raw>0` 的代码合同已有单元测试，但本批没有自然出现该实例。 |
| Gate 3 八次短学习 | 未执行 | 待执行。 |
| Gate 4 后台/恢复/失败留痕 | 未执行 | 待执行。 |

## Gate 2 缺陷与修正

首次完整运行在第 1200 步自动 reset 时发现 ResNet 帧缓存清理晚于 Isaac Lab 基类的首次观测计算，严格帧单调检查正确中止。修正方式是在 D0 增加默认空操作的 reset 前保护钩子，TASK-010 在基类计算 reset 观测前清空视觉缓存和恢复状态；未放宽帧检查。复测完整 1200 步通过。另将 TASK-010 `render_interval` 对齐 24 个物理子步，消除每个 10 Hz 边界内的冗余渲染，不改变物理或控制语义。

## 外部证据

外部工件不加入 Git。

| 工件 | 字节 | SHA-256 |
|---|---:|---|
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate0/prerequisites.json` | 4101 | `96cd1176657680429b6b45271494283ad07786684165c5e19816f6739e24979b` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/pytest.xml` | 25373 | `d3a582299c8cc2b991fcfc7edf14dab7e3c7d7c3796553ea0ff0735c1bee5065` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/pytest.log` | 2815 | `ade803cec9251f37b9c6f05144e2c3a6c90a90bd03a1d0b07bfc98876612bc9a` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/compileall.log` | 5757 | `77c4a97fa598a59e3dd2b07200ea83cee23499dc24c5dd365e1b08dc6aea1de4` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/final_prerequisites.json` | 4101 | `96cd1176657680429b6b45271494283ad07786684165c5e19816f6739e24979b` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/actor_boundary_rg.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate2/console.log` | 24173 | `a21eb0da29307f029c276556963fe0832c3840e6cda8c19ce161c1e134faa024` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate2/summary.json` | 1162 | `8ae7dd4aa2d4834fb00067c63589aceed6bfe6500ba0db3a051ac73c741dffe5` |
