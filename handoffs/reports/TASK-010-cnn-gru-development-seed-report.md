# TASK-010 冻结 ResNet18 + GRU 开发种子执行报告

## 执行边界

- Windows 规划提交：`ae086b7e98c0c181c7f0bd5e2870b07aeb9d10e6`
- Linux 实施基线：`1533bfa59f3d2d7b2f1769a9890efb354a5e4de6`
- Linux 实施分支：`feature/TASK-010-cnn-gru-development-seed`
- 报告提交前实现 HEAD：`40d0adbc5ba1b00e3cf182c3d2bae3a080712bd0`
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
| Gate 1 CPU/配置 | 通过 | 最终回归 180 项通过；compileall 通过；Actor 特权字段扫描零命中。 |
| Gate 2 十二环境 GPU 集成 | 通过 | 12 环境、1200 步、24 子步/动作、GPU PhysX/相机/覆盖、RGB 同帧、真终止、无 timeout bootstrap、ResNet 不变均通过；受控 `train-0419` 实际得到 `reachable C0=0`、`raw C0=0.0067469315` 并完成有效 reset。 |
| Gate 3 八次短学习 | 通过 | `seed=991010`，12 环境，8×64 步，共 6144 转移；update 4 严格恢复后完成 update 8；Actor/Critic 参数 L2 变化分别为 7.201421/1.023415，全部有限。 |
| Gate 4 后台/恢复/失败留痕 | 通过 | 2-update smoke 自然完成，update 2 严格恢复后生成 update 3；受控失败完整保留 traceback 且状态为 `failed/exit 23`。 |

## Gate 2 缺陷与修正

首次完整运行在第 1200 步自动 reset 时发现 ResNet 帧缓存清理晚于 Isaac Lab 基类的首次观测计算，严格帧单调检查正确中止。修正方式是在 D0 增加默认空操作的 reset 前保护钩子，TASK-010 在基类计算 reset 观测前清空视觉缓存和恢复状态；未放宽帧检查。复测完整 1200 步通过。另将 TASK-010 `render_interval` 对齐 24 个物理子步，消除每个 10 Hz 边界内的冗余渲染，不改变物理或控制语义。

Gate 3 首次运行在 update 4 保存成功后严格恢复时发现 `torch.load(map_location=cuda)` 将 CPU RNG ByteTensor 映射至 CUDA，PyTorch 正确拒绝 `torch.set_rng_state`。失败工件完整保留；修复为恢复 RNG 前显式转回 CPU，新增 CUDA map-location 回归测试。随后补齐逐边界/逐回合权威日志、分类与条件 Beta 熵、价值解释方差、吞吐、显存及官方 ResNet 权重身份后，重新执行 Gate 3；最终 update 4→8 严格恢复通过。

Gate 4 的监督器在训练前原子写入命令、UTC 时间、主机、PID、干净 Git HEAD、配置文件/哈希、种子、12 环境、计划更新数、Python/PyTorch/Isaac Lab/RSL-RL、驱动及 GPU 信息，并在 worker 启动前复核配置与 Git 状态。正常运行 `20260829T074917.271529Z-98a03b7c` 完成 update 2；独立恢复运行 `20260829T075028.710875Z-7db3fb50` 生成 update 3；失败运行 `20260829T075038.108251Z-4fffb00f` 按预期以退出码 23 结束。完整开发种子没有启动。

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
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate1/final_pytest.xml` | 28051 | `b46b9750b6f7ebbdcd1e5aee85a1c66fa8663262bd922855a34001d550b50d6a` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate2/summary.json` | 1340 | `b7b38f49041dcb555ba503953689f6c7efb909c47e32cc7aa20aba721631fd23` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate3/summary.json` | 763 | `e23329fa670f4cefb85aa345bc1060e6719eba4de4defd92be616f3c99654dc5` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate3/metrics.jsonl` | 5197 | `699dcbfd56b72f678ce4ed6da8bc7bd5d6af38c11b955716912de1a24937ad04` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate3/boundaries.jsonl` | 939138 | `446b6b4e73ec9a12944d08286cdd2ae452a459c4b0a5f7c30273aabd9a83cb48` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate3/checkpoints/update_0004.pt` | 9102021 | `bcefe9b6a58035232eae0a048b2bbebb86567c5b5155b30469999992f1c3742f` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate3/checkpoints/update_0008.pt` | 9102341 | `62676f0eb621028fb99fb0a3944486d66d8637ad5cd6f938d3691b8e559d994f` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke/20260829T074917.271529Z-98a03b7c/launch_manifest.json` | 2025 | `cce35f84457f888d0cc16b969ce66cc43ff6496ecec1ce4d70f1957954dabcb8` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke/20260829T074917.271529Z-98a03b7c/status.json` | 678 | `bddf97892a92e373247f38b53330575d0e472d22cadb094980d9940cf8dc39cb` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke/20260829T074917.271529Z-98a03b7c/checkpoints/update_0002.pt` | 9102021 | `495f2db684d9bb9ccce3699d3cd24c930d5e6030acc013de84dcd0efc8fd52e7` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke/20260829T075028.710875Z-7db3fb50/launch_manifest.json` | 2526 | `c437730ccd3d0af3a07ec07bf7c32ab9ecea8b269c9862b95e7d5238d72982bd` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke/20260829T075028.710875Z-7db3fb50/status.json` | 677 | `64b6b338e6577886857baf5d260469f611c718141bb28c0773b90776104b68fb` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/smoke/20260829T075028.710875Z-7db3fb50/checkpoints/update_0003.pt` | 9102341 | `7ce66333e76a96ba695c07544dffb27023020ffe5913094f95731483884321c8` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/failure/20260829T075038.108251Z-4fffb00f/status.json` | 606 | `0e10459eaf115e8ef3605476357e7a19e29bad8a8d3abccbdfa9a58ba2cdf63e` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/failure/20260829T075038.108251Z-4fffb00f/console.log` | 223 | `d2935768b8f355931ccacdec608c0b55065cada09f8efc42820d767f75db401c` |
| `/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate4/failure/20260829T075038.108251Z-4fffb00f/events.jsonl` | 477 | `ab9770dc54015e1ffac0172e43f765c5696451c4ff14375f717e10646ab6cb60` |

## 未执行与未验证边界

- 开发种子 `991000` 的 1000 次更新、768000 转移与 update 250/500/750/1000 固定验证：未执行、未验证。
- 正式训练种子 `991001`、`991002`、`991003`：未执行、未验证。
- TASK-009C 的七个随机策略未升级为论文比较门禁；Gate 3 只证明短学习链路工作，不代表覆盖改善、收敛或优于随机策略。
