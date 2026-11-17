# 已有实现与复用边界

## 仿真基础

[Isaac扩展](../source/robotarm_magnetic_lab) 已有机械臂、三轴外磁体、动态胶囊、胃部场景和相机接线，包含新胃基础任务与磁控任务。运行依赖外部Isaac、vendor和资产，本次没有重跑场景或独立复现历史GPU结果，更不能据此声称新硬件已接入。

## 磁控与执行

[轨迹模块](../source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/actuator_vector.py) 和 [磁场桥](../source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/legacy_bridge.py) 提供已有相对指令编译、伺服、磁力矩和安全检查。[新胃磁控演示](../scripts/stomach_migration/validate_new_stomach_magnetic.py) 可供后续技能适配复用；旧四秒预览等执行语义不变，不代表新技能接口已完成。

## 感知与记录

包内 [runtime](../source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime)、[coverage](../source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage)、[geometry](../source/robotarm_magnetic_lab/robotarm_magnetic_lab/geometry) 和 [ui](../source/robotarm_magnetic_lab/robotarm_magnetic_lab/ui) 已有相机同步、覆盖、坐标与记录逻辑。仿真接触和安全遥测不等于标定后的实物压阻触觉，也不能直接作为新模型的异步观测合同。

## 历史可行性验证

[learning](../source/robotarm_magnetic_lab/robotarm_magnetic_lab/learning)、[new_stomach_rl脚本](../scripts/new_stomach_rl) 和对应tests保留Actor、PPO、collector、恢复及监督实现。四组、N8和旧预算只是历史设置，不约束新模型或实验矩阵；代码存在不等于当前效果成立，历史证据见 [历史入口](archive/README.md)。

## 新方向尚缺内容

DINOv3特征提取与线性探测、正式病灶及距离数据采集仍按 [当前方案](design/2026-11-17-dinov3-gastric-vision-linear-probe.md) 交付，现有通用相机和RGB-D记录代码可复用，但不代表任务标签、几何距离真值或评测已经完成。实物触觉编码、异步多模态历史、分模式效果预测、统一技能选择与执行接管等仍需设计和验证，具体任务见 [下一阶段](NEXT.md)。
