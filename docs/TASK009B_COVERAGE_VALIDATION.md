# TASK-009B 面积覆盖验收

覆盖目标继续使用旧P0模块批准的胃腔内表面，不重新划分解剖区域。每个目标三角形面积
平均分给三个顶点，累计覆盖率因此是面积比例而不是顶点数量比例。

```bash
cd /tmp/robotarm-task009b
./run_isaaclab.sh -p scripts/stomach_coverage/validate_coverage_calculation.py \
  --device cuda:0 --raycast_device cuda:0
```

验证器只读取已通过Gate 3的固定20/20/20位姿，依次检查70 mm距离、120度圆形FOV、
胃腔侧法向、CUDA第一命中、面积累计、范围、单调性和reset后的初始覆盖。数据日志和摘要
保存在`coverage_manifest_v1.json`指向的外部目录，使用前必须校验外部日志SHA-256。
