# 病灶分类光照与旋转复测

在既有病灶分类划分上复测冻结的 DINOv3 ViT-B/16、RADIOv2.5-B 与 SigLIP2-B。三种模型沿用此前选定的权重、特征读取和线性分类头。原图、仅曝光、仅旋转和两者叠加的条件下，三种模型均达到 100% 准确率和 1.000 Macro-F1。[既有数据划分](../../dinov3-gastric-20261120/lesion/split-manifest.csv) · [复测汇总](summary.csv)

## 复测条件

使用同一批 35 张留出原图，包含正常胃壁 20 张、肿起 7 张和溃烂 8 张。对每张原图交叉使用四档顺时针旋转角 0°、90°、180°、270°，以及三档曝光 −0.25、0、+0.25 EV。曝光在 sRGB 解码后的线性 RGB 上施加。每张原图对应 11 个变换视图，共 385 个变换视图，其中仅曝光 70 个、仅旋转 105 个、旋转和曝光叠加 210 个。连同 35 张原图，每个模型共完成 420 次视图预测。[复测协议](test-protocol.json) · [视图与原图对应表](test-manifest.csv)

## 结果与范围

三种模型均正确分类了 35 张原图与 385 个变换视图。按原图统计，每个模型的 35 张原图在全部 11 种变换下都保持正确。变换视图是同一张原图的重复测量，独立原图数为 35。肿起和溃烂仍各来自一个采集来源。结果反映模型在此次曝光和旋转条件下的稳定性，独立病灶及新采集批次上的表现仍待验证。[逐变换统计](by-transform.csv) · [逐原图稳定性](by-original.csv) · [混淆计数](confusion-counts.csv)

## 复核

在 NVIDIA L40 上以冻结模型逐一预测，三个模型都复现了先前对 35 张原图的预测。随后用保存的特征和冻结分类头重放，再次得到相同的 420 条分类结果。分类头及原协议的散列记录在冻结模型清单中。复测脚本保存了曝光和旋转实现，统计脚本会核对原始划分、视图归属及预测身份后复算结果。[冻结模型清单](frozen-baseline.json) · [DINOv3 预测](dinov3/predictions.csv) · [RADIOv2.5-B 预测](radio-v2.5-b/predictions.csv) · [SigLIP2-B 预测](siglip2-base/predictions.csv) · [文件校验值](SHA256SUMS)

在本目录可用 Python 3.11 及以上版本复算汇总。生成变换图像和重跑推理使用部署环境中的原图和模型，参数可从复测脚本读取。[变换与推理脚本](scripts/retest_lesion.py)

```bash
python scripts/summarize_retest.py . ../../dinov3-gastric-20261120/lesion/split-manifest.csv
```
