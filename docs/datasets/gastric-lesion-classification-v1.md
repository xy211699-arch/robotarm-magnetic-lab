# 胃部病灶分类数据集 v1（初步标注）

## 文件

- Release 附件：`gastric_lesion_classification_v1.zip`
- 下载页：[胃部病灶分类数据集 v1（初步标注）](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/tag/gastric-lesion-classification-v1)
- SHA-256：`79d011e4c0bc145ede4ef11696f63cb91a133d11306e2e22ee330396ee8a925c`
- 包含 52 张胶囊相机画面、`labels.csv`、`review_required.csv` 和数据说明。

## 标注口径

`labels.csv` 使用图像级标签 `lesion_present`，表示相机画面中可见红色模拟病灶。未推断医学亚型，也未绘制病灶框或掩膜。1 张来源截图列入待复核清单，没有混入已标注图像。

## 当前限制

这是用于检查采集和标注流程的初版阳性样本，不是完整训练集：目前没有正常胃壁阴性样本，画面之间可能近重复，也尚未达到方案中每项约 300—500 张的目标规模。后续需补充正常/异常类别、视角和场景变化，并按采集轨迹分组划分 train/validation/test，避免近重复帧泄漏。仿真外观标签不代表临床诊断结论。

## 存储约定

压缩包作为仓库 Release 附件保存，不纳入 Git 历史。仓库只跟踪本说明、版本和校验值；如发布地址变更，应更新本文件。
