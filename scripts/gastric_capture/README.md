# 胃部相机采集脚本交接

本目录收录用户于2026-11-19提供的8个采集及质检脚本，源包为`脚本(1).zip`。逐文件大小、SHA-256和源包身份见[source_manifest.json](source_manifest.json)。脚本按上传字节归档，便于与已采集帧中的脚本身份对应。

| 文件 | 入口行为与用途 |
|---|---|
| capture_one_frame.py | 在Isaac Sim中运行后安排一次异步采集 |
| gastric_orbit_v8.py | 创建标定和采集面板 |
| gastric_small_views_v9.py | 加载面板后自动开始肿起50张采集 |
| start_small_views.py | 读取v8，生成并覆盖本机v9，再启动溃烂50张采集 |
| gastric_normal_views_v10.py | 加载登记的scout后安排正常胃壁采集 |
| start_normal_views.py | 使用已有面板和scout，生成并覆盖本机v10后执行 |
| move_camera_closer.py | 使用已有面板移动相机，并采一张scout |
| export_image_review.py | 后台读取7个指定批次，输出总览、文件校验及复核表 |

采集入口使用Isaac Sim Python环境及已加载胃部场景。源脚本保留实验电脑路径`/home/multirobo/gastric-single-frame-20261117/`，各版数据目录仍为`orbit_v6/`。迁移到其他电脑时须先核对路径、场景资产、scout及标定文件。v9和v10包含自动采集入口，启动前应暂停时间轴并确认目标与输出位置。

运行单帧入口时，脚本通过`__file__`记录实际执行文件，因此源代码中的历史默认文件名`capture_one_frame_v5.py`与本包文件名不同。`start_small_views.py`生成的默认溃烂版本也与本包现有肿起v9版本不同。复现某个已保存批次，应优先使用该批次的`collection_script.py`及对应配置、校验值。

本次通过8个文件的Python语法解析及归档字节比对。Isaac Sim运行验证、场景回放和采集效果仍依赖实验环境证据。数据交付进度见[交接报告](../../handoffs/reports/GASTRIC-DATA-UPLOAD-20261119.md)。

## 离线整理与交付

新增的`audit_delivery.py`和`package_delivery.py`用于离线核验及归档，独立于上述8个原始采集脚本。原脚本未改写。系统Python3.12.3、NumPy1.26.4、Pillow10.2.0已运行本次审计；打包另需`zstd`。不启动Isaac Sim、采集或训练。

```bash
/usr/bin/python3 scripts/gastric_capture/audit_delivery.py \
  --source /home/multirobo/gastric-single-frame-20261117 \
  --out /home/multirobo/gastric-delivery-20261119/audit-v1
/usr/bin/python3 scripts/gastric_capture/package_delivery.py \
  --source /home/multirobo/gastric-single-frame-20261117 \
  --audit /home/multirobo/gastric-delivery-20261119/audit-v1 \
  --repo /home/multirobo/gastric-data-upload-20261119 \
  --out /home/multirobo/gastric-delivery-20261119/package-v1
```

重跑时使用新的输出目录；原始目录不写入。审计逐文件计算SHA-256、解码图像、核对元数据/缓冲/保存校验值、检查重复并生成待复核总览。自动审计不批准语义标签；人工查看记录另存复核清单。打包保留全部原始证据及4项外部资产拷贝，流式解压核验每个成员并核验分卷拼接SHA-256，附件上限严格小于2 GiB。

本次[配置](../../configs/gastric_capture/delivery_20261119.json)、[审计索引及范围说明](../../handoffs/artifacts/gastric-data-20261119/README.md)和[交接报告](../../handoffs/reports/GASTRIC-DATA-UPLOAD-20261119.md)区分完整候选、未完成记录、技术候选和最终验收数量。场景迁移需根据外部资产索引恢复绝对引用；准确Kit构建与GPU首交点回放仍待取得/执行。
