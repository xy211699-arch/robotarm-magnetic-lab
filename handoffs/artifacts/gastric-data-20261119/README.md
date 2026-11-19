# 胃部仿真数据候选归档 2026-11-19 v1

状态 partial / prerelease。归档整理与技术质检已执行；分类和500张距离任务尚未正式验收。

源目录全部5661个文件、20606511887字节保存在 capture_source/，包括801×801原图、深度和实例缓冲、场景快照、完整日志、scout、pilot、点云、标定历史及源脚本。数据根目录是 capture_source/orbit_v6/。其他 output/debug 证据随包保存，不直接计入任务数量。

完整分类候选含 normal 100、raised旧50+新50、ulcerated旧50+新50；原版和替换候选各200条，正常100条共享同一身份，二者共300个完整候选。未完成记录另80条。全部380条原标签保留，最终通过数为0，待语义复核数为380。candidate_versions.csv 明确共享身份，不能把两个版本相加当作400个独立样本。

558份orbit样本全部通过文件、解码、缓冲、位姿和元数据一致性检查，974个图片文件解码零失败。4项外部资产重新计算哈希均匹配。source_file_index与原始校验清单的匹配不等于类别语义全部通过。

499条样本在30–60 mm内。保守去重后411个距离技术候选（分类来源308、pilot 22、scout 81），至少缺89张；最终验收数量仍为0。distance_index.csv提供每帧身份、去重簇、代表样本、与分类共用关系和未划分状态。原始558帧不删减，59条范围外记录不选入。全体样本中的82对精确位姿、20对近位姿和10对近图像可有重叠，不能将配对数简单相加用于减样本。

距离定义为相机光心沿局部−Z光轴至第一个可见胃壁交点，单位mm；位置单位m，四元数wxyz。保存的中心深度和实例缓冲已实际读取，与记录的几何距离/胃壁实例核对；没有重跑GPU场景或重算全场景首个射线交点。暂停时间轴的静态采样允许trajectory_id为空。

35张总览覆盖全部558个已保存样本，均已查看；本次恢复执行另查看5张801×801原图。原图未见明显软件UI或覆盖文字，但这不是逐图原分辨率语义验收。复核总览的标题和编号是质检工具生成的，不在原图中。旧raised和部分新ulcerated图存在混合病灶；normal有边缘病灶片段和场景边界疑点。classification_visual_review.csv记录原标签、建议标签（未确定则为空）、依据、查看范围与pending状态。指定classification_visual_review_200.csv未取得。

捕获组只保留真实来源；同scout或重复位姿的不同组号不能作为独立测试来源。原版两类病灶各一组，替换版各增加一个登记参考，但同胃资产且来源相关，不能据此声称已有独立三集合。split全部未分配。

5个中断capture目录仅留下快照和脚本，缺失文件见missing_files.json。原有200图/4资产零问题的历史报告仍在capture_source/image_reviews中，本次结果在audit/capture_checks.json，二者明确区分。

外部资产的源路径、期望/实际SHA-256及复现拷贝见audit/external_asset_index.json；拷贝保存在audit/external_assets/。场景中的绝对引用仍按证据保存，迁移时需依此映射恢复路径。运行需要匹配的Isaac Sim/Kit、USD和采集脚本依赖；采集记录中的Kit版本查询失败，准确Kit构建尚未取得，未声称独立回放成功。审计环境为系统Python3.12.3、NumPy1.26.4、Pillow10.2.0；原采集运行版本以逐帧runtime为准。

恢复方法：先依据package_manifest.json/SHA256SUMS核验下载文件。若数据包分卷，按.part-001、.part-002顺序cat合并为完整.tar.zst，再核对full_archive的SHA-256。随后执行 zstd -dc gastric-candidates-20261119-v1.tar.zst | tar -xf - 。进入capture_source后可执行 sha256sum -c ../audit/SHA256SUMS.source。完整归档已进行zstd完整性检查及每个成员的SHA-256/大小流式核验。归档无需复制覆盖原采集目录。

离线复查入口：/usr/bin/python3 scripts/gastric_capture/audit_delivery.py --source SOURCE --out NEW_AUDIT_DIRECTORY。审计不启动仿真、不写原始数据、不自动确认类别。打包入口见scripts/gastric_capture/package_delivery.py --help。

Git中的CSV清单仅将审计输出的CRLF换行为LF，已核对所有字段值一致；Git外审计原件及已校验数据包保持不变。
