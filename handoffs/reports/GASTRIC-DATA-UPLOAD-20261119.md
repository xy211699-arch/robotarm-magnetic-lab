# 胃部数据整理与上传交接

状态为`partial`：原始证据清点、离线技术核验和分卷归档已完成；分类语义及500张距离数据尚未正式验收。本次延续`handoff/gastric-data-upload-20261119`，fetch核对的基准为`75364992141eb148186305c88c2c738999fdaf1a`。审计提交head为`580fbda82d3011774282d765d6c0179ffbbfc5dd`，已推送并经`ls-remote`逐字核对；Release标签固定此提交。最终交接head为保存上传回执的本报告所在提交，身份以Git历史及推送后的远端核验为准，避免在提交内引用自身SHA。

## 工件与复现

采集源目录为`/home/multirobo/gastric-single-frame-20261117/`，数据根为其`orbit_v6/`。源目录全部5661个文件、20,606,511,887字节已实际计算SHA-256，审计期间未变化。原图、原始标签、逐帧元数据、深度/实例缓冲、场景快照、完整日志、scout/pilot、标定历史、点云及源脚本全部保留在Git外工件中，不覆盖采集证据。

Git保存[脚本和使用方法](../../scripts/gastric_capture/README.md)、[数据定义与策略](../../configs/gastric_capture/delivery_20261119.json)、[清单/复核记录/摘要](../artifacts/gastric-data-20261119/README.md)、[分卷清单](../artifacts/gastric-data-20261119/package_manifest.json)及[校验值](../artifacts/gastric-data-20261119/SHA256SUMS)。完整原始文件索引在`all_file_index.csv`，批次/配置在`batch_index.csv`和`batch_configs.json`，缺失清单在`missing_files.json`。

本地审计为`/home/multirobo/gastric-delivery-20261119/audit-v1/`，包为其同级`package-v1/`。归档包括`capture_source/`（全部原始证据）、`audit/`（完整审计、总览、外部资产拷贝）、`repository_context/scripts/gastric_capture/`（打包时脚本版本）。最终Release和上传回执作为包外元数据保存，不改写已校验归档。

## 实际数量与标签

类别映射`normal=0`正常胃壁、`raised=1`肿起、`ulcerated=2`溃烂。分类目标为100/50/50。

| 类别 | 完整批次候选 | 未完成批次记录 | 原标签记录合计/待复核 | 最终批准 |
|---|---:|---:|---:|---:|
| normal | 100 | 8 | 108 | 0 |
| raised | 旧50+新50 | 17+12 | 129 | 0 |
| ulcerated | 旧50+新50 | 43 | 143 | 0 |
| 合计 | 300 | 80 | 380 | 0 |

原版200与替换候选版200共享正常100个样本身份，合计300个完整候选，不能相加当作400个独立样本。`candidate_versions.csv`保留版本成员关系；`classification_index.csv`保留每帧来源与身份。旧病灶17+43条以及新肿起12条未完成记录均存在完整批次重复位姿，不能重复计入目标。

此次扫描发现后续肿起批次`collect_small_raised_20261118T165946_31011e`仅12条；`collect_small_raised_20261118T170633_edea65`为50条替换候选；`collect_small_ulcerated_20261118T171137_df6403`为50条替换候选。批次时间、源scout、脚本哈希、配置及完整状态已记录，时间较新不构成质量批准。

全体558份orbit逐帧记录中，499条位于30–60 mm；按图片字节/像素、位姿、近位姿和近图像的连通关系保守去重后，选出411个距离技术候选（分类来源308、pilot 22、scout 81），至少还缺89张。最终验收数量为0，未声称距离任务完成。`distance_index.csv`列出有效范围、去重簇、代表样本、分类共用身份及划分状态；其他output/debug文件随包保存，不直接计数。

## 本次直接核验

审计于2026-11-19 01:38:04–01:38:49 UTC运行，系统Python3.12.3、NumPy1.26.4、Pillow10.2.0。974个图片文件全部解码成功，558份canonical样本的801×801尺寸、SHA-256、原始校验清单、别名图片身份、位姿有限性/四元数归一化、元数据一致性和保存缓冲检查均无技术失败。4项外部资产实际哈希均匹配，源Python语法解析无错误。

距离标签单位mm，定义为相机光心沿局部−Z光轴至首个可见胃壁交点；位置m，四元数wxyz，相对登记参考视线5°–10°。本次实际读取中心深度与实例缓冲，核对保存的几何标签、胃壁实例、暂停时间轴及渲染同步记录；没有启动GPU场景回放或重算全场景首交点。静态位姿采样允许`trajectory_id`为空，保留真实`capture_group_id`和scout来源。

重新计算图像SHA-256、像素SHA-256与256位dHash；未发现独立canonical样本间完全相同图片。全体558记录检出82对精确位姿、20对近位姿（小于2 mm且小于5°，不含精确对）、10对近图像（dHash汉明距离小于4），类型可重叠且含范围外记录，不能直接把配对数量相加扣除。原数据不删除，距离选择按连通簇保留一个技术合格且范围内的代表。

35页总览覆盖558个样本并已全部查看，本次恢复执行另查看5张原图，具体依据见`visual_review_summary.json`及`classification_visual_review.csv`。总览未见明显软件UI/覆盖文字；原分辨率逐图语义检查未完成。旧肿起图同时可见左上角被裁切的溃烂，新溃烂图仍可见右下/底部被裁切的肿起，正常图存在边缘红色片段、亮前景/暗开口与场景边界疑点。建议标签未确定则留空，所有分类记录保持`pending_view_review`，文件核验通过不等于类别通过。

未取得指定的`classification_visual_review_200.csv`。历史`image_reviews/review_20261118T095704Z_cb480d/image_integrity_checks.json`记录200图/4资产零问题，本次重算该报告文件SHA-256为`7fb5ea3ecb32f7f39c188acdb548fc814b4511bf685ff5af64b41aefd90ef0e3`；该历史结果与此次`capture_checks.json`明确区分。

## 来源、缺失与未验证项

正常完整批次5个来源组；原版两类病灶各仅一个来源组，替换版各增加一个登记参考。`source_relations.csv`保留同scout和重复位姿关系，不通过更改组号制造独立性；同胃资产与相关参考不足以证明已有独立训练/验证/测试三集合。全部`split`未分配，不随机拆分近重复帧。

5个中断目录仅保存快照和脚本，缺少image/sample/depth/instance/config/replay/checksum，具体路径见`missing_files.json`，未计为有效样本。外部4项模型/纹理依赖已保留路径、版本身份和实际哈希，文件拷贝随`audit/external_assets/`交付；迁移需映射场景绝对路径。采集runtime的Kit版本查询失败，准确Kit构建尚未取得，不能声称已独立回放。仍待确定整图单标签或中央目标分类口径、完成原图语义复核、补足独立来源及距离候选并执行GPU首交点回放。本次未采新图、未训练。

## 数据包和恢复

新版本使用独立[候选数据预发行版](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/tag/gastric-candidates-20261119-v1)，不修改已有`gastric-lesion-classification-v1`和无标题空草稿。先保存草稿，所有附件上传并核对后已于2026-11-19 02:03:52 UTC发布为prerelease。Release ID为450783847，6个附件全部`uploaded`；实际地址、大小、校验值见[下载清单](../artifacts/gastric-data-20261119/DOWNLOADS.md)及[上传回执](../artifacts/gastric-data-20261119/release_receipt.json)。服务器SHA-256与本地逐项相符，4个小附件另经实际下载字节比对；两个大分卷未重新下载。

完整包`gastric-candidates-20261119-v1.tar.zst`为1,980,008,835字节，SHA-256：`5fba309f6b7b17028a0adc1502e355178444f581e8ddd998e67359b6d7f1bb08`。Release仅上传分卷和说明/清单，避免完整包与分卷重复上传。

| 分卷 | 字节 | SHA-256 |
|---|---:|---|
| .tar.zst.part-001 | 1,572,864,000 | 41836a6e204a2560a69c356b4ee54facc78016284b10f41eb8016b3f5e3d362c |
| .tar.zst.part-002 | 407,144,835 | 0889c315d2f76d184ad600cf2664e726c512b9ff5ecd09ec8fe2e0e0eea553a8 |

每卷严格小于2 GiB；第一卷1500 MiB。已执行`zstd -t`并流式解压逐个核对5731个成员的SHA-256/字节数，全部通过；未向磁盘展开20GB副本。成员合计20,669,594,101字节，清单摘要见`package_manifest.json`。另核对顺序拼接分卷的哈希等于完整包。

在空目录中下载附件，先对两卷分别核对`SHA256SUMS`中的相应项，再恢复：

```bash
cat gastric-candidates-20261119-v1.tar.zst.part-001 gastric-candidates-20261119-v1.tar.zst.part-002 > gastric-candidates-20261119-v1.tar.zst
sha256sum -c SHA256SUMS
zstd -dc gastric-candidates-20261119-v1.tar.zst | tar -xf -
cd capture_source
sha256sum -c ../audit/SHA256SUMS.source
```

## 实际命令与仓库交付

| 操作 | 结果/范围 |
|---|---|
| `git worktree list`、`git status --short`、fetch指定分支 | 保护其他工作树修改；交接工作树原先干净；远端基准7536499未变化 |
| 普通沙箱`pwd`，cwd为任务工作树 | 退出1：`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`；没有执行到shell命令 |
| 审批机制下相同`pwd` | 退出0，路径为`/home/multirobo/gastric-data-upload-20261119`；后续必要命令按审批执行 |
| `/usr/bin/python3 scripts/gastric_capture/audit_delivery.py --source /home/multirobo/gastric-single-frame-20261117 --out /home/multirobo/gastric-delivery-20261119/audit-v1` | 退出0，以上为本次实测结果 |
| `/usr/bin/python3 scripts/gastric_capture/package_delivery.py --source /home/multirobo/gastric-single-frame-20261117 --audit /home/multirobo/gastric-delivery-20261119/audit-v1 --repo /home/multirobo/gastric-data-upload-20261119 --out /home/multirobo/gastric-delivery-20261119/package-v1` | 退出0，压缩/成员/拼接哈希全部通过 |
| `python scripts/research_workflow.py check` | 文档和上传回执更新后退出0，管理路径、文档链接及任务引用通过 |
| `gh api repos/xy211699-arch/robotarm-magnetic-lab/releases` | 退出0，核对已有Release及附件，原版本不覆盖 |
| `git commit -m "Archive gastric capture audit and candidate package manifests"`、`git push origin HEAD:refs/heads/handoff/gastric-data-upload-20261119`（SSH443） | 均退出0，审计提交580fbda；`git ls-remote --heads`与本地HEAD完全一致 |
| `gh release create gastric-candidates-20261119-v1 --repo xy211699-arch/robotarm-magnetic-lab --target 580fbda82d3011774282d765d6c0179ffbbfc5dd --draft --prerelease --notes-file /home/multirobo/gastric-delivery-20261119/package-v1/README.md`（另设中文标题） | 退出0，新建独立草稿 |
| `gh release upload gastric-candidates-20261119-v1 --repo xy211699-arch/robotarm-magnetic-lab`，追加DOWNLOADS所列6个本地文件路径 | 退出0，没有clobber/覆盖；6个服务器摘要全部匹配 |
| `gh release edit gastric-candidates-20261119-v1 --repo xy211699-arch/robotarm-magnetic-lab --draft=false --prerelease --latest=false` | 退出0，公开发布为预发行版 |
| `gh release download gastric-candidates-20261119-v1 --repo xy211699-arch/robotarm-magnetic-lab --pattern README.md --pattern SHA256SUMS --pattern package_manifest.json --pattern archive_member_index.json --dir /home/multirobo/gastric-delivery-20261119/download-verification` | 退出0，4个下载文件与本地逐字节相同 |
| `gh api`查询发布tag、Release和附件 | 退出0，tag指向580fbda，draft=false，prerelease=true，6个附件名称/大小/摘要一致 |

SSH22超时和HTTPS证书错误通过GitHub SSH443恢复，仍使用已知主机校验：`git -c core.sshCommand='ssh -o BatchMode=yes -o ConnectTimeout=8 -o Hostname=ssh.github.com -o HostKeyAlias=github.com -p 443' fetch origin handoff/gastric-data-upload-20261119`。推送使用同一SSH配置，禁止force；最终检查暂存内容只包含本任务脚本、配置、说明和必要清单。暂存检查发现生成CSV的CRLF被视为空白问题，仅对Git副本转换为LF并核对所有CSV字段相等，Git外审计及数据包未改写；修复后`git diff --cached --check`通过。

## 初次网页交接历史

此前初次网页交接在`b8c4986a1c138a5a090e5b9202eb846b2b98ea70`基础上归档8个脚本、README、source_manifest及报告，远端最近提交为本次基准7536499。本次保留8个源脚本字节身份。此前ZIP CRC、语法与字节比对通过，仅代表脚本接收；稀疏检出补齐后工作流检查通过。网页与本地预备提交具有不同身份，以远端Git历史为准。旧源脚本末尾空行按原件保留。
