# 2026-11-19候选数据下载与校验

[预发行版](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/tag/gastric-candidates-20261119-v1)，状态partial；[机器可读上传回执](release_receipt.json)。标签指向审计提交`580fbda82d3011774282d765d6c0179ffbbfc5dd`，分支后续提交保存上传回执。

| 附件 | 字节 | SHA-256 |
|---|---:|---|
| [README.md](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/download/gastric-candidates-20261119-v1/README.md) | 1,885 | b4001e72df06bfaf78f6f6c3fa85b934a8c8f115f021e76d91a2771506cfa038 |
| [SHA256SUMS](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/download/gastric-candidates-20261119-v1/SHA256SUMS) | 333 | 17d016c3409ff2d57644ccd1bdcb6cd05e1799d1dd12ee8098190da1fe89e09c |
| [archive_member_index.json](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/download/gastric-candidates-20261119-v1/archive_member_index.json) | 1,278,240 | ed60cd33991797bfd7c19bb4f0e05fc874ffd1825756a5716df9bb6f16244f81 |
| [gastric-candidates-20261119-v1.tar.zst.part-001](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/download/gastric-candidates-20261119-v1/gastric-candidates-20261119-v1.tar.zst.part-001) | 1,572,864,000 | 41836a6e204a2560a69c356b4ee54facc78016284b10f41eb8016b3f5e3d362c |
| [gastric-candidates-20261119-v1.tar.zst.part-002](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/download/gastric-candidates-20261119-v1/gastric-candidates-20261119-v1.tar.zst.part-002) | 407,144,835 | 0889c315d2f76d184ad600cf2664e726c512b9ff5ecd09ec8fe2e0e0eea553a8 |
| [package_manifest.json](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/download/gastric-candidates-20261119-v1/package_manifest.json) | 1,080 | 18163a08b2994be7d042a1cb2221317e175ae2362011af7e6ea13ba04fbfad36 |

服务器返回的6项附件名称、大小与SHA-256均匹配；4个小附件另经下载字节比对。两个数据分卷未重新下载，以上传后服务器摘要与本地已校验分卷相符为证据。每个附件严格小于2 GiB。

完整包为1,980,008,835字节，SHA-256：`5fba309f6b7b17028a0adc1502e355178444f581e8ddd998e67359b6d7f1bb08`。分卷按数字顺序合并；恢复及原始文件核验命令见[交接报告](../../reports/GASTRIC-DATA-UPLOAD-20261119.md)。
