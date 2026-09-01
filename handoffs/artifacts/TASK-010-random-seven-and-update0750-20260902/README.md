# TASK-010 七随机策略与 update-0750 补测工件

本目录包含一个完整 `tar.zst` 归档的两个普通Git分片。归档没有修改Linux原始工件。

恢复命令：

```bash
cat task010-random-seven-and-update0750-20260902.tar.zst.part-* \
  > task010-random-seven-and-update0750-20260902.tar.zst

sha256sum task010-random-seven-and-update0750-20260902.tar.zst
tar --zstd -xf task010-random-seven-and-update0750-20260902.tar.zst
```

合并归档应为 `123923391` 字节，SHA-256应为：

```text
15f800a38877f8544623e9cdfe24b704810fb765c5e467229845cd3fbfb05b55
```

归档包含七随机策略150秒完整运行，以及991001、991002、991003三个种子的
`update_0750`冻结20位姿验证目录。详细结论和局限见对应交接报告。

`SHA256SUMS`记录合并归档及两个分片的哈希；`SOURCE_SHA256.txt`记录归档内关键源文件哈希。
