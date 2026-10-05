# 数据分层

- `raw/`：GISAID FASTA 与 Nextclade 原始 TSV；只读，不由分析脚本改写。
- `interim/`：可由 `raw/` 再生成的合并、去重、切片等中间文件。
- `processed/`：供 forecasting 和特征模块直接消费的规范化数据集。

## FASTA 合并与去重
```bash
cd /home/chenyh/workspace/Alright-strain
```

## 1. 合并同一目录下的 FASTA

```bash
./scripts/merge_fasta.sh data/raw/nextclade/H1N1/origin_seqs
```


## 2. 按序列内容去重

```bash
python src/strain_data/deduplicate_fasta.py data/raw/nextclade/H1N1/origin_seqs/merged.fasta
```

该命令保留每个唯一序列第一次出现时的 FASTA header，生成：

```text
data/raw/nextclade/H1N1/origin_seqs/deduplicated.fasta
```

## Nextclade 审计

```bash
python src/strain_data/nextclade_audit.py
```

默认读取 `data/raw/nextclade/`，并将仅含汇总统计的审计结果写到 `outputs/nextclade_audit/`。
