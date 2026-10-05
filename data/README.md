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

## 从 GISAID 获取 HA 序列

在 GISAID EpiFlu 中按目标亚型、`Human` 宿主和 `HA` 片段检索；下载量超过 20,000 条时，按提交日期拆成多个批次。下载时选择 DNA 或蛋白 FASTA，保留至少 `Virus name`、`Isolate ID`、`Type`、`Passage details/history` 与 `Lineage` 等表头字段，并勾选移除字段首尾空格。原始下载文件应按检索日期区间命名，例如 `19760101-20141231.fasta`，并保存到 `data/raw/`。

![GISAID 检索条件](../docs/images/gisaid-search-filters.png)
![GISAID 下载配置](../docs/images/gisaid-download-options.png)

> GISAID 数据的访问、下载和使用必须遵守其注册用户协议与数据使用条款。本项目不再分发 GISAID 序列数据。

## Nextclade 注释

以 H3N2 HA 为例，先安装 Nextclade CLI，再下载与输入片段相符的参考数据集：

```bash
nextclade dataset get \
  --name 'nextstrain/flu/h3n2/ha/CY163680' \
  --output-dir data/interim/nextclade_h3n2_ha_dataset

nextclade run \
  --input-dataset data/interim/nextclade_h3n2_ha_dataset \
  --output-tsv data/raw/nextclade/H3N2/h3n2_results.tsv \
  data/raw/fasta/H3N2_HA_RNA_all.fasta
```

运行前先用 `nextclade dataset list --only-names | grep 'flu/h3n2/ha'` 确认数据集名称仍可用。原始 FASTA、使用的数据集名称和生成的 TSV 应一同记录，保证注释结果可追溯。
