# H1N1 抗原距离与后续 clade 流行度

本目录是 `fluProfiler` H1N1 抗原距离模型的外层流程；不修改 `fluProfiler` 或 `LucaVirusTasks`。

## 问题与规则

在预测截止点 \(T\)，最近半年中抗原距离较高的 clade，是否会在随后半年占比上升或成为 top-1？

模型预测的是血清–病毒对的 HI 衍生抗原距离。对一条病毒的主分数定义为：

`平均综合距离 = 与所有 serumDate < T 的冻结血清面板的平均预测距离`

数值越大，表示该病毒与截止点前血清株预测距离越远。

对每个近期 clade，主比较为：

`最近半年 clade 的平均综合距离 − 截止点前一年总体的平均综合距离`

前一年总体包含最近半年，因此该值表示“高于年度总体多少”，不是独立组间效应；同时会报告相对前一个半年的独立对照。

## 流程

```text
全长 HA SQLite + metadata
  -> 提取 327-aa HA1、去重、复用已有 embedding
  -> TSV EPI_ISL -> 未去重 AA FASTA -> 唯一 scored full HA/query_id 回填
  -> 为缺失 HA1 生成 LucaVirus embedding
  -> 冻结血清面板 × 查询 HA1，多卡写入 SQLite
  -> 按 TSV 采样记录展开到 clade，做六个月滚动回测
```

所有输出写入 `outputs/h1n1_antigenicity/`。

## 输入

| 数据 | 位置 |
| --- | --- |
| 全长 HA 与 identity | `outputs/bert_reference_free_ranking_raw_scores.sqlite` |
| 日期、header、clade | `outputs/bert_reference_free_ranking_metadata.csv` |
| H1N1 checkpoint 与历史血清 | `fluProfiler/results/H1_HA1_v1.0/.../epoch_0050.pth`；`fluProfiler/data/dataset/H1_HA1_v1.0/processed/source.csv` |
| Nextclade 结局 | `data/raw/nextclade/H1N1/h1n1_results_MW626062_sorted.tsv` |

输入序列为 566-aa 全长 HA；模型只接受 327-aa HA1，因此使用已验证的切片 `sequence[17:344]`。

## 已完成的准备步骤

```bash
cd /home/chenyh/workspace/Alright-strain
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/antigenicity/prepare_h1n1_ha1_assets.py
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/antigenicity/build_frozen_serum_panels.py
```

默认会自动使用未去重 AA FASTA `data/raw/nextclade/H1N1/origin_seqs/AA/H1N1_All.fasta` 做样本级回填。若要保留旧的一条 scored unique sequence 一行的资产形式，显式加 `--no-sample-expansion`。

旧唯一序列模式的结果为：34,358 条原始序列、22,180 条唯一 HA1；其中 1,789 条 embedding 可复用，20,391 条需要新生成。已建立 9 个预测截止点的冻结血清面板，每个面板含 56–75 个血清条件。

如果要让 clade 聚合按 TSV 采样记录展开，而不是按去重 FASTA 行计数，需要提供未去重的样本 FASTA。若是 full HA AA FASTA，用 `--sample-aa-fasta`；若是 GISAID HA 核酸 FASTA，用 `--sample-nt-fasta`：

```bash
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/antigenicity/prepare_h1n1_ha1_assets.py \
  --sample-aa-fasta data/raw/nextclade/H1N1/origin_seqs/AA/H1N1_All.fasta \
  --nextclade-tsv data/raw/nextclade/H1N1/h1n1_results_MW626062_sorted.tsv \
  --overwrite
```

这个模式会按 `TSV EPI_ISL -> 未去重 HA FASTA -> 唯一 scored full HA/query_id` 回填：`query_ha1_registry.csv` 仍是一条唯一 HA1 一行，`query_instances.csv` 则是一条 TSV 采样记录一行，并带有 `nextclade_row` 和 `prediction_branch`。抗原性 scorer 在每个 cutoff 内仍按 `query_id` 去重，所以同一个 HA1 只计算一次；下游 clade 平均则通过 `query_id` merge 回 `query_instances.csv`，让同一 AA/HA1 在 TSV 中出现多少次就贡献多少个样本。核酸输入会翻译三个 reading frames，并只接受能唯一命中现有 scored full-HA AA 的样本。

主要文件：

- `input/missing_query_ha1.fasta`：待生成 embedding 的 HA1；
- `panels/forecast_cutoffs.csv`：近期半年、年度、前半年和未来半年的边界；
- `panels/frozen_serum_panels.csv`：每个截止点的 `serumDate < T` 面板。

## 后续代码

`score_h1n1_antigenicity_multi_gpu.py` 会在 embedding 齐全后运行：无 DDP 通信、按 hash 分卡、可中断续跑。它将写入：

- `pair_scores`：每个血清–病毒原始距离；
- `query_summary`：每条病毒的均值（主指标）、中位数、p10/p90、最小/最大距离。

### 过去毒株 reference 模式

`score_h1n1_antigenicity_multi_gpu.py` 也支持一个独立的 `past_strains` 模式。对于截止点 (T)，它使用唯一 HA1 序列定义：

- current query：((T-6\text{月}, T])；
- past reference：([T-5\text{年}, T-6\text{月}))；
- 若相同 HA1 同时出现在两个窗口，会从 past reference 中移除，避免自配对；
- past reference 与 current query 的 passage 都固定为 `<CELL>`。

这是一个 **virus–virus proxy distance**，并非原始实验条件下的 serum–virus 距离。结果默认写入按窗口和压缩参数命名的独立 SQLite，不会与血清面板结果混合。

```bash
cd /home/chenyh/workspace/Alright-strain
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/antigenicity/score_h1n1_antigenicity_multi_gpu.py \
  --reference-mode past_strains \
  --current-months 6 \
  --past-years 5 \
  --reference-passage '<CELL>'
```

在当前输入中，这一定义会产生约 1.29 亿个 unique-HA1 pair，适合多 GPU 分片续跑；不要直接按未去重实例做全配对。

### 快速 coreset：时间分层的加权代表株

若过去毒株窗口过大，可加 `--past-panel-size 192`。脚本会先按时间分层（默认每 6 个月），再在每层选择真实 HA1 代表株：75% 为按历史出现频次加权的序列多样性中心，25% 为序列离群的哨兵株。每条未选中的历史 HA1 都会归属至最近的代表株，代表株的 `reference_weight` 等于其簇覆盖的原始历史样本质量。

因此 `query_summary.mean_distance`、中位数与 p10/p90 均使用该权重；`pair_scores.reference_weight` 保留完整审计信息。以 `--past-years 3 --current-months 6` 为例，实际历史区间是 2.5 年，分为 5 个半年层，192 个代表株分配为 39/39/38/38/38。总计算量从 86,440,609 降至 2,929,344 pairs。

```bash
cd /home/chenyh/workspace/Alright-strain
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=7 \
  src/antigenicity/score_h1n1_antigenicity_multi_gpu.py \
  --reference-mode past_strains \
  --current-months 6 \
  --past-years 3 \
  --past-panel-size 192 \
  --gpu-cache-gb 36 \
  --reference-passage '<CELL>' \
  --output-sqlite outputs/h1n1_antigenicity/past_strain_3y_coreset192.sqlite
```

不要将这个压缩面板与全集结果写入同一个 SQLite；数据库元数据会拒绝参数不一致的续跑。

对于全集过去毒株面板，默认每个 worker 使用 36 GiB GPU 缓存。缓存优先保留当前 cutoff 的参考面板，并在空间不足时先淘汰较早的查询或前一 cutoff 的 embedding；若活动面板本身超过预算，才退回普通 LRU 淘汰。续跑时也应显式传入 `--gpu-cache-gb 36`，以免命令行中的旧值覆盖默认值。

`backtest_antigenicity_signal.py` 读取 `query_summary` 并写出 9 个时间截止点的 clade 明细、每窗口指标和紧凑汇总 JSON。

`split_fasta_records.py` 将待生成 embedding 的 FASTA 安全分为多张 GPU 的分片。

模型分数仅用于预测关联，不代表因果适应度或真实抗原地图坐标。
