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
  -> 为缺失 HA1 生成 LucaVirus embedding
  -> 冻结血清面板 × 查询 HA1，多卡写入 SQLite
  -> 与 Nextclade 对齐，做六个月滚动回测
```

所有输出写入 `outputs/h1n1_antigenicity/`。

## 输入

| 数据 | 位置 |
| --- | --- |
| 全长 HA 与 identity | `outputs/bert_reference_free_ranking_raw_scores.sqlite` |
| 日期、header、clade | `outputs/bert_reference_free_ranking_metadata.csv` |
| H1N1 checkpoint 与历史血清 | `fluProfiler/results/H1_HA1_v1.0/.../epoch_0050.pth`；`fluProfiler/data/dataset/H1_HA1_v1.0/processed/source.csv` |
| Nextclade 结局 | `data/clade注释_sorted/H1N1/h1n1_results_MW626062_sorted.tsv` |

输入序列为 566-aa 全长 HA；模型只接受 327-aa HA1，因此使用已验证的切片 `sequence[17:344]`。

## 已完成的准备步骤

```bash
cd /home/chenyh/workspace/Alright-strain
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/antigenicity/prepare_h1n1_ha1_assets.py
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/antigenicity/build_frozen_serum_panels.py
```

结果：34,358 条原始序列、22,180 条唯一 HA1；其中 1,789 条 embedding 可复用，20,391 条需要新生成。已建立 9 个预测截止点的冻结血清面板，每个面板含 56–75 个血清条件。

主要文件：

- `input/missing_query_ha1.fasta`：待生成 embedding 的 HA1；
- `panels/forecast_cutoffs.csv`：近期半年、年度、前半年和未来半年的边界；
- `panels/frozen_serum_panels.csv`：每个截止点的 `serumDate < T` 面板。

## 后续代码

`score_h1n1_antigenicity_multi_gpu.py` 会在 embedding 齐全后运行：无 DDP 通信、按 hash 分卡、可中断续跑。它将写入：

- `pair_scores`：每个血清–病毒原始距离；
- `query_summary`：每条病毒的均值（主指标）、中位数、p10/p90、最小/最大距离。

配置见 `analysis_config.json`。模型分数仅用于预测关联，不代表因果适应度或真实抗原地图坐标。
