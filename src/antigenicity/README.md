# H1N1 抗原新颖度与后续 clade 流行度

本目录是对 `fluProfiler` H1N1 抗原距离模型的**外层、可恢复**分析流程。
它不修改 `fluProfiler` 或 `LucaVirusTasks` 的任何代码；两者仅作为已训练模型、embedding 存储和 embedding 生成器使用。

## 要回答的问题

在一个病毒被采样时，它相对于当时以前可用 H1N1 血清面板的预测抗原距离越大，所在 clade 是否越可能在随后六个月成为高比例或 top-1 clade？

模型的单次输出不是“单条序列的抗原性”，而是：

`predicted_distance(serum_HA1, virus_HA1, serum_passage, virus_passage)`

因此本流程把病毒的抗原新颖度定义为对历史血清面板的聚合距离。主分析使用：

`novelty(window) = median(predicted_distance to all sera dated before feature_window_start)`

数值越大，表示模型预测该病毒离既往血清株越远。也会同时保存均值、10/90 分位数、最小值和最大值，供敏感性分析。

## 固定的数据契约

| 项目 | 位置 | 用途 |
| --- | --- | --- |
| 已打分 H1N1 全长 HA | `outputs/bert_reference_free_ranking_raw_scores.sqlite` | 原始序列、不可变 identity 与 BERT 分数；不作为抗原模型输入特征 |
| 日期与完整 header | `outputs/bert_reference_free_ranking_metadata.csv` | collection date、EPI ID 与后续 Nextclade 对应 |
| H1N1 checkpoint | `fluProfiler/results/H1_HA1_v1.0/SCMS-FiLM-H1/full_data_interpretation/H1N1/checkpoints/epoch_0050.pth` | 预测 HI 衍生抗原距离 |
| 历史血清数据 | `fluProfiler/data/dataset/H1_HA1_v1.0/processed/source.csv` | 血清 HA1、血清日期、passage 与既有 embedding ID |
| 基础 embedding | `fluProfiler/data/embedding/files/` | 已有 H1 HA1 / 血清 embedding |
| Nextclade 标注 | `data/clade注释_sorted/H1N1/h1n1_results_MW626062_sorted.tsv` | clade 与未来窗口结局 |

输入的 34,358 条序列均为 566 aa 全长 HA。已用 fluProfiler 同源的全长/HA1 配对记录验证：H1 HA1 几乎总是 `full_HA[17:344]`（0-based、长度 327）。资产准备脚本会强制检查此长度，绝不将 566 aa 直接送入 HA1 checkpoint。

## 流程

```text
SQLite 全长 HA + metadata
        |
        |  1. 提取 full_HA[17:344]；按 HA1 去重；保留每个原始 identity
        v
query_instances.csv + query_ha1_registry.csv + missing_query_ha1.fasta
        |
        |  2. LucaVirus 仅为缺失 HA1 生成 matrix_<embedding_id>.pt（可分卡、可断点）
        v
历史血清面板 × 查询 HA1
        |
        |  3. 每个 feature window 冻结 serumDate < feature_window_start 的血清面板；逐查询原子写 SQLite
        v
query_antigenicity.sqlite
        |
        |  4. 与 Nextclade 按唯一 EPI_ISL ID 对齐；六个月滚动回测
        v
clade-window 指标、相关性、top-1 准确率、图
```

所有生成物放在 `outputs/h1n1_antigenicity/`，不污染输入目录。

## 第 1 步：准备 HA1 与 embedding 清单

```bash
cd /home/chenyh/workspace/Alright-strain
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/antigenicity/prepare_h1n1_ha1_assets.py
```

默认输出：

| 文件 | 内容 |
| --- | --- |
| `input/query_instances.csv` | 一行对应一个原始 scored sequence；保留 SQLite identity、header、日期和 `query_id` |
| `input/query_ha1_registry.csv` | 一行对应一个唯一 HA1；包含 sequence hash、HA1、本地 `embedding_id` 与是否复用 fluProfiler embedding |
| `input/missing_query_ha1.fasta` | 仅缺失 embedding 的唯一 HA1，FASTA ID 正是后续 `matrix_<embedding_id>.pt` 的 ID |
| `input/asset_manifest.json` | 输入 hash、切片规则、精确计数和生成时间 |

脚本默认拒绝覆盖已有文件。确认重新生成时才加 `--overwrite`。

## 推荐的统计设计（当前默认）

1. **历史面板**：在每个 feature window 的起点冻结为 `serumDate < feature_window_start` 的 H1N1 血清条件；同一 `seq_id_a + serumPassCat + serumName` 只保留一次。该窗口内所有病毒共享同一面板，因此严格模拟“窗口开始时可作出的预测”。
2. **passage**：查询病毒固定为 `<CELL>`，因为训练集中 H1N1 查询病毒绝大多数为 `<CELL>`。此项会写入 manifest；后续可用 `<EGG>` 和 `<NONE>` 重跑敏感性分析。
3. **clade 特征**：以每个六个月 feature window 内所有已打分查询为单位，先求每条查询的 novelty，再在 clade 内求中位数（主指标）、均值和最大值。
4. **结局**：下一连续六个月的 Nextclade clade share、share change 与 top-1。与既有 BERT 回测一致，COVID 低采样目标窗从主分析排除，并报告 clade 持续性基线。
5. **不做的事**：不把模型分数解释成真实人群免疫压力、因果适应度或绝对抗原地图坐标；这只是严格时间切分下的预测关联检验。

## 待确认的三项选择

这些选择都会保存为配置，主分析建议使用首项。

| 问题 | 主分析 | 敏感性分析 |
| --- | --- | --- |
| 历史参考 | 每个窗口开始日前冻结血清面板 | 每条病毒采样日前的全部可用血清 |
| 聚合 | 血清距离的中位数 | 均值、p90、最小值、最大值 |
| clade 聚合 | clade 内病毒 novelty 的中位数 | 均值、最大值、上四分位数 |

在第 2 步启动大规模 GPU embedding 前，应先确认这三项；否则后续重新定义面板时会造成不必要的计算。

## 资源与可恢复性

资产盘点显示 22,180 条唯一 HA1 中 1,789 条可直接复用 fluProfiler embedding，预计需要新增 20,391 条。embedding 生成与模型打分都将采用“按稳定 hash 分片 + 每个完成对象立即落盘”的方式；任一进程中断后，只重新处理缺失的 `matrix_*.pt` 或 SQLite 记录。
