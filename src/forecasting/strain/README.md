# Strain 级候选排序预测

目标不是生成未来序列，而是在截止日已经观测到的毒株中，找出最接近未来六个月流行群体的候选毒株。

```text
选截止日 T
  ↓
选候选 subclade：自动 top-1，或手动指定
  ↓
将原始序列整理为 isolate 记录，再按氨基酸序列聚合为候选 strain
  ↓
为每个候选计算当时可见的比例、趋势、Fitness、抗原性和序列新颖性
  ↓
用 T 后六个月的真实病毒群体计算标签
  ↓
候选离未来群体的平均氨基酸距离越小，标签越好
  ↓
用多个历史截止日训练排序模型
  ↓
在最新 T 输出候选毒株 Top-N
```

候选单位是唯一氨基酸序列，而不是单条 GISAID 记录。相同序列可对应多个 isolate：候选本身只保留一条序列，但其频率等于这些 isolate 的数量；同时保留代表 strain ID 以便追溯。

默认与未来六个月的**全部**病毒群体比较，而不是只比较同一个 subclade。这样预测的是“哪个已知毒株最像未来总体”；若只比较同一 subclade，必须作为单独实验标记。

所有特征只能使用 `T` 前的数据，抗原性面板也必须冻结在 `T` 前。Fitness 和抗原性是排序特征，不是氨基酸距离的一部分。

# 数据预处理与样本定义

- 先将“序列记录”转换为“isolate 记录”。每条原始序列关联其 Nextclade 结果并提取 `EPI_ISL_ID`；同一 `EPI_ISL_ID` 的多条序列仅选择质量最好的一条作为代表，因此一个 isolate 在后续流程中只计一次。对代表记录执行统一 QC，并保留采样日期、clade 和氨基酸序列。

- 不进行均衡降采样.因为它会改变观测数据的时间和地区构成，进而改变训练使用的频率与增长信号。这里保留的是原始观测频率，并不等同于无偏的真实人群流行率。

- 将序列按照 `collection_date` 分bin。
- 确定要进行哪些pair之间的抗原差异计算。形成EPI_ISL_ID->HA序列->HA1序列（避免）->抗原性的映射。
- 建立EPI_ISL_ID->去重序列的映射。计算fitness。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/chenyh/miniconda3/envs/fluProfiler/bin/torchrun --standalone --nproc_per_node=4 \
src/fitness/reference_free_bert_ranking/sequence_bert_reference_free_ranking_multi_gpu.py \
  --fasta Alright-strain/data/raw/fasta/strain/H1N1/sequences.fasta \
  --score-table /home/chenyh/workspace/Alright-strain/data/raw/fasta/strain/H1N1/h1n1_recent_fitness.sqlite
```
