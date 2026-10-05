# Alright-strain 源码结构

源码按对象与职责分层，而不是按临时实验名称分散放置：`strain_data` 先把 FASTA、GISAID metadata 和 Nextclade 注释整理为可追溯的 strain 记录；`forecasting` 再在这些记录之上做不同粒度的预测；`fitness` 与 `antigenicity` 提供可复用的生物学特征。

```text
src/
  strain_data/             # FASTA、Nextclade、标签、审计与数据准备
  forecasting/
    subclade/              # 现有：subclade 频率预测、基线和回测
    strain/                # 新建：未来的 strain 级预测
  fitness/                 # LucaVirus reference-free sequence fitness
  antigenicity/            # fluProfiler antigenic-distance workflow
  analysis/                # 可视化与探索性分析工具
```

`forecasting/subclade` 是当前可运行的预测模块：`competition.py` 实现 change-point + softmax 竞争模型，`benchmarks.py` 提供 B0/B1/B2 基线，`forecasts_flu_mlr.py` 运行官方 forecasts-flu MLR，其他脚本负责把 Fitness 与抗原性特征纳入相同的回测框架。

`forecasting/strain` 目前是一个刻意留空的独立边界。strain 级模型应在这里接收 `strain_data` 提供的记录，并按具体毒株而非聚合 subclade 输出排序、增长潜力或未来频率预测；它不应复用或混入现有 subclade 预测脚本。

Fitness 的序列准备工具位于 `strain_data/fasta.py` 和 `strain_data/audit.py`，而 LucaVirus 打分与结果聚合位于 `fitness/`。抗原性模块保留其 HA1 资产准备、冻结血清面板、并行评分和信号回测流程。
