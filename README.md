# Alright-strain

Alright-strain 面向流感病毒演化预测：从病毒株序列、采样时间和 Nextclade 注释出发，构建可追溯的数据集，并结合频率趋势、序列 Fitness 与抗原性信号预测未来的群体变化。

当前可运行的是 **subclade 级频率预测**；**strain 级预测** 正在以独立模块构建，避免把“预测聚合谱系”和“预测具体毒株”混为同一个问题。

## 整体思路

```text
GISAID FASTA + metadata
          │
          ▼
Nextclade annotation ──► strain_data：日期、标签、序列与审计
          │                           │
          ▼                           ▼
  subclade counts/time series    Fitness + antigenicity features
          │                           │
          └──────────► forecasting ──┘
                         ├─ subclade：未来频率与回测
                         └─ strain：未来的毒株级预测
```

所有正式运行使用 `configs/` 中的参数，并在 `outputs/<run_id>/manifest.json` 记录最终参数和输入文件哈希，保证结果可追溯。

## 模块地图

| 模块 | 负责什么 | 从哪里开始 |
|---|---|---|
| 数据采集与整理 | GISAID 下载、FASTA 合并/去重、Nextclade 注释与审计 | [`data/README.md`](data/README.md) |
| 数据代码 | 将序列、metadata 与 Nextclade 结果整理为可建模记录 | [`src/strain_data/`](src/strain_data/) |
| Subclade 预测 | 频率基线、竞争模型、官方 MLR 与滚动回测 | [`src/forecasting/subclade/README.md`](src/forecasting/subclade/README.md) |
| Strain 预测 | 具体毒株的排序、增长潜力或未来频率预测 | [`src/forecasting/strain/`](src/forecasting/strain/) |
| Fitness | LucaVirus reference-free 序列适应性评分 | [`src/fitness/`](src/fitness/) |
| Antigenicity | fluProfiler 抗原距离、血清面板与回测 | [`src/antigenicity/README.md`](src/antigenicity/README.md) |
| 实验配置 | 预测窗口、标签策略和模型参数 | [`configs/README.md`](configs/README.md) |
| 分析与结果 | 探索性 notebook 与每次运行的可追溯输出 | [`analysis/`](analysis/) · [`outputs/`](outputs/) |

## 最快开始

若你要运行当前 H1N1 subclade MLR benchmark：

```bash
conda run -n fluProfiler python src/forecasting/subclade/forecasts_flu_mlr.py \
  --config configs/subclade/h1n1_mlr.json \
  --output-dir outputs/h1n1_mlr_20261005
```

若你要新增或整理序列数据，请先阅读 [`data/README.md`](data/README.md)，不要直接修改 `data/raw/` 中的原始文件。

## 目录约定

- `data/raw/`：原始 FASTA 与 Nextclade TSV，只读。
- `data/interim/`：可从原始数据再生成的中间数据。
- `data/processed/`：供模型直接消费的规范化数据。
- `outputs/<run_id>/`：一次实验的预测、指标、图与 manifest。

> GISAID 数据的获取和使用必须遵守其注册用户协议与数据使用条款。本项目不再分发 GISAID 序列数据。
