# Subclade 预测

本模块以按时间聚合的 Nextclade subclade counts 为输入，预测未来各 subclade 的频率。它服务于现有的聚合谱系预测任务；未来的具体毒株预测应进入相邻的 `../strain/`，不要和这里混用。

## 当前能力

- `benchmarks.py`：B0 持续性、B1 局部频率趋势、B2 renewal-selection 的可复现基线和滚动回测。
- `competition.py`：change-point + CLR/softmax 的多 clade 竞争预测器。
- `forecasts_flu_mlr.py`：以 evofr 运行官方 Nextstrain forecasts-flu MLR，并与本地基线使用同一回测协议比较。
- `*_biology_forecast.py`：评估 Fitness 与抗原性特征是否改善 clade 频率预测。
- `export_counts.py`：导出 forecasts-flu 所需的分箱 counts 表。

## 运行 MLR benchmark

```bash
conda run -n fluProfiler python src/forecasting/subclade/forecasts_flu_mlr.py \
  --config configs/subclade/h1n1_mlr.json \
  --output-dir outputs/h1n1_mlr_20261005
```

配置定义输入、标签策略、预测窗口与推断参数；运行目录中的 `manifest.json` 保存输入哈希和最终参数。详见 [`configs/README.md`](../../../configs/README.md)。

## Benchmark 边界

本地 B0/B1/B2 用于快速迭代和同协议比较，其中 B1/B2 是轻量 proxy，不是官方实现。`forecasts_flu_mlr.py` 才是 Nextstrain MLR 的正式比较入口。Previr、CovTransformer、位点突变动力学与 RelRe 可作为进一步的外部对照，但尚未被封装为本项目的统一运行入口。
