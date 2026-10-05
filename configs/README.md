# 实验配置

配置文件只记录会改变数据选择、预测窗口或模型行为的参数；运行输出位置由命令行 `--output-dir` 指定，以便每次实验使用独立的 `outputs/<run_id>/` 目录。

例如：

```bash
conda run -n fluProfiler python src/forecasting/subclade/forecasts_flu_mlr.py \
  --config configs/subclade/h1n1_mlr.json \
  --output-dir outputs/h1n1_mlr_20261005
```

运行结束后，输出目录中的 `manifest.json` 会记录解析后的参数、配置文件位置、输入文件大小和 SHA-256；它是复现实验的依据。
