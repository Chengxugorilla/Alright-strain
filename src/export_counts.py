import pandas as pd
from pathlib import Path
from src.h1n1_subclade_plot import read_nextclade_table, build_binned_tables, DEFAULT_START_DATE

# 1. 指定服务器原始数据路径
INPUT_TSV = '/home/chenyh/workspace/Alright-strain/data/clade注释_sorted/H1N1/h1n1_results_MW626062_sorted.tsv'

# 2. 用师兄的逻辑读取并提取字段
print("Loading records...")
records = read_nextclade_table(INPUT_TSV, label_mode='nextclade')

# 3. 聚合为 14-day counts 宽表
print("Building binned tables...")
counts, proportions, bin_totals = build_binned_tables(
    records, 
    bin_days=14, 
    start_date=DEFAULT_START_DATE, 
    end_date=None
)

# 4. 导出为 forecasts-flu 模型需要的 TSV 格式
OUTPUT_TSV = '/home/chenyh/workspace/forecasts-flu/h1n1_counts_input.tsv'
counts.to_csv(OUTPUT_TSV, sep='\t')
print(f"Success! Counts table exported to {OUTPUT_TSV}")