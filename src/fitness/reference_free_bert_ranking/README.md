# Reference-free BERT sequence ranking

## Scope and locations

This task scores protein sequences independently with the LucaVirus masked-protein model. It is designed for interruption and restart: the SQLite score table, rather than memory or a final CSV, is the source of truth.

| Item | Location |
| --- | --- |
| Input FASTA | `/home/chenyh/workspace/Alright-strain/src/ha_unaglined.fasta` |
| Task runner | `src/fitness/reference_free_bert_ranking/sequence_bert_reference_free_ranking_multi_gpu.py` |
| LucaVirus checkpoint | `/home/chenyh/workspace/LucaVirusTasks/llm/models/lucavirus/v1.0/token_level,span_level,seq_level/lucavirus/20240815023346/checkpoint-step3800000` |
| LucaVirus log/config input | `/home/chenyh/workspace/LucaVirusTasks/llm/logs/lucavirus/v1.0/token_level,span_level,seq_level/lucavirus/20240815023346/logs.txt` |
| Durable raw-score table | `/home/chenyh/workspace/Alright-strain/outputs/bert_reference_free_ranking_raw_scores.sqlite` |

The input FASTA remains under `src/`. Generated data remains under `outputs/`; do not put generated score files beside the FASTA.

## Computation

For every standard amino-acid position `i`, the runner masks that residue, runs LucaVirus, and obtains the conditional log probability of the original residue:

`log P(x_i | x_without_i)`

It records two values per sequence:

- `raw_log_prob_sum = sum_i log P(x_i | x_without_i)`: the unnormalised, auditable cumulative score.
- `bert_score = raw_log_prob_sum / valid_length`: the length-normalised mean score for later comparisons.

The model runs in FP16 for throughput. The final `log_softmax` is computed in FP32 before accumulating scores, to keep the recorded likelihood values numerically stable.

## Raw score table standard

The SQLite table `sequence_scores` contains both raw input and result columns:

| Column | Meaning |
| --- | --- |
| `fasta_sha256` | SHA-256 of the full input FASTA file. |
| `sequence_index` | Zero-based position in that FASTA file. |
| `sequence_id` | ID parsed from the FASTA header. |
| `sequence_sha256` | SHA-256 of the amino-acid sequence. |
| `sequence` | Original amino-acid sequence. |
| `valid_length` | Number of scored standard amino-acid positions. |
| `raw_log_prob_sum` | Raw cumulative log likelihood; `NULL` means not yet scored. |
| `bert_score` | `raw_log_prob_sum / valid_length`. |
| `scored_at_utc` | UTC time at which this row was written. |

The identity of a score is the combination of `fasta_sha256`, `sequence_index`, `sequence_id`, and `sequence_sha256`. This remains unambiguous even when sequence IDs repeat.

## Multi-GPU execution and restart policy

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 \
/home/chenyh/miniconda3/envs/fluProfiler/bin/torchrun --standalone \
  --nproc_per_node=6 \
  src/fitness/reference_free_bert_ranking/sequence_bert_reference_free_ranking_multi_gpu.py
```

At startup, every worker opens the score table and reads the still-unscored rows in `sequence_index` order (`raw_log_prob_sum IS NULL`). Those pending rows are distributed evenly by rank. There is no DDP/DataParallel wrapper or per-forward GPU communication.

After each full sequence is scored, its corresponding table row is atomically updated. SQLite WAL mode and `synchronous=FULL` make completed rows durable. An interruption can lose only the sequence currently being calculated; rerunning the same command skips all rows whose score is already present.

Use a small smoke test without creating a second code path:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/chenyh/miniconda3/envs/fluProfiler/bin/torchrun --standalone \
  --nproc_per_node=1 \
  src/fitness/reference_free_bert_ranking/sequence_bert_reference_free_ranking_multi_gpu.py \
  --score-table outputs/bert_reference_free_ranking_smoke_test.sqlite \
  --max-pending 2
```

## Output policy

This runner intentionally does **not** produce a ranked CSV. The SQLite table is the authoritative raw result. Ranking, filtering, and CSV export are separate downstream steps so they cannot alter or obscure the raw sequence-to-score mapping.

After every row has a score, create the final ranked CSV and a compact JSON summary with:

```bash
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/fitness/reference_free_bert_ranking/aggregate_scores.py
```

The script refuses to aggregate an incomplete table or a table containing more than one FASTA fingerprint. It writes:

- `outputs/bert_reference_free_ranking.csv`: complete sequence-level ranking, ordered by `bert_score` descending.
- `outputs/bert_reference_free_ranking_summary.json`: counts, score/length ranges and means, source fingerprint, and scoring time range.

## Time and clade analysis

`sequence_id` is the first whitespace-delimited part of a FASTA header and is therefore not sufficient for metadata parsing when isolate names contain spaces. Use the original FASTA plus `sequence_index` and `sequence_sha256` to recover the complete header exactly:

```bash
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/fitness/reference_free_bert_ranking/analyze_time_clade.py
```

This creates a local sequence-level metadata table and compact score summaries by year, clade, and year-clade. For pipe-delimited headers, the parser treats field 6 as the collection date and field 5 as the clade candidate; records without an ISO collection date fall back to the year present in the isolate name. Preserve the raw header and inspect its field conventions before interpreting a clade trend biologically.

## Next-window clade signal backtest

To test whether a clade's mean BERT score in one six-month window predicts its **next** six-month proportion or the next-window top-1 clade, run:

```bash
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/fitness/reference_free_bert_ranking/backtest_score_signal.py
```

The test follows the notebook's retained nine rolling windows and excludes the three COVID low-sampling target windows. For each target window, it uses only the immediately preceding six months to calculate clade-level mean, median, and maximum BERT score. It reports Pearson/Spearman correlations with next-window share and share change, plus top-1 accuracy for persistence, highest-score, and lowest-score rules. Matching to Nextclade uses unique `EPI_ISL_*` identifiers; the output records the match coverage. These are descriptive forecasting tests, not evidence that a score causes a clade to rise or disappear.

To inspect one chosen cutoff, use the end of the feature window as `--cutoff`; the next six calendar months are the held-out outcome window. For example:

```bash
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/fitness/reference_free_bert_ranking/backtest_score_signal.py \
  --cutoff 2024-09-30 \
  --output-dir outputs/h1n1_score_signal_backtest/cutoff_2024_09_30
```

## Score-by-date chart

Create a 14-day collection-date chart of mean and median BERT score, its 10–90% interval, and the sample count per bin:

```bash
/home/chenyh/miniconda3/envs/fluProfiler/bin/python \
  src/fitness/reference_free_bert_ranking/plot_score_over_time.py
```

It writes `outputs/bert_reference_free_ranking_score_by_date.png` and the compact underlying date-bin table `outputs/bert_reference_free_ranking_score_by_date.csv`.
