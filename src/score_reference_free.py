#!/usr/bin/env python3
"""Reference-free LucaVirus scoring, zero-shot (WT-marginal) by default.

Compared with ``benchmark_reference_free_ranking.py`` (masked marginal: one
forward pass per residue, ~L passes per sequence), this script scores each
sequence with a **single unmasked forward pass** and sums the model's log
probability for the true residue at every position -- the standard
zero-shot / WT-marginal score (Meier et al. 2021).  That removes the ~L
multiplication in compute and makes full H3/Victoria scoring practical.

Modes
-----
zero_shot (default):
    one forward per sequence; score_mean = sum(log p(x_i | full context)) / n.
masked:
    the original per-position masking protocol from the benchmark script,
    kept only for small-scale sanity checks against zero-shot rankings
    (use with --limit; full-dataset masked scoring is what we are replacing).

Practical features
------------------
* sequences are sorted by length before batching to minimise padding;
* bf16 autocast by default (--no-bf16 to disable); log-softmax always in fp32;
* --shard/--num-shards for multi-GPU runs (shard i reads records i, i+n, ...);
* --merge recombines shard CSVs into one ranked file;
* --dry-run reports the dataset without touching torch (works without CUDA).

Model/tokenizer interface is unchanged from the benchmark script:
``llm.lucavirus.get_embedding.load_model`` from the LucaVTasks checkout.

Examples
--------
Smoke test (64 sequences, one GPU)::

    python src/score_reference_free.py --fasta src/ha_unaglined_H3.fasta \
        --gpu 0 --limit 64 --output outputs/reference_free_ranking/h3_smoke.csv

Four-GPU full run::

    for i in 0 1 2 3; do
      nohup python src/score_reference_free.py --fasta src/ha_unaglined_H3.fasta \
        --gpu $i --shard $i --num-shards 4 \
        --output outputs/reference_free_ranking/h3_scores_shard$i.csv \
        > logs/h3_shard$i.log 2>&1 &
    done

Merge::

    python src/score_reference_free.py --merge \
      "outputs/reference_free_ranking/h3_scores_shard*.csv" \
      --output outputs/reference_free_ranking/h3_scores_ranked.csv
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")
DNA_BASES = set("ACGTUN")
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

DEFAULT_LUCA_ROOT = Path("/home/chenyh/workspace/LucaVirusTasks")
DEFAULT_RUN_ID = "lucavirus/v1.0/token_level,span_level,seq_level/lucavirus/20240815023346"


def read_fasta(path: Path):
    records = []
    header, lines = None, []

    def flush():
        if header is None:
            return
        sequence = "".join(lines).upper()
        if sequence:
            records.append((header.split()[0] or f"sequence_{len(records) + 1}", sequence))

    with path.open(encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header, lines = line[1:].strip(), []
            else:
                lines.append("".join(line.split()))
    flush()
    if not records:
        raise ValueError(f"{path}: no FASTA records found")
    return records


def valid_length(sequence: str) -> int:
    return sum(residue in AA for residue in sequence)


def looks_like_dna(records) -> bool:
    """Detect nucleotide FASTA without mistaking ordinary proteins for DNA."""

    sample = "".join(sequence for _, sequence in records[:100])
    return bool(sample) and sum(base in DNA_BASES for base in sample) / len(sample) >= 0.97


def translate_frame_zero(sequence: str) -> str:
    """Translate a coding DNA/RNA record in its first reading frame."""

    sequence = sequence.replace("U", "T")
    protein = "".join(
        CODON_TABLE.get(sequence[offset : offset + 3], "X")
        for offset in range(0, len(sequence) - 2, 3)
    )
    return protein.rstrip("*")


def select_shard(records, shard: int, num_shards: int):
    if num_shards <= 1:
        return records
    return [record for index, record in enumerate(records) if index % num_shards == shard]


def load_scorer(args: argparse.Namespace):
    """Import LucaVirus and build the per-batch scoring closure."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on this machine")
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} does not exist; found {torch.cuda.device_count()} GPUs")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    model_path = args.luca_root / "llm/models" / DEFAULT_RUN_ID / "checkpoint-step3800000"
    log_path = args.luca_root / "llm/logs" / DEFAULT_RUN_ID / "logs.txt"
    for path in (model_path, log_path):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path[:0] = [str(args.luca_root), str(args.luca_root / "src"), str(args.luca_root / "src/llm/lucavirus")]
    from llm.lucavirus.get_embedding import load_model

    print(f"GPU: {args.gpu} ({torch.cuda.get_device_name(device)})", flush=True)
    started = time.perf_counter()
    _, _, model, tokenizer = load_model(str(log_path), str(model_path), embedding_inference=False)
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    print(f"Model load: {time.perf_counter() - started:.1f}s", flush=True)
    return device, model, tokenizer


def encode_positions(tokenizer, sequence: str):
    """Return (ids, positions, targets) exactly as the benchmark script builds them."""

    tokens = tokenizer.encode(sequence)
    positions = [
        index + int(tokenizer.prepend_bos)
        for index, residue in enumerate(sequence)
        if residue in AA and tokens[index] != tokenizer.unk_idx
    ]
    ids = [tokenizer.cls_idx, *tokens, tokenizer.eos_idx]
    return ids, positions


def score_zero_shot(device, model, tokenizer, records, args: argparse.Namespace) -> pd.DataFrame:
    """One unmasked forward per sequence; sum true-token log probabilities."""

    import torch

    dtype_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16)
    rows: list[dict] = []

    # Sort by valid length so each batch pads as little as possible.
    prepared = []
    for sequence_id, sequence in records:
        ids, positions = encode_positions(tokenizer, sequence)
        if not positions:
            continue
        prepared.append((sequence_id, len(positions), ids, positions))
    prepared.sort(key=lambda item: item[1])

    started = time.perf_counter()
    for offset in range(0, len(prepared), args.batch_size):
        batch = prepared[offset : offset + args.batch_size]
        max_len = max(len(ids) for _, _, ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.padding_idx, dtype=torch.long)
        positions_pad = torch.full((len(batch), max(item[1] for item in batch)), -1, dtype=torch.long)
        targets_pad = torch.full((len(batch), positions_pad.shape[1]), -1, dtype=torch.long)
        for row, (_, _, ids, positions) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            positions_pad[row, : len(positions)] = torch.tensor(positions, dtype=torch.long)
            targets_pad[row, : len(positions)] = input_ids[row, positions_pad[row, : len(positions)]]

        attention = input_ids.ne(tokenizer.padding_idx).long()
        valid = positions_pad >= 0
        safe_positions = positions_pad.clamp(min=0)
        safe_targets = targets_pad.clamp(min=0)

        with torch.inference_mode(), dtype_ctx:
            output = model(
                input_ids_b=input_ids.to(device),
                token_type_ids_b=attention.to(device),
                output_keys_b={"token_level": {"prot_mask"}},
                need_head_weights=False,
                repr_layers=[],
                return_dict=True,
            )
            logits = output.outputs_b["token_level"]["prot_mask"].float()
            log_probs = logits.log_softmax(-1)

        gathered = log_probs[
            torch.arange(len(batch), device=device).unsqueeze(1),
            safe_positions.to(device),
            safe_targets.to(device),
        ]
        gathered = gathered.cpu()
        gathered = gathered.masked_fill(~valid, 0.0)
        sums = gathered.sum(dim=1)
        for row, (sequence_id, n_positions, _, _) in enumerate(batch):
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "valid_length": n_positions,
                    "score_sum": sums[row].item(),
                    "score_mean": sums[row].item() / n_positions,
                    "mode": "zero_shot",
                }
            )
        if (offset // args.batch_size) % 50 == 0:
            done = min(offset + args.batch_size, len(prepared))
            rate = done / max(time.perf_counter() - started, 1e-9)
            print(f"  {done:,}/{len(prepared):,} sequences ({rate:.1f} seq/s)", flush=True)

    return pd.DataFrame(rows)


def score_masked(device, model, tokenizer, records, args: argparse.Namespace) -> pd.DataFrame:
    """Original benchmark protocol: mask one position at a time (slow)."""

    import torch

    rows: list[dict] = []
    started = time.perf_counter()
    for index, (sequence_id, sequence) in enumerate(records):
        ids, positions = encode_positions(tokenizer, sequence)
        if not positions:
            continue
        id_tensor = torch.tensor(ids, dtype=torch.long)
        score_total = torch.zeros((), dtype=torch.float64)
        for offset in range(0, len(positions), args.batch_size):
            batch_positions = positions[offset : offset + args.batch_size]
            input_ids = id_tensor.unsqueeze(0).repeat(len(batch_positions), 1)
            rows_idx = torch.arange(len(batch_positions))
            position_tensor = torch.tensor(batch_positions)
            targets = input_ids[rows_idx, position_tensor].clone()
            input_ids[rows_idx, position_tensor] = tokenizer.mask_idx

            with torch.inference_mode():
                output = model(
                    input_ids_b=input_ids.to(device),
                    token_type_ids_b=input_ids.ne(tokenizer.padding_idx).long().to(device),
                    output_keys_b={"token_level": {"prot_mask"}},
                    need_head_weights=False,
                    repr_layers=[],
                    return_dict=True,
                )
                logits = output.outputs_b["token_level"]["prot_mask"]
                log_probs = logits.log_softmax(-1)[
                    rows_idx.to(device), position_tensor.to(device), targets.to(device)
                ]
            score_total += log_probs.cpu().to(torch.float64).sum()

        n_positions = len(positions)
        rows.append(
            {
                "sequence_id": sequence_id,
                "valid_length": n_positions,
                "score_sum": score_total.item(),
                "score_mean": score_total.item() / n_positions,
                "mode": "masked",
            }
        )
        if (index + 1) % 10 == 0:
            rate = (index + 1) / max(time.perf_counter() - started, 1e-9)
            print(f"  {index + 1:,}/{len(records):,} sequences ({rate:.2f} seq/s)", flush=True)

    return pd.DataFrame(rows)


def merge_shards(pattern: str, output: Path) -> None:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    frames = [pd.read_csv(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop(columns=["rank"], errors="ignore")  # shard files already carry a rank column
    merged = merged.sort_values("score_mean", ascending=False, ignore_index=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Merged {len(paths)} shard file(s): {len(merged):,} rows -> {output}")
    if merged["sequence_id"].duplicated().any():
        duplicated = merged.loc[merged["sequence_id"].duplicated(), "sequence_id"].head(5).tolist()
        print(f"WARNING: {merged['sequence_id'].duplicated().sum()} duplicated sequence_id values, e.g. {duplicated}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fasta", type=Path, help="input FASTA (protein; DNA is auto-detected and translated)")
    parser.add_argument("--gpu", type=int, default=0, help="physical CUDA GPU index (default 0)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-type", choices=("auto", "protein", "dna"), default="auto")
    parser.add_argument("--mode", choices=("zero_shot", "masked"), default="zero_shot")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N sequences (0 = all)")
    parser.add_argument("--shard", type=int, default=0, help="process records where index %% num_shards == shard")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="disable bf16 autocast")
    parser.add_argument("--output", type=Path, default=Path("outputs/reference_free_ranking/scores.csv"))
    parser.add_argument("--luca-root", type=Path, default=DEFAULT_LUCA_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="report dataset stats only; never imports torch")
    parser.add_argument(
        "--merge",
        metavar="GLOB",
        help="merge shard CSVs matching GLOB into --output (ranked) and exit",
    )
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.merge, args.output)
        return
    if args.fasta is None:
        parser.error("--fasta is required unless --merge is used")

    records = read_fasta(args.fasta)
    input_is_dna = args.input_type == "dna" or (args.input_type == "auto" and looks_like_dna(records))
    if input_is_dna:
        records = [(sequence_id, translate_frame_zero(sequence)) for sequence_id, sequence in records]
        print("Input detected as DNA/RNA; translated to protein in reading frame 0.", flush=True)
    else:
        print("Input treated as protein sequences.", flush=True)

    records = select_shard(records, args.shard, args.num_shards)
    if args.limit:
        records = records[: args.limit]

    lengths = [valid_length(sequence) for _, sequence in records]
    print(
        f"Dataset: {len(records):,} sequences, {sum(lengths):,} valid residues; "
        f"median={sorted(lengths)[len(lengths) // 2] if lengths else 0} aa, max={max(lengths) if lengths else 0} aa",
        flush=True,
    )
    if args.shard or args.num_shards > 1:
        print(f"Shard {args.shard}/{args.num_shards}", flush=True)
    if args.dry_run:
        print("Dry run complete: no model loaded.", flush=True)
        return
    if not records:
        raise ValueError("nothing to score after shard/limit selection")

    device, model, tokenizer = load_scorer(args)
    started = time.perf_counter()
    if args.mode == "zero_shot":
        frame = score_zero_shot(device, model, tokenizer, records, args)
    else:
        frame = score_masked(device, model, tokenizer, records, args)
    elapsed = time.perf_counter() - started

    frame = frame.sort_values("score_mean", ascending=False, ignore_index=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print("\n=== Result ===")
    print(f"Scored   : {len(frame):,} sequences in {elapsed:.1f}s ({len(frame) / max(elapsed, 1e-9):.1f} seq/s)")
    print(f"Score    : mean={frame['score_mean'].mean():.4f}, min={frame['score_mean'].min():.4f}, max={frame['score_mean'].max():.4f}")
    print(f"Output   : {args.output}")


if __name__ == "__main__":
    main()
