#!/usr/bin/env python3
"""Benchmark one real sequence for reference-free LucaVirus ranking.

Example:
    /home/chenyh/miniconda3/envs/fluProfiler/bin/python \
      src/fitness/benchmark_reference_free_ranking.py --gpu 2
"""

import argparse
import sys
import time
from pathlib import Path

import torch


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
                if header is None:
                    raise ValueError(f"{path}: sequence appears before a FASTA header")
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


def estimate_seconds(lengths, sample_length: int, sample_seconds: float, batch_size: int):
    """Return estimates assuming per-forward cost scales with L and L^2.

    The actual model lies between these two bounds; the linear estimate is
    usually more useful for this 1B model at the dataset's typical lengths.
    """
    sample_batches = (sample_length + batch_size - 1) // batch_size
    linear_work = sum(((length + batch_size - 1) // batch_size) * length for length in lengths)
    quadratic_work = sum(
        ((length + batch_size - 1) // batch_size) * length * length
        for length in lengths
    )
    linear_reference = sample_batches * sample_length
    quadratic_reference = sample_batches * sample_length * sample_length
    return (
        sample_seconds * linear_work / linear_reference,
        sample_seconds * quadratic_work / quadratic_reference,
    )


def format_duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.1f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def main():
    project_root = Path(__file__).resolve().parents[2]
    luca_root = Path("/home/chenyh/workspace/LucaVirusTasks")
    default_fasta = project_root / "data/raw/nextclade/H1N1/origin_seqs/deduplicated.fasta"
    run_id = "lucavirus/v1.0/token_level,span_level,seq_level/lucavirus/20240815023346"

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2, help="physical CUDA GPU index (default: 2)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--input-type",
        choices=("auto", "protein", "dna"),
        default="auto",
        help="input sequence type; auto detects DNA and translates frame 0 (default: auto)",
    )
    parser.add_argument("--fasta", type=Path, default=default_fasta)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=luca_root / "llm/models" / run_id / "checkpoint-step3800000",
    )
    parser.add_argument("--log-path", type=Path, default=luca_root / "llm/logs" / run_id / "logs.txt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} does not exist; found {torch.cuda.device_count()} GPU(s)")
    for path in (args.fasta, args.model_path, args.log_path):
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    sys.path[:0] = [str(luca_root), str(luca_root / "src"), str(luca_root / "src/llm/lucavirus")]
    from llm.lucavirus.get_embedding import load_model

    print(f"GPU: {args.gpu} ({torch.cuda.get_device_name(device)})", flush=True)
    records = read_fasta(args.fasta)
    input_is_dna = args.input_type == "dna" or (args.input_type == "auto" and looks_like_dna(records))
    if input_is_dna:
        records = [(sequence_id, translate_frame_zero(sequence)) for sequence_id, sequence in records]
        print("Input detected as DNA/RNA; translated to protein in reading frame 0.", flush=True)
    else:
        print("Input treated as protein sequences.", flush=True)
    lengths = [valid_length(sequence) for _, sequence in records]
    median_length = sorted(lengths)[len(lengths) // 2]
    test_index = min(range(len(records)), key=lambda i: abs(lengths[i] - median_length))
    sequence_id, sequence = records[test_index]
    print(
        f"Dataset: {len(records):,} sequences, {sum(lengths):,} valid residues; "
        f"median={median_length} aa, max={max(lengths)} aa",
        flush=True,
    )
    print(f"Benchmark sequence: #{test_index + 1}, {lengths[test_index]} aa, id={sequence_id}", flush=True)

    start = time.perf_counter()
    _, _, model, tokenizer = load_model(str(args.log_path), str(args.model_path), embedding_inference=False)
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    print(f"Model load: {format_duration(time.perf_counter() - start)}", flush=True)

    tokens = tokenizer.encode(sequence)
    positions = [
        index + int(tokenizer.prepend_bos)
        for index, residue in enumerate(sequence)
        if residue in AA and tokens[index] != tokenizer.unk_idx
    ]
    ids = torch.tensor([tokenizer.cls_idx, *tokens, tokenizer.eos_idx], dtype=torch.long)
    score_total = torch.zeros((), dtype=torch.float64)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for offset in range(0, len(positions), args.batch_size):
        batch_positions = positions[offset : offset + args.batch_size]
        input_ids = ids.unsqueeze(0).repeat(len(batch_positions), 1)
        rows = torch.arange(len(batch_positions))
        position_tensor = torch.tensor(batch_positions)
        targets = input_ids[rows, position_tensor].clone()
        input_ids[rows, position_tensor] = tokenizer.mask_idx

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
                rows.to(device), position_tensor.to(device), targets.to(device)
            ]
        score_total += log_probs.cpu().to(torch.float64).sum()

    torch.cuda.synchronize(device)
    score_seconds = time.perf_counter() - start
    linear_seconds, quadratic_seconds = estimate_seconds(
        lengths, len(positions), score_seconds, args.batch_size
    )
    print("\n=== Benchmark result ===")
    print(f"Score: {(score_total / len(positions)).item():.6f}")
    print(f"Scoring time (one sequence): {format_duration(score_seconds)}")
    print(f"Speed: {len(positions) / score_seconds:.1f} residues/s")
    print("\n=== Estimated full-dataset scoring time (one GPU) ===")
    print(f"Likely estimate (length-linear): {format_duration(linear_seconds)}")
    print(f"Conservative attention-heavy estimate: {format_duration(quadratic_seconds)}")
    print("These exclude model loading and CSV writing. Long sequences may need a smaller --batch-size.")


if __name__ == "__main__":
    main()
