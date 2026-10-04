"""Resumable multi-GPU LucaVirus scorer backed by one durable SQLite table.

Launch from the project root, for example:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 \
  /home/chenyh/miniconda3/envs/fluProfiler/bin/torchrun --standalone \
  --nproc_per_node=6 src/sequence_bert_reference_free_ranking_multi_gpu.py

The SQLite table stores every original FASTA sequence and its score columns.
Workers only communicate through atomic database updates; there is no DDP or
per-forward synchronization.
"""

import argparse
import hashlib
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LUCA_ROOT = Path("/home/chenyh/workspace/LucaVirusTasks")
AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: Path):
    records, empty_headers = [], []
    header, sequence_lines = None, []

    def flush_record():
        if header is None:
            return
        sequence = "".join(sequence_lines).upper()
        if not sequence:
            empty_headers.append(header)
            return
        sequence_id = header.split()[0] or f"sequence_{len(records) + 1}"
        records.append((sequence_id, sequence))

    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                header, sequence_lines = line[1:].strip(), []
            elif header is None:
                raise ValueError(f"{path}: line {line_number} appears before the first FASTA header (>).")
            else:
                sequence_lines.append("".join(line.split()))
    flush_record()

    if not records:
        raise ValueError(f"{path}: no non-empty FASTA records were found.")
    if empty_headers:
        raise ValueError(f"{path}: {len(empty_headers)} empty FASTA record(s), e.g. {', '.join(empty_headers[:3])}")
    return records


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def open_score_table(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60, isolation_level=None)
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sequence_scores (
            fasta_sha256 TEXT NOT NULL,
            sequence_index INTEGER NOT NULL,
            sequence_id TEXT NOT NULL,
            sequence_sha256 TEXT NOT NULL,
            sequence TEXT NOT NULL,
            valid_length INTEGER,
            raw_log_prob_sum REAL,
            bert_score REAL,
            scored_at_utc TEXT,
            PRIMARY KEY (fasta_sha256, sequence_index)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS pending_scores ON sequence_scores (fasta_sha256, raw_log_prob_sum, sequence_index)"
    )
    return connection


def seed_sequences(connection, fasta_sha256: str, records):
    """Insert raw FASTA rows once. BEGIN IMMEDIATE makes concurrent starts safe."""
    rows = [
        (fasta_sha256, index, sequence_id, sequence_sha256(sequence), sequence)
        for index, (sequence_id, sequence) in enumerate(records)
    ]
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.executemany(
            """
            INSERT OR IGNORE INTO sequence_scores
                (fasta_sha256, sequence_index, sequence_id, sequence_sha256, sequence)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


def pending_indices(connection, fasta_sha256: str):
    return [
        row[0]
        for row in connection.execute(
            """
            SELECT sequence_index
            FROM sequence_scores
            WHERE fasta_sha256 = ? AND raw_log_prob_sum IS NULL
            ORDER BY sequence_index
            """,
            (fasta_sha256,),
        )
    ]


def save_score(
    connection,
    fasta_sha256: str,
    index: int,
    sequence_id: str,
    sequence: str,
    valid_length: int,
    raw_score: float,
):
    """Atomically fill score columns for exactly the expected original row."""
    cursor = connection.execute(
        """
        UPDATE sequence_scores
        SET valid_length = ?, raw_log_prob_sum = ?, bert_score = ?, scored_at_utc = ?
        WHERE fasta_sha256 = ? AND sequence_index = ? AND sequence_id = ?
          AND sequence_sha256 = ? AND sequence = ? AND raw_log_prob_sum IS NULL
        """,
        (
            valid_length,
            raw_score,
            raw_score / valid_length,
            datetime.now(timezone.utc).isoformat(),
            fasta_sha256,
            index,
            sequence_id,
            sequence_sha256(sequence),
            sequence,
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError(f"序列 {index} 未能写入得分；identity 校验未通过或该行已被写入。")


def encode(sequence, tokenizer):
    tokens = tokenizer.encode(sequence)
    valid = [
        index + int(tokenizer.prepend_bos)
        for index, residue in enumerate(sequence)
        if residue in AA and tokens[index] != tokenizer.unk_idx
    ]
    return torch.tensor([tokenizer.cls_idx, *tokens, tokenizer.eos_idx], dtype=torch.long), valid


def score_sequence(sequence, model, tokenizer, device, mask_batch_size, progress):
    ids, positions = encode(sequence, tokenizer)
    if not positions:
        raise ValueError("至少一条序列没有可打分的标准氨基酸位点。")

    raw_score = 0.0
    for start in range(0, len(positions), mask_batch_size):
        batch_positions = positions[start : start + mask_batch_size]
        input_ids = ids.unsqueeze(0).repeat(len(batch_positions), 1)
        rows = torch.arange(len(batch_positions))
        positions_tensor = torch.tensor(batch_positions)
        targets = input_ids[rows, positions_tensor].clone()
        input_ids[rows, positions_tensor] = tokenizer.mask_idx
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
            log_probs = logits.float().log_softmax(-1)[
                rows.to(device), positions_tensor.to(device), targets.to(device)
            ]
        raw_score += log_probs.sum().item()
        progress.update(len(batch_positions))
    return raw_score, len(positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, default=PROJECT_ROOT / "src/ha_unaglined.fasta")
    parser.add_argument(
        "--score-table",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_raw_scores.sqlite",
        help="SQLite table containing original sequences and nullable score columns.",
    )
    parser.add_argument("--mask-batch-size", type=int, default=128)
    parser.add_argument(
        "--max-pending",
        type=int,
        help="Only process this many incomplete rows; useful for a smoke test.",
    )
    args = parser.parse_args()
    if args.max_pending is not None and args.max_pending < 0:
        raise ValueError("--max-pending 必须为非负整数。")

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model_path = LUCA_ROOT / "llm/models/lucavirus/v1.0/token_level,span_level,seq_level/lucavirus/20240815023346/checkpoint-step3800000"
    log_path = LUCA_ROOT / "llm/logs/lucavirus/v1.0/token_level,span_level,seq_level/lucavirus/20240815023346/logs.txt"
    if not args.fasta.is_file() or not model_path.is_dir() or not log_path.is_file():
        raise FileNotFoundError("请检查 FASTA、LucaVirus checkpoint 和 logs.txt 路径。")

    records = read_fasta(args.fasta)
    fasta_sha256 = sha256_file(args.fasta)
    connection = open_score_table(args.score_table)
    seed_sequences(connection, fasta_sha256, records)
    remaining_indices = pending_indices(connection, fasta_sha256)
    if args.max_pending is not None:
        remaining_indices = remaining_indices[: args.max_pending]
    local_indices = remaining_indices[rank::world_size]
    if rank == 0:
        done = len(records) - len(pending_indices(connection, fasta_sha256))
        print(f"已完成 {done:,}/{len(records):,}；本次待计算 {len(remaining_indices):,} 条。")

    sys.path[:0] = [str(LUCA_ROOT), str(LUCA_ROOT / "src"), str(LUCA_ROOT / "src/llm/lucavirus")]
    from llm.lucavirus.get_embedding import load_model

    progress = tqdm(
        total=sum(sum(residue in AA for residue in records[index][1]) for index in local_indices),
        desc=f"GPU {rank} BERT 打分",
        unit="aa",
        disable=rank != 0,
    )
    if local_indices:
        _, _, model, tokenizer = load_model(str(log_path), str(model_path), embedding_inference=False)
        model = model.to(device).half().eval()
        for index in local_indices:
            sequence_id, sequence = records[index]
            raw_score, valid_length = score_sequence(
                sequence, model, tokenizer, device, args.mask_batch_size, progress
            )
            save_score(
                connection,
                fasta_sha256,
                index,
                sequence_id,
                sequence,
                valid_length,
                raw_score,
            )
    progress.close()
    connection.close()


if __name__ == "__main__":
    main()
