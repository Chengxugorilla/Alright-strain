"""Shared paths and file-output safeguards for the antigenicity workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLUPROFILER_ROOT = Path("/home/chenyh/workspace/fluProfiler")
IDENTITY_COLUMNS = ("fasta_sha256", "sequence_index", "sequence_sha256")


def require_file(path: Path, label: str) -> Path:
    """Return an existing file as an absolute path."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    """Return an existing directory as an absolute path."""

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def prepare_output_directory(output_dir: Path, outputs: list[Path], *, overwrite: bool) -> Path:
    """Create an output directory after ensuring existing files are intentional."""

    output_dir = output_dir.expanduser().resolve()
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(
    path: Path,
    value: Any,
    *,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Write readable UTF-8 JSON with the workflow's standard trailing newline."""

    path.write_text(
        json.dumps(value, default=default, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
