"""Configuration loading and provenance manifests for forecasting runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json_config(path: Path, *, allowed_keys: set[str]) -> dict[str, Any]:
    """Load a flat JSON config and resolve input paths from the project root."""

    config_path = path.expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")
    unknown = set(data).difference(allowed_keys)
    if unknown:
        raise ValueError(f"Unknown config field(s): {', '.join(sorted(unknown))}")
    for key in ("input_tsv",):
        if key in data:
            value = Path(data[key])
            data[key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    output_dir: Path,
    *,
    mode: str,
    arguments: Mapping[str, Any],
    input_files: Mapping[str, Path],
    config_path: Path | None,
) -> Path:
    """Record resolved inputs, settings, and file hashes beside run outputs."""

    inputs: dict[str, dict[str, str | int]] = {}
    for name, value in input_files.items():
        path = value.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest input does not exist: {path}")
        inputs[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    resolved_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in arguments.items()
        if not key.startswith("_")
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "config_path": str(config_path.expanduser().resolve()) if config_path else None,
        "arguments": resolved_args,
        "inputs": inputs,
    }
    destination = output_dir / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
