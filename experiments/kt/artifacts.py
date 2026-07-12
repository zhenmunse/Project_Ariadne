"""Deterministic artifact utilities for the KT set-oracle pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with the canonical settings used for artifact hashes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_path(path: str | Path) -> str:
    """Represent repository files with stable POSIX-style relative paths."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_json(path: str | Path, value: Any) -> None:
    """Write deterministic, readable JSON with a final newline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    destination.write_text(rendered, encoding="utf-8", newline="\n")

