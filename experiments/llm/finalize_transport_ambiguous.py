"""Finalize one dispatched-without-raw request as terminal transport-ambiguous."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.llm.artifacts import atomic_write_json, load_json


def finalize(
    *,
    output_root: Path,
    logical_run_key: str,
    attempt: int,
    error_class: str,
    error_message: str,
) -> Path:
    request_path = output_root / "requests" / logical_run_key / f"{attempt:03d}.json"
    raw_path = output_root / "raw" / logical_run_key / f"{attempt:03d}.json"
    parsed_path = output_root / "parsed" / logical_run_key / f"{attempt:03d}.json"
    artifact = load_json(request_path)
    if artifact.get("status") != "request_dispatched":
        raise ValueError("Only request_dispatched artifacts may be finalized")
    if raw_path.exists() or parsed_path.exists():
        raise ValueError("Cannot finalize transport-ambiguous when raw/parsed exists")
    artifact.update({
        "status": "transport_ambiguous_terminal",
        "terminal_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport_error": {
            "classification": "transport_ambiguous",
            "error_class": error_class,
            "message": error_message,
            "retryable": False,
            "provider_response_received": False,
            "excluded_from_model_validity_denominator": True,
            "included_in_pipeline_yield_denominator": True,
        },
    })
    atomic_write_json(request_path, artifact)
    return request_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-run-key", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--error-class", required=True)
    parser.add_argument("--error-message", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "llm")
    args = parser.parse_args()
    print(finalize(
        output_root=args.output_root,
        logical_run_key=args.logical_run_key,
        attempt=args.attempt,
        error_class=args.error_class,
        error_message=args.error_message,
    ))


if __name__ == "__main__":
    main()
