"""Validate and score standard sequence JSONL files with the common evaluator."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DAG_PATH,
    DEFAULT_MANIFEST_PATH,
)
from experiments.common.schema import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Sequence JSONL files")
    parser.add_argument("--output", required=True, type=Path, help="Scored CSV path")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dag", type=Path, default=DEFAULT_DAG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for path in args.inputs:
        records.extend(read_jsonl(path))

    evaluator = SequenceEvaluator.from_artifacts(
        manifest_path=args.manifest,
        dag_path=args.dag,
        checkpoint_path=args.checkpoint,
    )
    rows = [scored.to_dict() for scored in evaluator.score_records(records)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "target_node",
        "run_id",
        "valid",
        "evaluation_cost",
        "optimal_cost",
        "normalized_regret",
        "sequence_hash",
        "invalid_reason",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"scored_records={len(rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
