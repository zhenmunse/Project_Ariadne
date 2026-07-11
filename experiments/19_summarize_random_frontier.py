"""Summarize common-scorer regret for the Random Frontier Policy."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scored_csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def summarize(scope: str, target_node: str, regrets: list[float]) -> dict:
    return {
        "scope": scope,
        "target_node": target_node,
        "records": len(regrets),
        "mean_regret": statistics.fmean(regrets),
        "median_regret": statistics.median(regrets),
        "std_regret": statistics.pstdev(regrets),
        "p05_regret": float(np.percentile(regrets, 5)),
        "p95_regret": float(np.percentile(regrets, 95)),
        "best_regret": min(regrets),
        "worst_regret": max(regrets),
    }


def main() -> None:
    args = parse_args()
    with args.scored_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or any(row["method"] != "random_frontier" for row in rows):
        raise ValueError("Input must contain Random Frontier scorer rows")
    if any(row["valid"] != "True" for row in rows):
        raise ValueError("Cannot summarize invalid Random Frontier sequences")

    by_target: dict[str, list[float]] = {}
    all_regrets = []
    for row in rows:
        regret = float(row["normalized_regret"])
        by_target.setdefault(row["target_node"], []).append(regret)
        all_regrets.append(regret)
    if len(rows) != 1000 or any(len(values) != 100 for values in by_target.values()):
        raise ValueError("Expected exactly 100 runs for each of ten targets")

    summaries = [summarize("overall", "", all_regrets)]
    summaries.extend(
        summarize("target", target, by_target[target])
        for target in sorted(by_target, key=int)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"summary={args.output}")


if __name__ == "__main__":
    main()
