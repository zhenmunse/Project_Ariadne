"""Estimate the locked first-attempt m*Delta/J* diagnostic proxy for Junyi."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

MIN_BUCKET_SUPPORT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem-log",
        type=Path,
        default=Path("data/raw/junyi/junyi_ProblemLog_original.csv"),
    )
    parser.add_argument("--input-dir", type=Path, default=Path("junyi_diag"))
    parser.add_argument("--output-dir", type=Path, default=Path("junyi_diag"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def normalize_correct(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }
    unknown = sorted(set(normalized.dropna()) - set(mapping))
    if unknown:
        raise ValueError(f"Unrecognized correct values: {unknown[:10]}")
    return normalized.map(mapping).astype(bool)


def normalize_time(values: pd.Series) -> tuple[pd.Series, float, str]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_bad = float(numeric.isna().mean())
    if numeric_bad <= 0.01:
        return numeric, numeric_bad, "numeric"

    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    nat_rate = float(timestamps.isna().mean())
    if nat_rate > 0.01:
        raise ValueError(f"time_done NaT rate {nat_rate:.2%} exceeds 1%.")
    return timestamps.astype("int64"), nat_rate, "datetime_utc_ns"


def load_relevant_logs(
    path: Path, relevant: set[str], chunksize: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    input_rows = 0
    retained_rows = 0
    offset = 0
    for chunk in pd.read_csv(
        path,
        usecols=["user_id", "exercise", "correct", "time_done"],
        chunksize=chunksize,
    ):
        input_rows += len(chunk)
        chunk["exercise"] = chunk["exercise"].astype(str).str.strip()
        chunk = chunk.loc[chunk["exercise"].isin(relevant)].copy()
        chunk["_row_order"] = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        retained_rows += len(chunk)
        frames.append(chunk)
        print(f"read {input_rows:,} rows; retained {retained_rows:,}")

    if not frames:
        raise ValueError("No problem-log rows matched closure exercises.")
    logs = pd.concat(frames, ignore_index=True)
    logs["correct"] = normalize_correct(logs["correct"])
    logs["time_done"], bad_time_rate, time_mode = normalize_time(logs["time_done"])
    if logs["user_id"].isna().any():
        raise ValueError("user_id contains missing values.")
    metadata = {
        "problem_log": str(path),
        "actual_columns": ["user_id", "exercise", "correct", "time_done"],
        "input_rows": input_rows,
        "retained_relevant_rows": retained_rows,
        "bad_time_rate": bad_time_rate,
        "time_mode": time_mode,
    }
    return logs, metadata


def exercise_statistics(
    logs: pd.DataFrame, graph: nx.DiGraph, relevant: set[str]
) -> pd.DataFrame:
    logs.sort_values(
        ["user_id", "time_done", "_row_order"], kind="mergesort", inplace=True
    )
    first = logs.drop_duplicates(["user_id", "exercise"], keep="first").copy()
    first["first_id"] = np.arange(len(first), dtype=np.int64)

    rates = first.groupby("exercise", sort=False)["correct"].agg(
        first_rate="mean", first_attempts="size"
    )
    total_attempts = logs["exercise"].value_counts().rename("total_attempts")
    earliest_correct = (
        logs.loc[logs["correct"]]
        .groupby(["user_id", "exercise"], sort=False)["time_done"]
        .min()
        .rename("mastered_time")
        .reset_index()
        .rename(columns={"exercise": "prerequisite"})
    )

    edge_frame = pd.DataFrame(
        [(parent, child) for parent, child in graph.edges if child in relevant],
        columns=["prerequisite", "exercise"],
    )
    with_prereqs = first.merge(edge_frame, on="exercise", how="inner")
    with_prereqs = with_prereqs.merge(
        earliest_correct, on=["user_id", "prerequisite"], how="left"
    )
    with_prereqs["mastered_before"] = (
        with_prereqs["mastered_time"].notna()
        & (with_prereqs["mastered_time"] < with_prereqs["time_done"])
    )
    mastered_counts = with_prereqs.groupby("first_id")["mastered_before"].sum()
    first["k"] = first["first_id"].map(mastered_counts).fillna(0).astype(int)

    buckets = (
        first.groupby(["exercise", "k"])["correct"]
        .agg(success_rate="mean", support="size")
        .reset_index()
    )
    supported = buckets.loc[buckets["support"] >= MIN_BUCKET_SUPPORT]
    deltas = supported.groupby("exercise")["success_rate"].agg(
        delta_min="min", delta_max="max", supported_buckets="size"
    )
    deltas["delta"] = deltas["delta_max"] - deltas["delta_min"]
    deltas = deltas[["delta", "supported_buckets"]]
    deltas.loc[deltas["supported_buckets"] < 2, "delta"] = np.nan

    stats = pd.DataFrame(index=sorted(relevant))
    stats.index.name = "exercise"
    stats = stats.join(rates).join(total_attempts).join(deltas)
    return stats.reset_index()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(args.input_dir / "junyi_graph_edges.csv")
    graph = nx.DiGraph()
    graph.add_edges_from(edges[["u", "v"]].itertuples(index=False, name=None))

    closures = pd.read_csv(args.input_dir / "junyi_closures.csv")
    targets = list(dict.fromkeys(closures["target"].astype(str)))
    graph.add_nodes_from(targets)
    closure_nodes = {
        target: nx.ancestors(graph, target) | {target} for target in targets
    }
    relevant = set().union(*closure_nodes.values())

    logs, metadata = load_relevant_logs(args.problem_log, relevant, args.chunksize)
    stats = exercise_statistics(logs, graph, relevant).set_index("exercise")
    missing_rates = stats.index[stats["first_rate"].isna()].tolist()
    zero_rates = stats.index[stats["first_rate"].eq(0)].tolist()

    rows: list[dict[str, object]] = []
    for target in targets:
        nodes = sorted(closure_nodes[target])
        all_selected = stats.loc[nodes]
        selected = all_selected.loc[all_selected["first_rate"].gt(0)]
        available_delta = selected["delta"].dropna()
        delta_median = float(available_delta.median()) if len(available_delta) else np.nan
        delta_max = float(available_delta.max()) if len(available_delta) else np.nan
        j_star = float((1.0 / selected["first_rate"]).sum())
        size = len(nodes)
        observed_size = len(selected)
        med_ratio = (
            observed_size * delta_median / j_star if j_star > 0 else np.nan
        )
        max_ratio = observed_size * delta_max / j_star if j_star > 0 else np.nan
        rows.append(
            {
                "target": target,
                "m": observed_size,
                "m_total": size,
                "delta_median": delta_median,
                "delta_max": delta_max,
                "J_star_proxy": j_star,
                "mdelta_med_ratio": med_ratio,
                "mdelta_max_ratio": max_ratio,
                "coverage": (
                    len(available_delta) / observed_size if observed_size else 0.0
                ),
                "obs_fraction": observed_size / size,
            }
        )

    pd.DataFrame(rows).to_csv(args.output_dir / "junyi_mdelta.csv", index=False)
    metadata.update(
        {
            "minimum_bucket_support": MIN_BUCKET_SUPPORT,
            "closure_exercises": len(relevant),
            "first_attempt_rows": int(stats["first_attempts"].sum()),
            "missing_rate_exercise_count": len(missing_rates),
            "zero_rate_exercise_count": len(zero_rates),
            "zero_rate_exercises": [
                {
                    "exercise": exercise,
                    "first_attempts": int(stats.at[exercise, "first_attempts"]),
                    "total_attempts": int(stats.at[exercise, "total_attempts"]),
                }
                for exercise in zero_rates
            ],
        }
    )
    (args.output_dir / "mdelta_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} closure rows; delta coverage defined per closure")


if __name__ == "__main__":
    main()
