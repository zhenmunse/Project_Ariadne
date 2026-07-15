"""Build and deterministically clean the Junyi prerequisite graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exercise-table",
        type=Path,
        default=Path("data/raw/junyi/junyi_Exercise_table.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("junyi_diag"))
    return parser.parse_args()


def cycle_edges(cycle: list[str]) -> list[tuple[str, str]]:
    return list(zip(cycle, cycle[1:] + cycle[:1]))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(args.exercise_table)
    required = {"name", "prerequisites", "creation_date"}
    missing = required - set(ex.columns)
    if missing:
        raise ValueError(f"Missing exercise-table columns: {sorted(missing)}")

    exercise_columns = list(ex.columns)
    ex["name"] = ex["name"].astype(str).str.strip()
    if ex["name"].eq("").any():
        raise ValueError("Exercise names must be non-empty after strip().")
    ex["_file_order"] = range(len(ex))
    ex["_creation_date_sort"] = pd.to_datetime(
        ex["creation_date"], errors="coerce", utc=True
    )
    original_exercise_rows = len(ex)
    ex = (
        ex.sort_values(
            ["name", "_creation_date_sort", "_file_order"],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates("name", keep="first")
        .sort_values("_file_order", kind="mergesort")
        .reset_index(drop=True)
    )
    duplicate_exercises_removed = original_exercise_rows - len(ex)
    if ex["name"].duplicated().any():
        raise AssertionError("Exercise names are not unique after deduplication.")
    names = ex["name"]

    graph = nx.DiGraph()
    for idx, row in ex.iterrows():
        name = names.iloc[idx]
        graph.add_node(name, topic=row.get("topic"), area=row.get("area"))

    dangling_count = 0
    self_loop_count = 0
    prerequisite_refs = 0
    matched_refs = 0
    for idx, row in ex.iterrows():
        child = names.iloc[idx]
        prereqs = row["prerequisites"]
        if pd.isna(prereqs):
            continue
        for raw_parent in str(prereqs).split(","):
            parent = raw_parent.strip()
            if not parent:
                continue
            prerequisite_refs += 1
            if parent not in graph:
                dangling_count += 1
                continue
            matched_refs += 1
            if parent == child:
                self_loop_count += 1
                continue
            graph.add_edge(parent, child)

    match_rate = matched_refs / prerequisite_refs if prerequisite_refs else 1.0
    if match_rate <= 0.95:
        raise AssertionError(
            f"Prerequisite match rate {match_rate:.2%} is not greater than 95%."
        )

    raw_edge_count = graph.number_of_edges()
    removed: list[dict[str, object]] = []
    iteration = 0
    while not nx.is_directed_acyclic_graph(graph):
        cycles = list(nx.simple_cycles(graph))
        print(f"FOUND {len(cycles)} CYCLES - first 10:")
        for cycle in cycles[:10]:
            print("  ", cycle)

        iteration += 1
        selected = sorted({max(cycle_edges(cycle)) for cycle in cycles})
        for parent, child in selected:
            if graph.has_edge(parent, child):
                graph.remove_edge(parent, child)
                removed.append(
                    {"u": parent, "v": child, "removal_iteration": iteration}
                )

    edges = pd.DataFrame(sorted(graph.edges()), columns=["u", "v"])
    edges.to_csv(args.output_dir / "junyi_graph_edges.csv", index=False)
    pd.DataFrame(
        removed, columns=["u", "v", "removal_iteration"]
    ).to_csv(args.output_dir / "removed_edges.csv", index=False)

    metadata = {
        "exercise_table": str(args.exercise_table),
        "exercise_columns": exercise_columns,
        "original_exercises": original_exercise_rows,
        "deduplicated_exercises": int(len(ex)),
        "duplicate_exercises_removed": duplicate_exercises_removed,
        "nodes": graph.number_of_nodes(),
        "edges_raw": raw_edge_count,
        "edges_clean": graph.number_of_edges(),
        "dangling_prerequisites": dangling_count,
        "self_loops": self_loop_count,
        "prerequisite_references": prerequisite_refs,
        "matched_prerequisite_references": matched_refs,
        "prerequisite_match_rate": match_rate,
        "removed_cycle_edges": len(removed),
    }
    (args.output_dir / "graph_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, "
        f"dangling={dangling_count}, self_loops={self_loop_count}, "
        f"removed_cycle_edges={len(removed)}, match_rate={match_rate:.2%}"
    )


if __name__ == "__main__":
    main()
