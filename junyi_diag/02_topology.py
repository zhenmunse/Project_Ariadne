"""Compute Junyi poset width, target closures, and reachable ideal counts."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

GUARD = 10_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exercise-table",
        type=Path,
        default=Path("data/raw/junyi/junyi_Exercise_table.csv"),
    )
    parser.add_argument(
        "--problem-log",
        type=Path,
        default=Path("data/raw/junyi/junyi_ProblemLog_original.csv"),
    )
    parser.add_argument("--input-dir", type=Path, default=Path("junyi_diag"))
    parser.add_argument("--output-dir", type=Path, default=Path("junyi_diag"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def poset_width(graph: nx.DiGraph) -> int:
    """Return the width of the poset induced by a DAG."""
    closure = nx.transitive_closure_dag(graph)
    left = [("L", node) for node in closure.nodes]
    right = [("R", node) for node in closure.nodes]
    bipartite = nx.Graph()
    bipartite.add_nodes_from(left, bipartite=0)
    bipartite.add_nodes_from(right, bipartite=1)
    bipartite.add_edges_from(
        (("L", parent), ("R", child)) for parent, child in closure.edges
    )
    matching = nx.bipartite.maximum_matching(bipartite, top_nodes=left)
    return closure.number_of_nodes() - len(matching) // 2


def count_ideals(graph: nx.DiGraph, guard: int = GUARD) -> int | str:
    """Count ideals exactly up to a hard guard using maximal-element recursion."""
    nodes = tuple(nx.topological_sort(graph))
    index = {node: idx for idx, node in enumerate(nodes)}
    downsets = []
    descendants = []
    for node in nodes:
        downset = nx.ancestors(graph, node) | {node}
        downsets.append(sum(1 << index[item] for item in downset))
        descendants.append(sum(1 << index[item] for item in nx.descendants(graph, node)))

    cap = guard + 1

    @lru_cache(maxsize=None)
    def count(remaining: int) -> int:
        if not remaining:
            return 1

        maximal = [
            idx
            for idx in range(len(nodes))
            if remaining & (1 << idx) and not (descendants[idx] & remaining)
        ]
        chosen = max(
            maximal,
            key=lambda idx: (downsets[idx] & remaining).bit_count(),
        )
        without_chosen = remaining & ~(1 << chosen)
        excluded = count(without_chosen)
        if excluded >= cap:
            return cap
        included = count(remaining & ~downsets[chosen])
        return min(cap, excluded + included)

    total = count((1 << len(nodes)) - 1)
    return f">{guard:.0e}" if total > guard else total


def load_graph(exercise_table: Path, edge_file: Path) -> nx.DiGraph:
    exercises = pd.read_csv(exercise_table, usecols=["name"])
    names = exercises["name"].astype(str).str.strip().tolist()
    edges = pd.read_csv(edge_file)
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    graph.add_edges_from(edges[["u", "v"]].itertuples(index=False, name=None))
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Cleaned graph is not a DAG.")
    return graph


def top_volume_targets(
    problem_log: Path, graph: nx.DiGraph, chunksize: int
) -> tuple[list[str], dict[str, int]]:
    counts = pd.Series(dtype="int64")
    for chunk in pd.read_csv(problem_log, usecols=["exercise"], chunksize=chunksize):
        current = chunk["exercise"].astype(str).str.strip().value_counts()
        counts = counts.add(current, fill_value=0)
    counts = counts.astype("int64")
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    targets = [name for name, _ in ordered[:20]]
    missing = sorted(set(targets) - set(graph.nodes))
    if missing:
        raise ValueError(f"Top-20 volume targets absent from exercise graph: {missing}")
    if len(targets) != 20:
        raise ValueError(f"Expected 20 volume targets, found {len(targets)}.")
    return targets, {str(name): int(value) for name, value in counts.items()}


def save_figure(results: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metrics = [
        ("closure_size", "Closure size"),
        ("width", "Width"),
        ("ideal_count_numeric", "Ideal count (guard at 1e7)"),
    ]
    colors = {"T-sink": "#4c78a8", "T-vol": "#f58518"}
    for axis, (column, label) in zip(axes, metrics):
        for group in ("T-sink", "T-vol"):
            values = results.loc[results["group"] == group, column].to_numpy()
            axis.hist(values, bins="auto", alpha=0.6, label=group, color=colors[group])
        axis.set_xlabel(label)
        axis.set_ylabel("Closures")
    axes[-1].set_xscale("log")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph = load_graph(args.exercise_table, args.input_dir / "junyi_graph_edges.csv")
    reduced = nx.transitive_reduction(graph)
    full_width = poset_width(reduced)
    sinks = sorted(node for node, degree in graph.out_degree() if degree == 0)
    volume_targets, attempt_counts = top_volume_targets(
        args.problem_log, graph, args.chunksize
    )

    metrics: dict[str, tuple[int, int, int, int | str]] = {}
    rows: list[dict[str, object]] = []
    for group, targets in (("T-sink", sinks), ("T-vol", volume_targets)):
        for index, target in enumerate(targets, start=1):
            if target not in metrics:
                closure_nodes = nx.ancestors(graph, target) | {target}
                closure = reduced.subgraph(closure_nodes).copy()
                metrics[target] = (
                    closure.number_of_nodes(),
                    closure.number_of_edges(),
                    poset_width(closure),
                    count_ideals(closure),
                )
            size, edge_count, width, ideal_count = metrics[target]
            rows.append(
                {
                    "target": target,
                    "group": group,
                    "closure_size": size,
                    "edges_reduced": edge_count,
                    "width": width,
                    "ideal_count": ideal_count,
                }
            )
            print(
                f"{group} {index}/{len(targets)}: {target} "
                f"m={size}, width={width}, ideals={ideal_count}"
            )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "junyi_closures.csv", index=False)
    results["ideal_count_numeric"] = pd.to_numeric(
        results["ideal_count"], errors="coerce"
    ).fillna(GUARD)
    save_figure(results, args.output_dir / "junyi_topology_diagnostic.png")

    metadata = {
        "nodes": graph.number_of_nodes(),
        "edges_clean": graph.number_of_edges(),
        "edges_reduced": reduced.number_of_edges(),
        "poset_width": full_width,
        "sink_count": len(sinks),
        "volume_targets": volume_targets,
        "volume_target_attempts": {
            target: attempt_counts[target] for target in volume_targets
        },
        "ideal_guard": GUARD,
    }
    (args.output_dir / "topology_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"full_width={full_width}, sinks={len(sinks)}, "
        f"edges_raw={graph.number_of_edges()}, edges_reduced={reduced.number_of_edges()}"
    )


if __name__ == "__main__":
    main()
