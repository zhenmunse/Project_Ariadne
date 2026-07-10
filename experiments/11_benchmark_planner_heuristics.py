"""
Benchmark LAO* heuristics on prerequisite closures for ten target concepts.

Each target t is converted to V_t = Ancestors(t) union {t}.  Planning then
runs on the DAG induced by V_t with the entire closure as the terminal set.

Usage:
    python experiments/11_benchmark_planner_heuristics.py
    python experiments/11_benchmark_planner_heuristics.py --oracle monotonic
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import FrozenSet

import networkx as nx
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.planner_engine.solver import DAGPlanner, DAGPlannerDP


@dataclass
class OracleBundle:
    oracle: object
    edge_index: object
    num_nodes: int
    torch_module: object | None = None


class StaticBenchmarkOracle:
    """Deterministic static oracle for isolating heuristic search effort."""

    def __init__(self, base_cost: float):
        self.default_base_cost = base_cost

    def success_prob(self, v: int, mastered: FrozenSet[int]) -> float:
        return 0.55 + 0.05 * (v % 7)

    def base_cost(self, v: int) -> float:
        return self.default_base_cost * (1.0 + 0.02 * (v % 5))

    def best_case_success_prob(self, v: int) -> float:
        return self.success_prob(v, frozenset())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle",
        choices=("static", "monotonic"),
        help="Override experiments.planner_benchmark_oracle from config.",
    )
    return parser.parse_args()


def load_config() -> dict:
    path = os.path.join(ROOT, "configs", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ecs32a_dag(config: dict) -> nx.DiGraph:
    concept_path = os.path.join(ROOT, config["data"]["ecs32a_concepts"])
    edge_path = os.path.join(ROOT, config["data"]["ecs32a_dag_edges"])

    graph = nx.DiGraph()
    with open(concept_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            graph.add_node(int(row["node_id"]))

    with open(edge_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            graph.add_edge(int(row["src"]), int(row["dst"]))

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The ECS32A prerequisite graph must be a DAG")
    return graph


def compute_node_depths(graph: nx.DiGraph) -> dict[int, int]:
    depths: dict[int, int] = {}
    for node in nx.topological_sort(graph):
        predecessors = list(graph.predecessors(node))
        depths[node] = (
            max(depths[p] for p in predecessors) + 1
            if predecessors
            else 0
        )
    return depths


def sample_targets(graph: nx.DiGraph, config: dict) -> list[int]:
    experiment = config["experiments"]
    depths = compute_node_depths(graph)
    candidates = sorted(
        node
        for node, depth in depths.items()
        if depth >= int(experiment["min_depth"])
    )
    count = min(int(experiment["num_targets"]), len(candidates))
    rng = np.random.RandomState(int(config["seed"]))
    return sorted(int(v) for v in rng.choice(candidates, count, replace=False))


def set_planning_seed(seed: int, torch_module: object | None = None) -> None:
    """Reset RNGs once at the beginning of a planning run."""
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is None:
        return

    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def load_oracle(
    oracle_name: str,
    graph: nx.DiGraph,
    config: dict,
) -> OracleBundle:
    if oracle_name == "static":
        oracle = StaticBenchmarkOracle(float(config["planner"]["base_cost"]))
        return OracleBundle(oracle, None, max(graph.nodes()) + 1)

    try:
        import torch
        from src.oracle_core.model import MonotonicOracle
    except ImportError as exc:
        raise RuntimeError(
            "MonotonicOracle benchmark requires torch and torch-geometric. "
            "Install requirements.txt before using --oracle monotonic."
        ) from exc

    checkpoint_path = os.path.join(ROOT, config["oracle"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    node_id_to_idx = checkpoint["node_id_to_idx"]
    if any(node_id_to_idx[node] != node for node in graph.nodes()):
        raise ValueError("Planner currently requires node IDs to match model indices")

    model = MonotonicOracle(
        num_nodes=checkpoint["num_nodes"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    indexed_edges = [
        (node_id_to_idx[src], node_id_to_idx[dst])
        for src, dst in graph.edges()
    ]
    edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
    return OracleBundle(model, edge_index, checkpoint["num_nodes"], torch)


def run_lao(
    target: int,
    closure_graph: nx.DiGraph,
    heuristic: str,
    oracle_name: str,
    bundle: OracleBundle,
    config: dict,
    planning_seed: int,
) -> dict:
    run_config = copy.deepcopy(config)
    run_config["planner"]["heuristic"] = heuristic
    goal = set(closure_graph.nodes())

    set_planning_seed(planning_seed, bundle.torch_module)
    planner = DAGPlanner(
        oracle=bundle.oracle,
        nx_graph=closure_graph,
        config=run_config,
        edge_index=bundle.edge_index,
        num_nodes=bundle.num_nodes,
    )

    started = time.perf_counter()
    result = planner.solve_result(set(), goal)
    elapsed = time.perf_counter() - started
    start = frozenset()
    path = DAGPlanner._extract_path(start, frozenset(goal), result.policy)

    if not result.converged or len(path) != len(goal):
        raise RuntimeError(f"LAO* did not solve prerequisite closure for target {target}")

    return {
        "oracle": oracle_name,
        "target_node": target,
        "closure_size": len(goal),
        "closure_edges": closure_graph.number_of_edges(),
        "solver": "lao_star",
        "heuristic": heuristic,
        "planning_seed": planning_seed,
        "expected_cost": result.values[start],
        "path_length": len(path),
        "expanded_states": result.expanded_count,
        "iterations": result.iterations,
        "planning_time_seconds": elapsed,
    }


def run_dp(
    target: int,
    closure_graph: nx.DiGraph,
    oracle_name: str,
    bundle: OracleBundle,
    config: dict,
    planning_seed: int,
) -> dict:
    goal = set(closure_graph.nodes())
    set_planning_seed(planning_seed, bundle.torch_module)
    planner = DAGPlannerDP(
        oracle=bundle.oracle,
        nx_graph=closure_graph,
        config=config,
        edge_index=bundle.edge_index,
        num_nodes=bundle.num_nodes,
    )

    started = time.perf_counter()
    cost, path = planner.solve(set(), goal)
    elapsed = time.perf_counter() - started

    return {
        "oracle": oracle_name,
        "target_node": target,
        "closure_size": len(goal),
        "closure_edges": closure_graph.number_of_edges(),
        "solver": "exact_dp",
        "heuristic": "none",
        "planning_seed": planning_seed,
        "expected_cost": cost,
        "path_length": len(path),
        "expanded_states": len(planner.memo),
        "iterations": "",
        "planning_time_seconds": elapsed,
    }


def write_results(rows: list[dict], config: dict, oracle_name: str) -> str:
    experiment = config["experiments"]
    results_dir = os.path.join(ROOT, experiment["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(
        results_dir,
        experiment["planner_benchmark_results"].format(oracle=oracle_name),
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_summary(rows: list[dict]) -> None:
    print("\nMean LAO* results across prerequisite closures")
    print("heuristic  expanded_states  planning_time_seconds")
    for heuristic in ("sum", "max", "zero"):
        selected = [
            row for row in rows
            if row["solver"] == "lao_star" and row["heuristic"] == heuristic
        ]
        mean_expanded = np.mean([row["expanded_states"] for row in selected])
        mean_time = np.mean([row["planning_time_seconds"] for row in selected])
        print(f"{heuristic:9s} {mean_expanded:15.1f} {mean_time:22.6f}")


def main() -> None:
    args = parse_args()
    config = load_config()
    experiment = config["experiments"]
    oracle_name = args.oracle or experiment["planner_benchmark_oracle"]
    planning_seed = int(experiment["planning_seed"])
    variants = list(experiment["heuristic_variants"])
    if set(variants) != {"sum", "max", "zero"}:
        raise ValueError("heuristic_variants must contain sum, max, and zero")

    graph = load_ecs32a_dag(config)
    targets = sample_targets(graph, config)
    bundle = load_oracle(oracle_name, graph, config)

    print(f"Oracle: {oracle_name}; planning seed: {planning_seed}")
    print(f"Targets: {targets}")
    rows: list[dict] = []

    for target in targets:
        closure = nx.ancestors(graph, target) | {target}
        closure_graph = graph.subgraph(closure).copy()
        print(
            f"\nTarget {target}: prerequisite closure has "
            f"{closure_graph.number_of_nodes()} nodes and "
            f"{closure_graph.number_of_edges()} edges"
        )

        target_rows = []
        for heuristic in variants:
            row = run_lao(
                target,
                closure_graph,
                heuristic,
                oracle_name,
                bundle,
                config,
                planning_seed,
            )
            rows.append(row)
            target_rows.append(row)
            print(
                f"  h={heuristic:4s}: expanded={row['expanded_states']:6d} "
                f"time={row['planning_time_seconds']:.6f}s"
            )

        if oracle_name == "static":
            dp_row = run_dp(
                target,
                closure_graph,
                oracle_name,
                bundle,
                config,
                planning_seed,
            )
            rows.append(dp_row)
            print(
                f"  DP    : states={dp_row['expanded_states']:6d} "
                f"time={dp_row['planning_time_seconds']:.6f}s"
            )
            for row in target_rows:
                if abs(row["expected_cost"] - dp_row["expected_cost"]) > 1e-6:
                    raise AssertionError(
                        f"LAO* and DP costs differ for target {target}: "
                        f"{row['expected_cost']} vs {dp_row['expected_cost']}"
                    )

    output_path = write_results(rows, config, oracle_name)
    print_summary(rows)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
