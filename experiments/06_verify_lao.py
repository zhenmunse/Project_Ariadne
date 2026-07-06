"""
experiments/06_verify_lao.py
=============================
Verify LAO* against exact memoized DP on small prerequisite DAGs.

Usage:
    python experiments/06_verify_lao.py
"""

import os
import sys
import time
import csv
from typing import FrozenSet

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.planner_engine.solver import DAGPlanner, DAGPlannerDP


class MonotoneToyOracle:
    """Fast monotone oracle used for solver verification."""

    def __init__(self, graph: nx.DiGraph, base_T: float = 1.0):
        self.graph = graph
        self.base_T = base_T

    def success_prob(self, v: int, mastered: FrozenSet[int]) -> float:
        preds = set(self.graph.predecessors(v))
        prereq_bonus = 0.1 * len(preds & set(mastered))
        state_bonus = 0.01 * len(mastered)
        node_penalty = 0.005 * (v % 7)
        return min(0.95, max(0.2, 0.45 + prereq_bonus + state_bonus - node_penalty))

    def base_cost(self, v: int) -> float:
        return self.base_T + 0.05 * (v % 5)

    def best_case_success_prob(self, v: int) -> float:
        return 0.95


def empty_edge_index():
    return None


def run_case(name: str, graph: nx.DiGraph, target: set[int]) -> None:
    config = {"planner": {"base_cost": 1.0}, "oracle": {"mc_samples": 1}}
    oracle = MonotoneToyOracle(graph)
    num_nodes = max(graph.nodes()) + 1 if graph.number_of_nodes() else 0

    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} target={len(target)}")

    lao = DAGPlanner(oracle, graph, config, empty_edge_index(), num_nodes=num_nodes)
    dp = DAGPlannerDP(oracle, graph, config, empty_edge_index(), num_nodes=num_nodes)

    t0 = time.time()
    lao_result = lao.solve_result(set(), target)
    start = frozenset()
    lao_cost = lao_result.values[start]
    lao_path = DAGPlanner._extract_path(start, frozenset(target), lao_result.policy)
    lao_time = time.time() - t0

    t1 = time.time()
    dp_cost, dp_path = dp.solve(set(), target)
    dp_time = time.time() - t1

    print(f"LAO*: cost={lao_cost:.6f} len={len(lao_path)} expanded={lao_result.expanded_count} iterations={lao_result.iterations} time={lao_time:.3f}s")
    print(f"DP:   cost={dp_cost:.6f} len={len(dp_path)} states={len(dp.memo)} time={dp_time:.3f}s")

    assert lao_result.converged, "LAO* did not converge"
    assert abs(lao_cost - dp_cost) < 1e-6, (lao_cost, dp_cost)
    assert lao_path == dp_path, (lao_path, dp_path)
    print("[OK] LAO* matches exact DP\n")


def build_handmade_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([
        (0, 1), (0, 2),
        (1, 3), (2, 3),
        (2, 4), (3, 5), (4, 5),
    ])
    return G


def build_ecs32a_subgraph(k: int = 15) -> nx.DiGraph | None:
    edge_path = os.path.join(ROOT, "data", "ecs32a_dag_edges_required_full_v1.csv")
    if not os.path.isfile(edge_path):
        return None

    G_full = nx.DiGraph()
    with open(edge_path, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        if {"src", "dst"} <= set(columns):
            src_col, dst_col = "src", "dst"
        elif {"source", "target"} <= set(columns):
            src_col, dst_col = "source", "target"
        else:
            src_col, dst_col = columns[:2]

        for row in reader:
            G_full.add_edge(int(row[src_col]), int(row[dst_col]))

    nodes = sorted(G_full.nodes())[:k]
    G = G_full.subgraph(nodes).copy()
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("ECS32A subgraph is not a DAG")
    return G


def main() -> None:
    G_small = build_handmade_graph()
    run_case("Handmade DAG", G_small, set(G_small.nodes()))

    G_ecs = build_ecs32a_subgraph(k=15)
    if G_ecs is None:
        print("[SKIP] ECS32A edge CSV not found")
        return
    run_case("ECS32A 15-node induced subgraph", G_ecs, set(G_ecs.nodes()))


if __name__ == "__main__":
    main()
