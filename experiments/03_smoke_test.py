"""
experiments/03_smoke_test.py
=============================
Planner smoke tests for the current Ariadne SSP formulation.

Part A: ZPD unit tests
Part B: Chain DAG LAO* test
Part C: Branch-order LAO* test
Part D: Greedy trap test
Part E: LAO* vs exact DP on a small DAG

Usage:
    python experiments/03_smoke_test.py
"""

import os
import pickle
import sys
from typing import Dict, FrozenSet, Tuple

import networkx as nx

# ---------- make project root importable -------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.planner_engine.heuristics import max_heuristic, sum_heuristic
from src.planner_engine.solver import DAGPlanner, DAGPlannerDP
from src.planner_engine.zpd_utils import get_valid_actions


class TableOracle:
    """Small deterministic oracle for planner tests."""

    def __init__(
        self,
        probs: Dict[Tuple[int, FrozenSet[int]], float],
        base_costs: Dict[int, float] | None = None,
        default_p: float = 1.0,
        default_T: float = 1.0,
    ):
        self.probs = probs
        self.base_costs = base_costs or {}
        self.default_p = default_p
        self.default_T = default_T

    def success_prob(self, v: int, mastered: FrozenSet[int]) -> float:
        return self.probs.get((v, mastered), self.default_p)

    def base_cost(self, v: int) -> float:
        return self.base_costs.get(v, self.default_T)

    def best_case_success_prob(self, v: int) -> float:
        vals = [p for (node, _state), p in self.probs.items() if node == v]
        vals.append(self.default_p)
        return max(vals)


class SuccessOnlyOracle:
    """Oracle intentionally lacking a best-case probability method."""

    def __init__(self):
        self.queries = []

    def success_prob(self, v: int, mastered: FrozenSet[int]) -> float:
        self.queries.append((v, mastered))
        return 0.25


def empty_edge_index():
    return None


# ==================================================================
# Part A: ZPD unit tests
# ==================================================================

def zpd_unit_tests():
    print("=" * 60)
    print("Part A: ZPD unit tests")
    print("=" * 60)

    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (0, 2), (2, 3)])

    checks = [
        (set(), [0]),
        ({0}, [1, 2]),
        ({0, 2}, [1, 3]),
        ({0, 1}, [2]),
        ({0, 1, 2}, [3]),
        ({0, 1, 2, 3}, []),
    ]

    for state, expected in checks:
        actual = get_valid_actions(G, state)
        print(f"  state={state}: valid = {actual}")
        assert actual == expected, f"Expected {expected}, got {actual}"

    G_cyc = nx.DiGraph()
    G_cyc.add_edges_from([(0, 1), (1, 2), (2, 0)])
    actual = get_valid_actions(G_cyc, set())
    print(f"  fallback test: valid = {actual}")
    assert actual == [0], f"Expected [0], got {actual}"

    print("  [OK] ZPD tests passed\n")


def heuristic_unit_tests():
    print("=" * 60)
    print("Part A2: Heuristic unit tests")
    print("=" * 60)

    graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    oracle = SuccessOnlyOracle()
    config = {
        "planner": {"base_cost": 2.0},
        "oracle": {"mc_samples": 1},
    }
    planner = DAGPlanner(oracle, graph, config, empty_edge_index(), num_nodes=4)

    p_bar = planner.best_case_success_prob(0)
    h_sum = sum_heuristic(frozenset(), frozenset(graph.nodes()), planner)
    h_max = max_heuristic(
        frozenset(), frozenset(graph.nodes()), planner, graph
    )

    assert p_bar == 1.0, f"Expected fallback p_bar=1.0, got {p_bar}"
    assert h_sum == 8.0, f"Expected sum heuristic 8.0, got {h_sum}"
    assert h_max == 6.0, f"Expected max heuristic 6.0, got {h_max}"
    assert oracle.queries == [], "Heuristic fallback queried success_prob"

    print(f"  fallback p_bar = {p_bar:.1f}")
    print(f"  h_sum = {h_sum:.1f}, h_max = {h_max:.1f}")
    print("  [OK] Heuristic tests passed\n")


# ==================================================================
# Part B: Chain DAG
# ==================================================================

def chain_test():
    print("=" * 60)
    print("Part B: Chain DAG")
    print("=" * 60)

    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2)])

    probs = {
        (0, frozenset()): 0.8,
        (1, frozenset({0})): 0.5,
        (2, frozenset({0, 1})): 0.9,
    }
    oracle = TableOracle(probs)
    config = {"planner": {"base_cost": 1.0}, "oracle": {"mc_samples": 1}}

    planner = DAGPlanner(oracle, G, config, empty_edge_index(), num_nodes=3)
    cost, path = planner.solve(set(), {0, 1, 2})
    expected = 1 / 0.8 + 1 / 0.5 + 1 / 0.9

    print(f"  path = {path}, cost = {cost:.6f}")
    assert path == [0, 1, 2], f"Expected [0, 1, 2], got {path}"
    assert abs(cost - expected) < 1e-6, f"Expected {expected}, got {cost}"

    print("  [OK] Chain DAG test passed\n")


# ==================================================================
# Part C: Branch ordering
# ==================================================================

def branch_order_test():
    print("=" * 60)
    print("Part C: Branch-order test")
    print("=" * 60)

    #      0
    #     / \
    #    1   2
    #     \ /
    #      3
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

    probs = {
        (0, frozenset()): 1.0,
        (1, frozenset({0})): 0.9,
        (1, frozenset({0, 2})): 0.9,
        (2, frozenset({0})): 0.3,
        (2, frozenset({0, 1})): 0.5,
        (3, frozenset({0, 1, 2})): 0.8,
    }
    oracle = TableOracle(probs)
    config = {"planner": {"base_cost": 1.0}, "oracle": {"mc_samples": 1}}

    planner = DAGPlanner(oracle, G, config, empty_edge_index(), num_nodes=4)
    cost, path = planner.solve(set(), {0, 1, 2, 3})
    expected = 1.0 + 1 / 0.9 + 1 / 0.5 + 1 / 0.8

    print(f"  path = {path}, cost = {cost:.6f}")
    assert path == [0, 1, 2, 3], f"Expected [0, 1, 2, 3], got {path}"
    assert abs(cost - expected) < 1e-6, f"Expected {expected}, got {cost}"

    print("  [OK] Branch-order test passed\n")


# ==================================================================
# Part D: Greedy trap
# ==================================================================

def trap_test():
    print("=" * 60)
    print("Part D: Greedy trap")
    print("=" * 60)

    # S(0) is pre-mastered. A(1) is cheap but harmful; B(2) unlocks Goal(3).
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (0, 2), (2, 3)])

    probs = {
        (1, frozenset({0})): 1.0,          # immediate cost 1
        (2, frozenset({0})): 0.2,          # immediate cost 5
        (2, frozenset({0, 1})): 0.2,
        (3, frozenset({0, 2})): 0.2,       # cost 5 if A is skipped
        (3, frozenset({0, 1, 2})): 0.01,   # cost 100 if A is mastered
    }
    oracle = TableOracle(probs)
    config = {"planner": {"base_cost": 1.0}, "oracle": {"mc_samples": 1}}

    planner = DAGPlanner(oracle, G, config, empty_edge_index(), num_nodes=4)
    cost, path = planner.solve({0}, {3})

    print(f"  LAO* path = {path}, cost = {cost:.1f}")
    print("  Greedy would pick A(1), then B(2), then Goal(3): 1 + 5 + 100")
    print("  Optimal skips A and picks B(2), then Goal(3): 5 + 5")

    assert path == [2, 3], f"Expected [2, 3], got {path}"
    assert abs(cost - 10.0) < 1e-6, f"Expected 10.0, got {cost}"

    print("  [OK] Greedy trap passed\n")


# ==================================================================
# Part E: LAO* vs exact DP
# ==================================================================

def lao_vs_dp_test():
    print("=" * 60)
    print("Part E: LAO* vs exact DP")
    print("=" * 60)

    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4)])

    probs = {
        (0, frozenset()): 0.8,
        (1, frozenset({0})): 0.7,
        (1, frozenset({0, 2})): 0.9,
        (2, frozenset({0})): 0.6,
        (2, frozenset({0, 1})): 0.6,
        (3, frozenset({0, 1, 2})): 0.8,
        (3, frozenset({0, 1, 2, 4})): 0.8,
        (4, frozenset({0, 2})): 0.5,
        (4, frozenset({0, 1, 2})): 0.5,
    }
    oracle = TableOracle(probs, default_p=0.75)
    config = {"planner": {"base_cost": 1.0}, "oracle": {"mc_samples": 1}}

    target = {0, 1, 2, 3, 4}
    lao = DAGPlanner(oracle, G, config, empty_edge_index(), num_nodes=5)
    dp = DAGPlannerDP(oracle, G, config, empty_edge_index(), num_nodes=5)

    lao_cost, lao_path = lao.solve(set(), target)
    dp_cost, dp_path = dp.solve(set(), target)

    print(f"  LAO*: cost = {lao_cost:.6f}, path = {lao_path}")
    print(f"  DP:   cost = {dp_cost:.6f}, path = {dp_path}")

    assert abs(lao_cost - dp_cost) < 1e-6
    assert lao_path == dp_path

    print("  [OK] LAO* matches exact DP\n")


# ==================================================================
# Optional real Oracle integration
# ==================================================================

def real_oracle_smoke():
    print("=" * 60)
    print("Optional: Real Oracle integration")
    print("=" * 60)

    try:
        import torch
        import yaml
    except ImportError:
        print("  [SKIP] torch or pyyaml missing.\n")
        return

    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = os.path.join(ROOT, cfg["data"]["processed_dir"])
    graph_path = os.path.join(processed_dir, "graph.pkl")
    ckpt_path = os.path.join(processed_dir, "oracle_ckpt.pt")

    if not os.path.isfile(graph_path) or not os.path.isfile(ckpt_path):
        print("  [SKIP] graph.pkl or oracle checkpoint missing.\n")
        return

    with open(graph_path, "rb") as f:
        graph_data = pickle.load(f)

    from src.oracle_core.model import MonotonicOracle

    ckpt = torch.load(ckpt_path, weights_only=False)
    model = MonotonicOracle(
        num_nodes=ckpt["num_nodes"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        dropout=ckpt["config"]["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    nx_dag = graph_data["nx_dag"]
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
    target_node = graph_data["node_ids"][-1]

    planner = DAGPlanner(
        model,
        nx_dag,
        cfg,
        edge_index,
        num_nodes=len(graph_data["node_ids"]),
    )
    cost, path = planner.solve(set(), {target_node})

    print(f"  target = {target_node}, path length = {len(path)}, cost = {cost:.2f}")
    print("  [OK] Real Oracle integration completed\n")


def main():
    zpd_unit_tests()
    heuristic_unit_tests()
    chain_test()
    branch_order_test()
    trap_test()
    lao_vs_dp_test()
    real_oracle_smoke()
    print("=== Step 3 COMPLETE ===")


if __name__ == "__main__":
    main()
