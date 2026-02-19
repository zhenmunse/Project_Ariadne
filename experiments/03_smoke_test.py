"""
experiments/03_smoke_test.py
=============================
Planner smoke tests.

Part A: ZPD unit tests  (standard + fallback)
Part B: Trap Test       (FakeOracle, assert planner is NOT greedy)
Part C: Real Oracle     (lambda_risk=0 vs 0.5 comparison)

Usage:
    python experiments/03_smoke_test.py
"""

import os
import pickle
import sys

import networkx as nx
import torch
import yaml

# ---------- make project root importable -------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.planner_engine.zpd_utils import get_valid_actions
from src.planner_engine.solver import DAGPlanner


# ==================================================================
# Part A: ZPD unit tests
# ==================================================================

def zpd_unit_tests():
    print("=" * 60)
    print("Part A: ZPD unit tests")
    print("=" * 60)

    # --- Standard rule on a small DAG ---
    # Graph: 0->1, 0->2, 2->3  (A is dead-end, Goal only needs B)
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (0, 2), (2, 3)])

    # state={}: only source node 0 (no prereqs)
    a = get_valid_actions(G, set())
    print(f"  state={{}}:       valid = {a}")
    assert a == [0], f"Expected [0], got {a}"

    # state={0}: both 1 and 2 have prereq 0 met
    a = get_valid_actions(G, {0})
    print(f"  state={{0}}:      valid = {a}")
    assert a == [1, 2], f"Expected [1, 2], got {a}"

    # state={0,2}: node 1 has prereq 0 met; node 3 has prereq 2 met
    a = get_valid_actions(G, {0, 2})
    print(f"  state={{0,2}}:    valid = {a}")
    assert a == [1, 3], f"Expected [1, 3], got {a}"

    # state={0,1}: node 2 has prereq 0 met; node 3 needs prereq 2 (unmet)
    a = get_valid_actions(G, {0, 1})
    print(f"  state={{0,1}}:    valid = {a}")
    assert a == [2], f"Expected [2], got {a}"

    # state={0,1,2}: only 3 left
    a = get_valid_actions(G, {0, 1, 2})
    print(f"  state={{0,1,2}}:  valid = {a}")
    assert a == [3], f"Expected [3], got {a}"

    # All mastered -> empty
    a = get_valid_actions(G, {0, 1, 2, 3})
    print(f"  state=all:      valid = {a}")
    assert a == [], f"Expected [], got {a}"

    print("  [OK] Standard rule tests passed")

    # --- Fallback rule (use a cyclic graph to force fallback) ---
    # This only tests zpd_utils directly; the planner asserts DAG.
    G_cyc = nx.DiGraph()
    G_cyc.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # state={}: all nodes have unmet prereqs -> fallback
    a = get_valid_actions(G_cyc, set())
    # ratio for each: 0/1 = 0.  tie-break: score = 0 + 1e-6/(node+1)
    # node 0: 1e-6/1 = 1e-6,  node 1: 1e-6/2 = 0.5e-6,  node 2: 1e-6/3
    # best = node 0
    print(f"  fallback test:  valid = {a}")
    assert a == [0], f"Expected [0], got {a}"

    # state={0}: node 1 prereq 0 met -> standard! node 2 prereq 1 unmet.
    a = get_valid_actions(G_cyc, {0})
    print(f"  fb state={{0}}:  valid = {a}")
    assert a == [1], f"Expected [1], got {a}"

    # state={1}: node 2 prereq 1 met -> standard! node 0 prereq 2 unmet.
    # But also check ratio-based fallback with partial satisfaction:
    G_fb = nx.DiGraph()
    G_fb.add_nodes_from([0, 1, 2])
    G_fb.add_edges_from([(0, 2), (1, 2)])  # node 2 needs both 0 and 1
    # state = {}: standard = {0, 1} (no prereqs)
    a = get_valid_actions(G_fb, set())
    print(f"  fb2 state={{}}:  valid = {a}")
    assert a == [0, 1], f"Expected [0, 1], got {a}"

    print("  [OK] Fallback rule tests passed\n")


# ==================================================================
# Part B: Trap Test
# ==================================================================

class FakeOracle:
    """Hardcoded oracle for trap test.

    Returns P_succ=1.0, sigma2=0.0, T_base=<varies>
    so that Cost = T_base exactly (since (1-1)*penalty + lambda*0 = 0).

    Trap graph (edges: 0->1, 0->2, 2->3):
      - S(0) is pre-mastered; A(1) is a dead-end distractor
      - B(2) is sole prerequisite of Goal(3)

    Cost design:
      action 0 (S):    T_base = 0   (never called; pre-mastered)
      action 1 (A):    T_base = 1   (cheap but useless)
      action 2 (B):    T_base = 5
      action 3 (Goal): T_base = 5   if A NOT mastered (mask[1]==0)
                        T_base = 100 if A IS mastered  (mask[1]==1)

    Greedy from {0}: picks A (cost=1) then B (5) then Goal (100) = 106
    Optimal from {0}: picks B (5) then Goal (5) = 10
    """

    def predict_mc(self, x, edge_index, target_node, current_prereq_mask,
                   mc_samples=20, t_base=60.0):
        node = target_node.item() if hasattr(target_node, 'item') else int(target_node)
        mask = current_prereq_mask

        if node == 0:
            tb = 0.0
        elif node == 1:
            tb = 1.0
        elif node == 2:
            tb = 5.0
        elif node == 3:
            # State-dependent: mastering A (node 1) makes Goal expensive
            tb = 100.0 if mask[1].item() > 0.5 else 5.0
        else:
            tb = 10.0

        return (
            torch.tensor(1.0),   # mean_p  (P_succ = 1 -> no penalty)
            torch.tensor(0.0),   # var_p   (sigma2 = 0)
            torch.tensor(tb),    # T_base
        )


def trap_test():
    print("=" * 60)
    print("Part B: Trap Test")
    print("=" * 60)

    # Build trap graph
    #   S(0) -> A(1)        A is dead-end distractor
    #   S(0) -> B(2) -> Goal(3)
    # Note: edge (1,3) is intentionally absent.
    # Goal only requires B; A is unnecessary for reaching Goal.
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (0, 2), (2, 3)])

    assert nx.is_directed_acyclic_graph(G)

    edge_index = torch.tensor([[0, 0, 2], [1, 2, 3]], dtype=torch.long)

    config = {
        "planner": {"t_penalty": 300, "lambda_risk": 0.0},
        "oracle":  {"mc_samples": 1},
    }

    oracle = FakeOracle()
    planner = DAGPlanner(oracle, G, config, edge_index, num_nodes=4)

    # S(0) is pre-mastered as the root starting-knowledge node.
    cost, path = planner.solve({0}, {3})

    print(f"\n  Planner result:  path = {path},  cost = {cost:.1f}")
    print(f"  Greedy would pick A(1) first (cost 1 < 5)")
    print(f"  -> greedy path [1, 2, 3], total = 1 + 5 + 100 = 106")
    print(f"  Optimal skips A, goes B(2) -> Goal(3)")
    print(f"  -> optimal path [2, 3], total = 5 + 5 = 10")

    assert path == [2, 3], f"TRAP FAILED: expected [2, 3], got {path}"
    assert abs(cost - 10.0) < 1e-6, f"TRAP FAILED: expected cost=10.0, got {cost}"

    print(f"\n  [PASS] Trap test passed!  Planner is NOT greedy.\n")


# ==================================================================
# Part C: Real Oracle integration
# ==================================================================

def real_oracle_test():
    print("=" * 60)
    print("Part C: Real Oracle integration")
    print("=" * 60)

    # Load config
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = os.path.join(ROOT, cfg["data"]["processed_dir"])

    # Load graph
    with open(os.path.join(processed_dir, "graph.pkl"), "rb") as f:
        graph_data = pickle.load(f)

    nx_dag = graph_data["nx_dag"]
    num_nodes = len(graph_data["node_ids"])
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)

    print(f"  Graph: {num_nodes} nodes, {edge_index.shape[1]} edges")

    # Load oracle
    from src.oracle_core.model import MonotonicOracle

    ckpt_path = os.path.join(processed_dir, "oracle_ckpt.pt")
    if not os.path.isfile(ckpt_path):
        print("  [SKIP] No oracle checkpoint found. Run 02_train_oracle.py first.")
        return

    ckpt = torch.load(ckpt_path, weights_only=False)
    model = MonotonicOracle(
        num_nodes=ckpt["num_nodes"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        dropout=ckpt["config"]["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Oracle loaded from checkpoint")

    # Initial state: root nodes; target: last node
    root_nodes = [n for n in nx_dag.nodes() if nx_dag.in_degree(n) == 0]
    initial_state = set(root_nodes)
    target_node = graph_data["node_ids"][-1]
    target_set = {target_node}

    print(f"  Initial state (roots): {initial_state}")
    print(f"  Target: {target_set}")

    # --- Run 1: lambda_risk = 0 ---
    cfg_r1 = {
        "planner": {"t_penalty": cfg["planner"]["t_penalty"], "lambda_risk": 0.0},
        "oracle": cfg["oracle"],
    }
    planner1 = DAGPlanner(model, nx_dag, cfg_r1, edge_index, num_nodes)
    cost1, path1 = planner1.solve(initial_state, target_set)
    print(f"\n  lambda_risk = 0.0:  path = {path1},  cost = {cost1:.2f}")

    # --- Run 2: lambda_risk = 0.5 ---
    cfg_r2 = {
        "planner": {"t_penalty": cfg["planner"]["t_penalty"], "lambda_risk": 0.5},
        "oracle": cfg["oracle"],
    }
    planner2 = DAGPlanner(model, nx_dag, cfg_r2, edge_index, num_nodes)
    cost2, path2 = planner2.solve(initial_state, target_set)
    print(f"  lambda_risk = 0.5:  path = {path2},  cost = {cost2:.2f}")

    if path1 == path2:
        print("\n  [NOTE] Paths are identical. This may be due to small graph size")
        print("         or low oracle variance. Both runs completed successfully.")
    else:
        print("\n  [OK] Paths differ! Risk-aversion affects planning.")

    print(f"\n  [PASS] Real oracle integration completed.\n")


# ==================================================================
# Main
# ==================================================================

def main():
    zpd_unit_tests()
    trap_test()
    real_oracle_test()
    print("=== Step 3 COMPLETE ===")


if __name__ == "__main__":
    main()
