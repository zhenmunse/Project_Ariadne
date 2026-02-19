"""
experiments/04a_baseline_check.py
==================================
Minimal self-check for baselines:
  1) GreedyPlanner vs DAGPlanner on the trap graph (FakeOracle)
  2) FrequencyOracle interface sanity check

Usage:
    python experiments/04a_baseline_check.py
"""

import os
import pickle
import sys

import networkx as nx
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.planner_engine.solver import DAGPlanner
from src.planner_engine.baselines import GreedyPlanner, FrequencyOracle


# ==================================================================
# FakeOracle (same as Step 3 trap test)
# ==================================================================

class FakeOracle:
    """Trap-graph oracle.

    Cost design (P_succ=1, sigma2=0 -> cost = T_base):
      node 0 (S):  0   (pre-mastered)
      node 1 (A):  1   (cheap bait)
      node 2 (B):  5
      node 3 (Goal): 5 if A not mastered, 100 if A mastered
    """

    def predict_mc(self, x, edge_index, target_node, current_prereq_mask,
                   mc_samples=20, t_base=60.0):
        node = target_node.item() if hasattr(target_node, "item") else int(target_node)
        mask = current_prereq_mask
        if node == 0:
            tb = 0.0
        elif node == 1:
            tb = 1.0
        elif node == 2:
            tb = 5.0
        elif node == 3:
            tb = 100.0 if mask[1].item() > 0.5 else 5.0
        else:
            tb = 10.0
        return torch.tensor(1.0), torch.tensor(0.0), torch.tensor(tb)


# ==================================================================
# Test 1: Trap graph  -- DP vs Greedy
# ==================================================================

def test_trap_graph():
    print("=" * 60)
    print("Test 1: Trap graph (DP vs Greedy)")
    print("=" * 60)

    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (0, 2), (2, 3)])

    edge_index = torch.tensor([[0, 0, 2], [1, 2, 3]], dtype=torch.long)
    oracle = FakeOracle()

    config = {
        "planner": {"t_penalty": 300, "lambda_risk": 0.0},
        "oracle": {"mc_samples": 1},
    }

    # --- DAGPlanner ---
    dp = DAGPlanner(oracle, G, config, edge_index, num_nodes=4)
    dp_cost, dp_path = dp.solve({0}, {3})
    print(f"  DAGPlanner:    path={dp_path}  cost={dp_cost:.1f}")

    # --- GreedyPlanner ---
    gp = GreedyPlanner(oracle, G, config, edge_index, num_nodes=4)
    gp_cost, gp_path = gp.solve({0}, {3})
    print(f"  GreedyPlanner: path={gp_path}  cost={gp_cost:.1f}")

    # Assertions
    assert dp_path == [2, 3], f"DP path expected [2,3], got {dp_path}"
    assert gp_path == [1, 2, 3], f"Greedy path expected [1,2,3], got {gp_path}"
    assert dp_cost < gp_cost, "DP should beat Greedy on trap graph"

    print(f"\n  DP cost ({dp_cost:.1f}) < Greedy cost ({gp_cost:.1f})")
    print(f"  [PASS] Greedy falls into trap; DP avoids it.\n")


# ==================================================================
# Test 2: FrequencyOracle interface check
# ==================================================================

def test_frequency_oracle():
    print("=" * 60)
    print("Test 2: FrequencyOracle interface")
    print("=" * 60)

    # Load real data if available; otherwise use toy samples
    processed_dir = os.path.join(ROOT, "data", "processed")
    graph_path = os.path.join(processed_dir, "graph.pkl")

    if os.path.isfile(graph_path):
        with open(graph_path, "rb") as f:
            graph_data = pickle.load(f)
        with open(os.path.join(processed_dir, "train_sessions.pkl"), "rb") as f:
            samples = pickle.load(f)
        num_nodes = len(graph_data["node_ids"])
        nid2idx = graph_data["node_id_to_idx"]
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
        print(f"  Using real data: {num_nodes} nodes, {len(samples)} samples")
    else:
        # Toy fallback
        num_nodes = 4
        nid2idx = {0: 0, 1: 1, 2: 2, 3: 3}
        samples = [
            ([], 0, 0.8),
            ([(0, 0.8)], 1, 0.6),
            ([(0, 0.8), (1, 0.6)], 2, 1.0),
        ]
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
        print(f"  Using toy data: {num_nodes} nodes, {len(samples)} samples")

    freq = FrequencyOracle(samples, num_nodes, nid2idx, t_base=60.0)

    print(f"  Global mean label: {freq.global_mean:.4f}")
    print(f"  Per-node p_hat:    {freq.p_hat.tolist()}")

    # Test every node
    mask = torch.zeros(num_nodes)
    for nid in range(num_nodes):
        tgt = torch.tensor(nid, dtype=torch.long)
        mean_p, var_p, tb = freq.predict_mc(
            torch.zeros(num_nodes, 2), edge_index, tgt, mask
        )
        mp = mean_p.item()
        vp = var_p.item()
        print(f"    node {nid}: mean_p={mp:.4f}  var_p={vp:.4f}  t_base={tb.item():.1f}")
        assert 0.0 <= mp <= 1.0, f"mean_p out of range for node {nid}"
        assert vp >= 0.0, f"var_p negative for node {nid}"

    print(f"\n  [PASS] FrequencyOracle interface OK.\n")


# ==================================================================
# Main
# ==================================================================

def main():
    test_trap_graph()
    test_frequency_oracle()
    print("=== Step 4.1 COMPLETE ===")


if __name__ == "__main__":
    main()
