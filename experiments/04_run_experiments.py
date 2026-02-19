"""
experiments/04_run_experiments.py
==================================
Batch experiment pipeline for Project Ariadne v0.2.

Runs 4 strategies on sampled target nodes, records trajectories,
and exports results/trajectories.json + results/metrics.csv.

Strategies:
  A  Ariadne-Full:     MonotonicOracle  + DAGPlanner   (lambda_risk=0.5)
  B  Ariadne-Neutral:  MonotonicOracle  + DAGPlanner   (lambda_risk=0)
  C  Myopic:           MonotonicOracle  + GreedyPlanner (lambda_risk=0)
  D  No-Prior:         FrequencyOracle  + DAGPlanner   (lambda_risk=0)

Usage:
    python experiments/04_run_experiments.py
"""

import csv
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.oracle_core.model import MonotonicOracle
from src.planner_engine.solver import DAGPlanner
from src.planner_engine.baselines import GreedyPlanner, FrequencyOracle


# ==================================================================
# Helpers
# ==================================================================

def compute_node_depths(G: nx.DiGraph) -> Dict[int, int]:
    """Compute topological depth (longest path from any root) for each node."""
    depths: Dict[int, int] = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            depths[node] = 0
        else:
            depths[node] = max(depths[p] for p in preds) + 1
    return depths


def compute_bottleneck(
    oracle,
    path: List[int],
    edge_index: torch.Tensor,
    num_nodes: int,
    initial_state: Set[int],
    mc_samples: int,
) -> Dict[str, Any]:
    """Walk the path step-by-step, query oracle, find the bottleneck node."""
    if not path:
        return {"node_id": -1, "p_succ": 1.0}

    state = set(initial_state)
    worst_p = 1.0
    worst_node = path[0]

    x = torch.zeros(num_nodes, 2)
    for action in path:
        mask = torch.zeros(num_nodes)
        for n in state:
            if 0 <= n < num_nodes:
                mask[n] = 1.0
        tgt = torch.tensor(action, dtype=torch.long)
        mean_p, _, _ = oracle.predict_mc(
            x, edge_index, tgt, mask, mc_samples=mc_samples,
        )
        p = mean_p.item()
        if p < worst_p:
            worst_p = p
            worst_node = action
        state.add(action)

    return {"node_id": int(worst_node), "p_succ": float(worst_p)}


def select_subgraph(
    nx_dag: nx.DiGraph,
    subgraph_size: int,
) -> nx.DiGraph:
    """Select an induced DAG subgraph of desired size.

    Takes the first K nodes (sorted by node_id) and builds the induced subgraph.
    Gracefully handles graphs smaller than subgraph_size.
    """
    all_nodes = sorted(nx_dag.nodes())
    k = min(len(all_nodes), subgraph_size)
    sub_nodes = all_nodes[:k]
    sub = nx_dag.subgraph(sub_nodes).copy()
    assert nx.is_directed_acyclic_graph(sub), "Subgraph is not a DAG!"
    return sub


def rebuild_edge_index(G: nx.DiGraph, num_nodes: int) -> torch.Tensor:
    """Build edge_index [2, E] from nx.DiGraph (nodes must be 0..N-1)."""
    if G.number_of_edges() == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    src, dst = zip(*G.edges())
    return torch.tensor([list(src), list(dst)], dtype=torch.long)


# ==================================================================
# Main
# ==================================================================

def main():
    t0 = time.time()

    # --- Load config -------------------------------------------------
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    processed_dir = os.path.join(ROOT, cfg["data"]["processed_dir"])
    exp_cfg = cfg["experiments"]
    results_dir = os.path.join(ROOT, exp_cfg["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    num_targets = exp_cfg["num_targets"]
    min_depth = exp_cfg["min_depth"]
    subgraph_size = exp_cfg["subgraph_size"]
    mc_samples = cfg["oracle"]["mc_samples"]

    # --- Load graph --------------------------------------------------
    with open(os.path.join(processed_dir, "graph.pkl"), "rb") as f:
        graph_data = pickle.load(f)

    full_dag: nx.DiGraph = graph_data["nx_dag"]
    full_num_nodes = len(graph_data["node_ids"])
    node_id_to_idx = graph_data["node_id_to_idx"]

    print(f"Full graph: {full_num_nodes} nodes, {full_dag.number_of_edges()} edges")

    # --- Select subgraph ---------------------------------------------
    sub_dag = select_subgraph(full_dag, subgraph_size)
    sub_nodes = sorted(sub_dag.nodes())
    sub_num_nodes = len(sub_nodes)
    sub_edge_index = rebuild_edge_index(sub_dag, sub_num_nodes)

    print(f"Subgraph:   {sub_num_nodes} nodes, {sub_dag.number_of_edges()} edges, DAG={nx.is_directed_acyclic_graph(sub_dag)}")

    # --- Load MonotonicOracle ----------------------------------------
    ckpt_path = os.path.join(processed_dir, "oracle_ckpt.pt")
    ckpt = torch.load(ckpt_path, weights_only=False)
    mono_oracle = MonotonicOracle(
        num_nodes=ckpt["num_nodes"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        dropout=ckpt["config"]["dropout"],
    )
    mono_oracle.load_state_dict(ckpt["state_dict"])
    mono_oracle.eval()
    print(f"MonotonicOracle loaded ({ckpt['num_nodes']} nodes)")

    # --- Load FrequencyOracle ----------------------------------------
    with open(os.path.join(processed_dir, "train_sessions.pkl"), "rb") as f:
        train_samples = pickle.load(f)

    freq_oracle = FrequencyOracle(
        train_samples, ckpt["num_nodes"], node_id_to_idx, t_base=60.0,
    )
    print(f"FrequencyOracle loaded (global_mean={freq_oracle.global_mean:.4f})")

    # Use the full graph edge_index for oracle calls (oracle was trained on it)
    oracle_edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)

    # --- Sample target nodes -----------------------------------------
    depths = compute_node_depths(sub_dag)
    max_depth = max(depths.values()) if depths else 0
    print(f"Node depths: max={max_depth}, min_depth_filter={min_depth}")

    # Adaptively lower min_depth if needed
    effective_min_depth = min_depth
    deep_nodes = [n for n, d in depths.items() if d >= effective_min_depth]
    while len(deep_nodes) == 0 and effective_min_depth > 0:
        effective_min_depth -= 1
        deep_nodes = [n for n, d in depths.items() if d >= effective_min_depth]

    if not deep_nodes:
        # Last resort: use all non-root nodes
        roots = {n for n in sub_dag.nodes() if sub_dag.in_degree(n) == 0}
        deep_nodes = sorted(sub_dag.nodes() - roots)
        if not deep_nodes:
            deep_nodes = sorted(sub_dag.nodes())

    deep_nodes.sort()
    rng = np.random.RandomState(seed)
    n_sample = min(num_targets, len(deep_nodes))
    target_nodes = list(rng.choice(deep_nodes, size=n_sample, replace=False))
    target_nodes.sort()

    print(f"Sampled {len(target_nodes)} targets (depth>={effective_min_depth}): {target_nodes}")

    # --- Strategy definitions ----------------------------------------
    strategies = {
        "Ariadne-Full": {
            "oracle": mono_oracle,
            "planner_cls": DAGPlanner,
            "config": {
                "planner": {"t_penalty": cfg["planner"]["t_penalty"], "lambda_risk": 0.5},
                "oracle": cfg["oracle"],
            },
        },
        "Ariadne-Neutral": {
            "oracle": mono_oracle,
            "planner_cls": DAGPlanner,
            "config": {
                "planner": {"t_penalty": cfg["planner"]["t_penalty"], "lambda_risk": 0.0},
                "oracle": cfg["oracle"],
            },
        },
        "Myopic": {
            "oracle": mono_oracle,
            "planner_cls": GreedyPlanner,
            "config": {
                "planner": {"t_penalty": cfg["planner"]["t_penalty"], "lambda_risk": 0.0},
                "oracle": cfg["oracle"],
            },
        },
        "No-Prior": {
            "oracle": freq_oracle,
            "planner_cls": DAGPlanner,
            "config": {
                "planner": {"t_penalty": cfg["planner"]["t_penalty"], "lambda_risk": 0.0},
                "oracle": cfg["oracle"],
            },
        },
    }

    # --- Run experiment matrix ---------------------------------------
    print(f"\n{'='*60}")
    print(f"Running experiment matrix: {len(strategies)} strategies x {len(target_nodes)} targets")
    print(f"{'='*60}")

    trajectories: List[Dict[str, Any]] = []
    initial_state: Set[int] = set()

    for ti, target in enumerate(target_nodes):
        target_set = {target}
        print(f"\n  Target {ti+1}/{len(target_nodes)}: node {target} (depth={depths.get(target, '?')})")

        for sname, sdef in strategies.items():
            oracle = sdef["oracle"]
            planner = sdef["planner_cls"](
                oracle=oracle,
                nx_graph=sub_dag,
                config=sdef["config"],
                edge_index=oracle_edge_index,
                num_nodes=ckpt["num_nodes"],  # oracle's num_nodes
            )

            cost, path = planner.solve(initial_state, target_set)

            # Compute bottleneck
            bn = compute_bottleneck(
                oracle, path, oracle_edge_index,
                ckpt["num_nodes"], initial_state, mc_samples,
            )

            record = {
                "target_node": int(target),
                "strategy_name": sname,
                "path": [int(n) for n in path],
                "expected_total_cost": float(cost),
                "path_length": len(path),
                "bottleneck_node": {
                    "node_id": bn["node_id"],
                    "p_succ": round(bn["p_succ"], 6),
                },
            }
            trajectories.append(record)
            print(f"    {sname:20s}: cost={cost:9.2f}  len={len(path)}  path={path}  bn={bn['node_id']}(p={bn['p_succ']:.3f})")

    # --- Check: any target where Greedy > DP? ------------------------
    print(f"\n{'='*60}")
    print("Greedy vs DP comparison")
    print(f"{'='*60}")

    dp_costs = {}
    greedy_costs = {}
    for rec in trajectories:
        t = rec["target_node"]
        if rec["strategy_name"] == "Ariadne-Neutral":
            dp_costs[t] = rec["expected_total_cost"]
        elif rec["strategy_name"] == "Myopic":
            greedy_costs[t] = rec["expected_total_cost"]

    diffs = []
    for t in target_nodes:
        if t in dp_costs and t in greedy_costs:
            d = greedy_costs[t] - dp_costs[t]
            diffs.append((t, d, dp_costs[t], greedy_costs[t]))

    diffs.sort(key=lambda x: -x[1])
    n_greedy_worse = sum(1 for _, d, _, _ in diffs if d > 0.01)

    if n_greedy_worse > 0:
        print(f"  {n_greedy_worse}/{len(diffs)} targets: Greedy cost > DP cost")
    else:
        print(f"  No targets where Greedy is strictly worse (may be due to small graph)")

    print(f"  Top-{min(5, len(diffs))} largest diff (greedy - DP):")
    for t, d, dp_c, gr_c in diffs[:5]:
        print(f"    node {t}: diff={d:+.2f}  (DP={dp_c:.2f}, Greedy={gr_c:.2f})")

    # --- Save outputs ------------------------------------------------
    json_path = os.path.join(results_dir, "trajectories.json")
    with open(json_path, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"\nSaved trajectories.json -> {json_path}")

    csv_path = os.path.join(results_dir, "metrics.csv")
    fieldnames = [
        "target_node", "strategy_name", "expected_total_cost",
        "path_length", "bottleneck_node_id", "bottleneck_p",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in trajectories:
            writer.writerow({
                "target_node": rec["target_node"],
                "strategy_name": rec["strategy_name"],
                "expected_total_cost": round(rec["expected_total_cost"], 4),
                "path_length": rec["path_length"],
                "bottleneck_node_id": rec["bottleneck_node"]["node_id"],
                "bottleneck_p": round(rec["bottleneck_node"]["p_succ"], 6),
            })
    print(f"Saved metrics.csv     -> {csv_path}")

    # --- Summary stats -----------------------------------------------
    print(f"\n{'='*60}")
    print("Summary by strategy")
    print(f"{'='*60}")

    from collections import defaultdict
    by_strat: Dict[str, List[float]] = defaultdict(list)
    for rec in trajectories:
        by_strat[rec["strategy_name"]].append(rec["expected_total_cost"])

    for sname in strategies:
        costs = by_strat[sname]
        print(f"  {sname:20s}: mean={np.mean(costs):9.2f}  std={np.std(costs):8.2f}  n={len(costs)}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n=== Step 4.2 COMPLETE ===")


if __name__ == "__main__":
    main()
