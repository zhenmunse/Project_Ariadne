"""Run DKT + LAO* from archived local DKT predictions."""

import json
import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DKT_DATA = ROOT / "data" / "baselines" / "pykt" / "ecs32a_ariadne"
OUTPUT = ROOT / "results" / "dkt_lao"
sys.path.insert(0, str(ROOT))

from src.planner_engine.solver import DAGPlanner


class DKTOracle:
    def __init__(self, probabilities: dict[int, float], fallback: float) -> None:
        self.probabilities = probabilities
        self.fallback = float(fallback)

    def success_prob(self, node: int, mastered: frozenset[int]) -> float:
        return self.probabilities.get(node, self.fallback)

    def base_cost(self, node: int) -> float:
        return 60.0


def is_valid_path(graph: nx.DiGraph, path: list[int]) -> bool:
    mastered: set[int] = set()
    for node in path:
        if node in mastered or not set(graph.predecessors(node)) <= mastered:
            return False
        mastered.add(node)
    return True


def main() -> None:
    with (ROOT / "configs" / "config.yaml").open() as file:
        config = yaml.safe_load(file)
    with (PROCESSED / "graph.pkl").open("rb") as file:
        graph = pickle.load(file)
    with (PROCESSED / "train_sessions.pkl").open("rb") as file:
        train_samples = pickle.load(file)

    node_probabilities = pd.read_csv(DKT_DATA / "train_node_probabilities.csv")
    validation_metrics = json.loads(
        (DKT_DATA / "validation_metrics.json").read_text(encoding="utf8")
    )
    required = {"node_id", "mean_probability", "samples"}
    if not required <= set(node_probabilities.columns):
        raise ValueError(f"DKT artifact missing columns: {required - set(node_probabilities.columns)}")
    oracle = DKTOracle(
        dict(zip(node_probabilities["node_id"], node_probabilities["mean_probability"])),
        float(node_probabilities["mean_probability"].mean()),
    )

    dag: nx.DiGraph = graph["nx_dag"]
    num_nodes = len(graph["node_ids"])
    observed_nodes = {target for _, target, _ in train_samples}
    targets = sorted(node for node in observed_nodes if dag.in_degree(node) > 0)
    rng = np.random.default_rng(config["seed"])
    target_count = min(config["experiments"]["num_targets"], len(targets))
    targets = sorted(rng.choice(targets, size=target_count, replace=False).tolist())
    planner_config = {"planner": config["planner"], "oracle": config["oracle"]}
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)

    trajectories = []
    for target in targets:
        closure = nx.ancestors(dag, target) | {target}
        closure_graph = dag.subgraph(closure).copy()
        planner = DAGPlanner(
            oracle, closure_graph, planner_config, edge_index, num_nodes
        )
        started = time.perf_counter()
        result = planner.solve_result(set(), closure)
        start = frozenset()
        path = DAGPlanner._extract_path(start, frozenset(closure), result.policy)
        elapsed = time.perf_counter() - started
        valid = is_valid_path(closure_graph, path)
        assert result.converged and path and path[-1] == target and valid, (target, path)
        trajectories.append({
            "target_node": target,
            "expected_total_cost": result.values[start],
            "path_length": len(path),
            "required_nodes": len(closure),
            "off_target_actions": len(set(path) - closure),
            "expanded_states": result.expanded_count,
            "iterations": result.iterations,
            "planning_seconds": elapsed,
            "converged": result.converged,
            "path_is_valid": valid,
            "path": json.dumps(path),
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([validation_metrics]).to_csv(
        OUTPUT / "oracle_valid_metrics.csv", index=False
    )
    pd.DataFrame(trajectories).to_csv(OUTPUT / "planner_trajectories.csv", index=False)
    summary = {
        "condition": "DKT + LAO*",
        "targets": targets,
        "mean_expected_total_cost": float(np.mean([row["expected_total_cost"] for row in trajectories])),
        "mean_path_length": float(np.mean([row["path_length"] for row in trajectories])),
        "mean_off_target_actions": float(np.mean([row["off_target_actions"] for row in trajectories])),
        "mean_expanded_states": float(np.mean([row["expanded_states"] for row in trajectories])),
        "total_planning_seconds": float(sum(row["planning_seconds"] for row in trajectories)),
        "all_paths_valid": all(row["path_is_valid"] for row in trajectories),
        "all_converged": all(row["converged"] for row in trajectories),
        "probability_source": "archived_dkt_train_node_predictions",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    print(pd.DataFrame([validation_metrics]).to_string(index=False))
    print(pd.DataFrame(trajectories).drop(columns="path").to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
