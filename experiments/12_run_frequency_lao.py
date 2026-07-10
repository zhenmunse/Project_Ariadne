"""Run the ECS32A FrequencyOracle + LAO* baseline."""

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
OUTPUT = ROOT / "results" / "frequency_lao"
sys.path.insert(0, str(ROOT))

from src.planner_engine.baselines import FrequencyOracle
from src.planner_engine.solver import DAGPlanner


def auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    ranks = pd.Series(probabilities).rank(method="average").to_numpy()
    positives = labels == 1.0
    n_positive = positives.sum()
    n_negative = len(labels) - n_positive
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


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
    with (PROCESSED / "valid_sessions.pkl").open("rb") as file:
        valid_samples = pickle.load(file)

    dag: nx.DiGraph = graph["nx_dag"]
    num_nodes = len(graph["node_ids"])
    oracle = FrequencyOracle(train_samples, num_nodes, graph["node_id_to_idx"])

    labels = np.array([label for _, _, label in valid_samples])
    probabilities = np.array(
        [oracle.success_prob(target, frozenset()) for _, target, _ in valid_samples]
    )
    binary = np.isin(labels, [0.0, 1.0])
    binary_labels = labels[binary]
    binary_probabilities = probabilities[binary]
    oracle_metrics = {
        "samples": len(labels),
        "binary_samples": int(binary.sum()),
        "mse": float(np.mean((labels - probabilities) ** 2)),
        "rmse": float(np.sqrt(np.mean((labels - probabilities) ** 2))),
        "mae": float(np.mean(np.abs(labels - probabilities))),
        "auc": auc_score(binary_labels, binary_probabilities),
        "accuracy": float(np.mean((binary_probabilities >= 0.5) == (binary_labels == 1.0))),
    }

    observed_nodes = {target for _, target, _ in train_samples}
    targets = sorted(node for node in observed_nodes if dag.in_degree(node) > 0)
    rng = np.random.default_rng(config["seed"])
    target_count = min(config["experiments"]["num_targets"], len(targets))
    targets = sorted(rng.choice(targets, size=target_count, replace=False).tolist())
    planner_config = {"planner": config["planner"], "oracle": config["oracle"]}
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)

    trajectories = []
    for target in targets:
        planner = DAGPlanner(oracle, dag, planner_config, edge_index, num_nodes)
        started = time.perf_counter()
        result = planner.solve_result(set(), {target})
        start = frozenset()
        path = DAGPlanner._extract_path(start, frozenset({target}), result.policy)
        cost = result.values[start]
        elapsed = time.perf_counter() - started
        valid = is_valid_path(dag, path)
        assert result.converged and path and path[-1] == target and valid, (target, path)
        required = nx.ancestors(dag, target) | {target}
        trajectories.append({
            "target_node": target,
            "expected_total_cost": cost,
            "path_length": len(path),
            "required_nodes": len(required),
            "off_target_actions": len(set(path) - required),
            "expanded_states": result.expanded_count,
            "iterations": result.iterations,
            "planning_seconds": elapsed,
            "converged": result.converged,
            "path_is_valid": valid,
            "path": json.dumps(path),
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([oracle_metrics]).to_csv(OUTPUT / "oracle_valid_metrics.csv", index=False)
    pd.DataFrame(trajectories).to_csv(OUTPUT / "planner_trajectories.csv", index=False)
    summary = {
        "condition": "FrequencyOracle + LAO*",
        "targets": targets,
        "mean_expected_total_cost": float(np.mean([row["expected_total_cost"] for row in trajectories])),
        "mean_path_length": float(np.mean([row["path_length"] for row in trajectories])),
        "mean_off_target_actions": float(np.mean([row["off_target_actions"] for row in trajectories])),
        "total_planning_seconds": float(sum(row["planning_seconds"] for row in trajectories)),
        "all_paths_valid": all(row["path_is_valid"] for row in trajectories),
        "all_converged": all(row["converged"] for row in trajectories),
        "planner_heuristic": config["planner"]["heuristic"],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(pd.DataFrame([oracle_metrics]).to_string(index=False))
    print(pd.DataFrame(trajectories).drop(columns="path").to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
