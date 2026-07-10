"""Run the ECS32A BKT + LAO* baseline."""

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
PARAMETERS = ROOT / "data" / "baselines" / "pybkt" / "concept_parameters.csv"
OUTPUT = ROOT / "results" / "bkt_lao"
sys.path.insert(0, str(ROOT))

from src.planner_engine.solver import DAGPlanner


class BKTOracle:
    """Convert fitted BKT parameters into expected attempt costs."""

    def __init__(self, parameters: pd.DataFrame, fallback: float) -> None:
        self.parameters = {
            int(row.concept_id): (
                float(row.p_init),
                float(row.p_learn),
                float(row.p_guess),
                float(row.p_slip),
            )
            for row in parameters.itertuples()
        }
        self.fallback = (float(fallback), 0.1, 0.5, 0.1)

    def _parameters(self, node: int) -> tuple[float, float, float, float]:
        return self.parameters.get(node, self.fallback)

    def success_prob(self, node: int, mastered: frozenset[int]) -> float:
        p_init, _p_learn, p_guess, p_slip = self._parameters(node)
        return p_init * (1.0 - p_slip) + (1.0 - p_init) * p_guess

    def action_cost(self, node: int, mastered: frozenset[int]) -> float:
        p_init, p_learn, p_guess, p_slip = self._parameters(node)
        belief = p_init
        survival = 1.0
        expected_attempts = 0.0

        for _ in range(100_000):
            expected_attempts += survival
            correct = belief * (1.0 - p_slip) + (1.0 - belief) * p_guess
            survival *= 1.0 - correct
            if survival < 1e-12:
                return 60.0 * expected_attempts

            posterior_known = belief * p_slip / max(1.0 - correct, 1e-12)
            belief = posterior_known + (1.0 - posterior_known) * p_learn

        raise ValueError(f"BKT expected-attempt calculation did not converge for node {node}")

    def base_cost(self, node: int) -> float:
        return 60.0


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

    parameters = pd.read_csv(PARAMETERS)
    required = {"concept_id", "p_init", "p_learn", "p_guess", "p_slip"}
    if not required <= set(parameters.columns):
        raise ValueError(f"BKT parameter file missing columns: {required - set(parameters.columns)}")

    fallback = np.mean([label for _, _, label in train_samples])
    oracle = BKTOracle(parameters, fallback)
    toy = BKTOracle(
        pd.DataFrame([{"concept_id": 0, "p_init": 0.0, "p_learn": 0.0, "p_guess": 0.5, "p_slip": 0.0}]),
        0.0,
    )
    assert abs(toy.action_cost(0, frozenset()) - 120.0) < 1e-9

    labels = np.array([label for _, _, label in valid_samples])
    probabilities = np.array([
        oracle.success_prob(target, frozenset())
        for _, target, _ in valid_samples
    ])
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
        "probability_source": "bkt_initial_correct_probability",
    }

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
        path = DAGPlanner._extract_path(frozenset(), frozenset(closure), result.policy)
        elapsed = time.perf_counter() - started
        valid = is_valid_path(closure_graph, path)
        assert result.converged and path and path[-1] == target and valid, (target, path)
        trajectories.append({
            "target_node": target,
            "expected_total_cost": result.values[frozenset()],
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
    pd.DataFrame([oracle_metrics]).to_csv(OUTPUT / "oracle_valid_metrics.csv", index=False)
    pd.DataFrame(trajectories).to_csv(OUTPUT / "planner_trajectories.csv", index=False)
    summary = {
        "condition": "BKT + LAO*",
        "targets": targets,
        "mean_expected_total_cost": float(np.mean([row["expected_total_cost"] for row in trajectories])),
        "mean_path_length": float(np.mean([row["path_length"] for row in trajectories])),
        "mean_off_target_actions": float(np.mean([row["off_target_actions"] for row in trajectories])),
        "mean_expanded_states": float(np.mean([row["expanded_states"] for row in trajectories])),
        "total_planning_seconds": float(sum(row["planning_seconds"] for row in trajectories)),
        "all_paths_valid": all(row["path_is_valid"] for row in trajectories),
        "all_converged": all(row["converged"] for row in trajectories),
        "probability_source": "bkt_expected_attempt_cost",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    print(pd.DataFrame([oracle_metrics]).to_string(index=False))
    print(pd.DataFrame(trajectories).drop(columns="path").to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
