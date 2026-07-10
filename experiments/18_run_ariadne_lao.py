"""Run Ariadne's MonotonicOracle with the LAO* planner."""

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
OUTPUT = ROOT / "results" / "ariadne_lao"
sys.path.insert(0, str(ROOT))

from src.oracle_core.dataset import get_dataloader
from src.oracle_core.model import MonotonicOracle
from src.planner_engine.solver import DAGPlanner


class AriadneOracle:
    """Planner adapter for the monotonic checkpoint Oracle."""

    def __init__(self, model, edge_index, num_nodes: int, mc_samples: int):
        self.model = model
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.mc_samples = mc_samples
        self._probabilities = {}
        self._best_case = {}

    def _predict(self, node: int, mastered: frozenset[int]) -> float:
        x = torch.zeros(self.num_nodes, 2)
        mask = torch.zeros(self.num_nodes)
        for mastered_node in mastered:
            mask[mastered_node] = 1.0
        probability, _, _ = self.model.predict_mc(
            x,
            self.edge_index,
            torch.tensor(node, dtype=torch.long),
            mask,
            mc_samples=self.mc_samples,
        )
        return float(probability.item())

    def success_prob(self, node: int, mastered: frozenset[int]) -> float:
        key = (node, mastered)
        if key not in self._probabilities:
            self._probabilities[key] = self._predict(node, mastered)
        return self._probabilities[key]

    def best_case_success_prob(self, node: int) -> float:
        if node not in self._best_case:
            all_mastered = frozenset(range(self.num_nodes))
            self._best_case[node] = self._predict(node, all_mastered)
        return self._best_case[node]

    def base_cost(self, node: int) -> float:
        return 60.0


def auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    ranks = pd.Series(probabilities).rank(method="average").to_numpy()
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        raise ValueError("AUC requires both binary classes")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def valid_path(graph: nx.DiGraph, path: list[int], target: int) -> bool:
    mastered: set[int] = set()
    for node in path:
        if node in mastered or not set(graph.predecessors(node)) <= mastered:
            return False
        mastered.add(node)
    return bool(path) and path[-1] == target


def load_oracle(graph: dict, checkpoint: dict, mc_samples: int) -> AriadneOracle:
    if checkpoint["node_id_to_idx"] != graph["node_id_to_idx"]:
        raise ValueError("checkpoint and graph use different node mappings")
    model = MonotonicOracle(
        num_nodes=checkpoint["num_nodes"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    return AriadneOracle(model, edge_index, checkpoint["num_nodes"], mc_samples)


def validation_metrics(graph: dict, checkpoint: dict) -> dict:
    with (PROCESSED / "valid_sessions.pkl").open("rb") as file:
        samples = pickle.load(file)

    model = MonotonicOracle(
        num_nodes=checkpoint["num_nodes"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    loader = get_dataloader(
        samples,
        graph["node_id_to_idx"],
        len(graph["node_ids"]),
        batch_size=checkpoint["config"]["batch_size"],
        shuffle=False,
    )

    probabilities, labels = [], []
    with torch.no_grad():
        for x, target, mask, label in loader:
            probability, _ = model.forward_batch(x, edge_index, target, mask)
            probabilities.append(probability)
            labels.append(label)

    y_prob = torch.cat(probabilities).numpy()
    y_true = torch.cat(labels).numpy()
    binary = np.isin(y_true, [0.0, 1.0])
    binary_true, binary_prob = y_true[binary], y_prob[binary]
    return {
        "samples": len(y_true),
        "binary_samples": int(binary.sum()),
        "mse": float(np.mean((y_true - y_prob) ** 2)),
        "rmse": float(np.sqrt(np.mean((y_true - y_prob) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_prob))),
        "auc": auc_score(binary_true, binary_prob),
        "accuracy": float(np.mean((binary_prob >= 0.5) == (binary_true == 1.0))),
        "probability_source": "local_ariadne_oracle_checkpoint",
    }


def main() -> None:
    with (ROOT / "configs" / "config.yaml").open(encoding="utf8") as file:
        config = yaml.safe_load(file)
    with (PROCESSED / "graph.pkl").open("rb") as file:
        graph = pickle.load(file)
    checkpoint = torch.load(
        PROCESSED / "oracle_ckpt.pt", map_location="cpu", weights_only=False
    )
    with (PROCESSED / "train_sessions.pkl").open("rb") as file:
        train_samples = pickle.load(file)

    oracle = load_oracle(graph, checkpoint, config["oracle"]["mc_samples"])
    dag: nx.DiGraph = graph["nx_dag"]
    observed = {target for _, target, _ in train_samples}
    candidates = sorted(node for node in observed if dag.in_degree(node) > 0)
    rng = np.random.default_rng(config["seed"])
    targets = sorted(
        rng.choice(
            candidates,
            size=min(config["experiments"]["num_targets"], len(candidates)),
            replace=False,
        ).tolist()
    )
    planner_config = {"planner": config["planner"], "oracle": config["oracle"]}
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)

    trajectories = []
    for target in targets:
        goal_nodes = nx.ancestors(dag, target) | {target}
        closure_graph = dag.subgraph(goal_nodes).copy()
        planner = DAGPlanner(
            oracle, closure_graph, planner_config, edge_index, len(graph["node_ids"])
        )
        started = time.perf_counter()
        # Requiring the target and all ancestors is equivalent on a prerequisite
        # DAG, and gives LAO* an informative admissible lower bound.
        result = planner.solve_result(set(), goal_nodes)
        path = DAGPlanner._extract_path(frozenset(), frozenset(goal_nodes), result.policy)
        elapsed = time.perf_counter() - started
        is_valid = valid_path(closure_graph, path, target)
        assert result.converged and is_valid, (target, path)
        trajectories.append(
            {
                "target_node": target,
                "expected_total_cost": result.values[frozenset()],
                "path_length": len(path),
                "required_nodes": len(goal_nodes),
                "off_target_actions": len(set(path) - goal_nodes),
                "expanded_states": result.expanded_count,
                "iterations": result.iterations,
                "planning_seconds": elapsed,
                "converged": result.converged,
                "path_is_valid": is_valid,
                "path": json.dumps(path),
            }
        )

    metrics = validation_metrics(graph, checkpoint)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(OUTPUT / "oracle_valid_metrics.csv", index=False)
    pd.DataFrame(trajectories).to_csv(OUTPUT / "planner_trajectories.csv", index=False)
    summary = {
        "condition": "Ariadne + LAO*",
        "targets": targets,
        "mean_expected_total_cost": float(np.mean([r["expected_total_cost"] for r in trajectories])),
        "mean_path_length": float(np.mean([r["path_length"] for r in trajectories])),
        "mean_off_target_actions": float(np.mean([r["off_target_actions"] for r in trajectories])),
        "mean_expanded_states": float(np.mean([r["expanded_states"] for r in trajectories])),
        "total_planning_seconds": float(sum(r["planning_seconds"] for r in trajectories)),
        "all_paths_valid": all(r["path_is_valid"] for r in trajectories),
        "all_converged": all(r["converged"] for r in trajectories),
        "probability_source": "local_ariadne_oracle_checkpoint",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(pd.DataFrame(trajectories).drop(columns="path").to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
