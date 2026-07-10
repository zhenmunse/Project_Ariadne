"""Evaluate a random valid topological ordering on ECS32A."""

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
OUTPUT = ROOT / "results" / "random_valid_ordering"
sys.path.insert(0, str(ROOT))

from src.oracle_core.dataset import get_dataloader
from src.oracle_core.model import MonotonicOracle


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


def validation_metrics(graph: dict, checkpoint: dict, model: MonotonicOracle) -> dict:
    with (PROCESSED / "valid_sessions.pkl").open("rb") as file:
        samples = pickle.load(file)
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


def random_order(graph: nx.DiGraph, target: int, rng: np.random.Generator) -> list[int]:
    required = nx.ancestors(graph, target) | {target}
    mastered: set[int] = set()
    order = []
    while mastered != required:
        actions = sorted(
            node
            for node in required - mastered
            if set(graph.predecessors(node)) <= mastered
        )
        if not actions:
            raise ValueError(f"No valid action remains for target {target}")
        choice = int(rng.choice(actions))
        order.append(choice)
        mastered.add(choice)
    return order


def action_cost(model, edge_index, num_nodes: int, node: int, mastered: set[int]) -> float:
    x = torch.zeros(num_nodes, 2)
    mask = torch.zeros(num_nodes)
    for mastered_node in mastered:
        mask[mastered_node] = 1.0
    probability, _, _ = model.predict_mc(
        x,
        edge_index,
        torch.tensor(node, dtype=torch.long),
        mask,
        mc_samples=20,
    )
    return 60.0 / max(float(probability.item()), 1e-12)


def main() -> None:
    with (ROOT / "configs" / "config.yaml").open(encoding="utf8") as file:
        config = yaml.safe_load(file)
    with (PROCESSED / "graph.pkl").open("rb") as file:
        graph = pickle.load(file)
    with (PROCESSED / "oracle_ckpt.pt").open("rb") as file:
        checkpoint = torch.load(file, map_location="cpu", weights_only=False)
    with (PROCESSED / "train_sessions.pkl").open("rb") as file:
        train_samples = pickle.load(file)

    if checkpoint["node_id_to_idx"] != graph["node_id_to_idx"]:
        raise ValueError("checkpoint and graph use different node mappings")
    model = MonotonicOracle(
        num_nodes=checkpoint["num_nodes"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

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
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    trajectories = []
    for target in targets:
        started = time.perf_counter()
        path = random_order(dag, target, rng)
        mastered: set[int] = set()
        total_cost = 0.0
        for node in path:
            total_cost += action_cost(model, edge_index, len(graph["node_ids"]), node, mastered)
            mastered.add(node)
        required = nx.ancestors(dag, target) | {target}
        assert path[-1] == target and set(path) == required
        trajectories.append(
            {
                "target_node": target,
                "expected_total_cost": total_cost,
                "path_length": len(path),
                "required_nodes": len(required),
                "off_target_actions": len(set(path) - required),
                "planning_seconds": time.perf_counter() - started,
                "path_is_valid": True,
                "path": json.dumps(path),
            }
        )

    metrics = validation_metrics(graph, checkpoint, model)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(OUTPUT / "oracle_valid_metrics.csv", index=False)
    pd.DataFrame(trajectories).to_csv(OUTPUT / "planner_trajectories.csv", index=False)
    summary = {
        "condition": "Random Valid Ordering",
        "targets": targets,
        "mean_expected_total_cost": float(np.mean([r["expected_total_cost"] for r in trajectories])),
        "mean_path_length": float(np.mean([r["path_length"] for r in trajectories])),
        "mean_off_target_actions": 0.0,
        "total_planning_seconds": float(sum(r["planning_seconds"] for r in trajectories)),
        "all_paths_valid": True,
        "evaluation_oracle": "local_ariadne_oracle_checkpoint",
        "random_seed": config["seed"],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(pd.DataFrame(trajectories).drop(columns="path").to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
