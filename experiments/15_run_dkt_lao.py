"""Run the ECS32A DKT + LAO* baseline from the existing pyKT checkpoint."""

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
PYKT_ROOT = ROOT.parent / "pykt-toolkit"
DATASET = PYKT_ROOT / "data" / "ecs32a_ariadne"
MODEL_DIR = PYKT_ROOT / "examples" / "saved_model" / (
    "ecs32a_ariadne_dkt_qid_saved_model_42_0_0.2_200_0.001_0_0"
)
OUTPUT = ROOT / "results" / "dkt_lao"
sys.path.insert(0, str(PYKT_ROOT))
sys.path.insert(0, str(ROOT))

from pykt.models.init_model import init_model
from src.planner_engine.solver import DAGPlanner


class DKTOracle:
    def __init__(self, probabilities: dict[int, float], fallback: float) -> None:
        self.probabilities = probabilities
        self.fallback = float(fallback)

    def success_prob(self, node: int, mastered: frozenset[int]) -> float:
        return self.probabilities.get(node, self.fallback)

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


def parse_sequence(row: pd.Series) -> tuple[list[int], list[int], list[int]]:
    concepts = [int(value) for value in row["concepts"].split(",")]
    responses = [int(value) for value in row["responses"].split(",")]
    selected = [int(value) for value in row["selectmasks"].split(",")]
    length = next((index for index, value in enumerate(concepts) if value < 0), len(concepts))
    return concepts[:length], responses[:length], selected[:length]


def predict_sequences(model, rows: pd.DataFrame, idx_to_node: dict[int, int]):
    labels: list[float] = []
    probabilities: list[float] = []
    by_node: dict[int, list[float]] = {}
    device = next(model.parameters()).device

    with torch.no_grad():
        for _, row in rows.iterrows():
            concepts, responses, selected = parse_sequence(row)
            if len(concepts) < 2:
                continue
            q = torch.tensor(concepts[:-1], dtype=torch.long, device=device).unsqueeze(0)
            r = torch.tensor(responses[:-1], dtype=torch.long, device=device).unsqueeze(0)
            output = model(q, r)[0].cpu().numpy()
            for index, target_idx in enumerate(concepts[1:]):
                if target_idx not in idx_to_node:
                    continue
                node = idx_to_node[target_idx]
                probability = float(output[index, target_idx])
                by_node.setdefault(node, []).append(probability)
                if selected[index + 1] == 1:
                    labels.append(float(responses[index + 1]))
                    probabilities.append(probability)
    return labels, probabilities, by_node


def main() -> None:
    with (ROOT / "configs" / "config.yaml").open() as file:
        config = yaml.safe_load(file)
    with (PROCESSED / "graph.pkl").open("rb") as file:
        graph = pickle.load(file)
    with (PROCESSED / "train_sessions.pkl").open("rb") as file:
        train_samples = pickle.load(file)

    key_map = json.loads((DATASET / "keyid2idx.json").read_text(encoding="utf8"))
    idx_to_node = {int(index): int(node) for node, index in key_map["concepts"].items()}
    data_config = {"num_c": len(idx_to_node), "num_q": 141, "emb_path": ""}
    model = init_model("dkt", {"dropout": 0.2, "emb_size": 200}, data_config, "qid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "qid_model.ckpt", map_location=device))
    model.eval()

    sequences = pd.read_csv(DATASET / "train_valid_sequences.csv")
    train_rows = sequences[sequences["fold"] != 0]
    valid_rows = sequences[sequences["fold"] == 0]
    _, _, train_by_node = predict_sequences(model, train_rows, idx_to_node)
    valid_labels, valid_probabilities, _ = predict_sequences(model, valid_rows, idx_to_node)
    labels = np.array(valid_labels)
    probabilities = np.array(valid_probabilities)
    oracle_metrics = {
        "samples": len(labels),
        "binary_samples": len(labels),
        "mse": float(np.mean((labels - probabilities) ** 2)),
        "rmse": float(np.sqrt(np.mean((labels - probabilities) ** 2))),
        "mae": float(np.mean(np.abs(labels - probabilities))),
        "auc": auc_score(labels, probabilities),
        "accuracy": float(np.mean((probabilities >= 0.5) == (labels == 1.0))),
        "probability_source": "dkt_train_fold_predictions",
    }
    oracle = DKTOracle(
        {node: float(np.mean(values)) for node, values in train_by_node.items()},
        float(np.mean(probabilities)),
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
        planner = DAGPlanner(oracle, dag, planner_config, edge_index, num_nodes)
        started = time.perf_counter()
        result = planner.solve_result(set(), {target})
        start = frozenset()
        path = DAGPlanner._extract_path(start, frozenset({target}), result.policy)
        cost = result.values[start]
        elapsed = time.perf_counter() - started
        valid = is_valid_path(dag, path)
        assert result.converged and path and path[-1] == target and valid, (target, path)
        required_nodes = nx.ancestors(dag, target) | {target}
        trajectories.append({
            "target_node": target,
            "expected_total_cost": cost,
            "path_length": len(path),
            "required_nodes": len(required_nodes),
            "off_target_actions": len(set(path) - required_nodes),
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
        "condition": "DKT + LAO*",
        "targets": targets,
        "mean_expected_total_cost": float(np.mean([row["expected_total_cost"] for row in trajectories])),
        "mean_path_length": float(np.mean([row["path_length"] for row in trajectories])),
        "mean_off_target_actions": float(np.mean([row["off_target_actions"] for row in trajectories])),
        "mean_expanded_states": float(np.mean([row["expanded_states"] for row in trajectories])),
        "total_planning_seconds": float(sum(row["planning_seconds"] for row in trajectories)),
        "all_paths_valid": all(row["path_is_valid"] for row in trajectories),
        "all_converged": all(row["converged"] for row in trajectories),
        "probability_source": "dkt_train_fold_predictions",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    print(pd.DataFrame([oracle_metrics]).to_string(index=False))
    print(pd.DataFrame(trajectories).drop(columns="path").to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
