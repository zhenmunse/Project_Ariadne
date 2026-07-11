"""Run Ariadne + Greedy under the deterministic shared protocol."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "results" / "ariadne_greedy"
SEQUENCES_PATH = OUTPUT / "sequences.jsonl"
METRICS_PATH = OUTPUT / "oracle_valid_metrics.csv"
sys.path.insert(0, str(ROOT))

from experiments.common.frozen_oracle import FrozenMonotonicOracle
from experiments.common.manifest import load_manifest
from experiments.common.schema import Method, SequenceRecord, write_jsonl
from src.oracle_core.dataset import get_dataloader
from src.planner_engine.baselines import GreedyPlanner


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


def _metrics(labels: np.ndarray, probabilities: np.ndarray, prefix: str) -> dict:
    binary = np.isin(labels, [0.0, 1.0])
    binary_labels = labels[binary]
    binary_probabilities = probabilities[binary]
    squared_error = (labels - probabilities) ** 2
    return {
        f"{prefix}_mse": float(np.mean(squared_error)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(squared_error))),
        f"{prefix}_mae": float(np.mean(np.abs(labels - probabilities))),
        f"{prefix}_auc": auc_score(binary_labels, binary_probabilities),
        f"{prefix}_accuracy": float(
            np.mean((binary_probabilities >= 0.5) == (binary_labels == 1.0))
        ),
    }


def validation_metrics(
    oracle: FrozenMonotonicOracle,
    graph: dict,
) -> dict:
    """Report full-feature and zero-history planning-mode validation metrics."""
    validation_path = PROCESSED / "valid_sessions.pkl"
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Missing {validation_path}; run experiments/09_prepare_oracle_data.py"
        )
    with validation_path.open("rb") as file:
        samples = pickle.load(file)

    if graph["node_id_to_idx"] != oracle.node_id_to_idx:
        raise ValueError("validation graph and frozen checkpoint use different node mappings")

    loader = get_dataloader(
        samples,
        graph["node_id_to_idx"],
        len(graph["node_ids"]),
        batch_size=128,
        shuffle=False,
    )
    full_probabilities = []
    planning_probabilities = []
    labels = []
    oracle.model.eval()
    with torch.no_grad():
        for x, target, mastery_mask, label in loader:
            x = x.to(oracle.device)
            target = target.to(oracle.device)
            mastery_mask = mastery_mask.to(oracle.device)
            full_probability, _ = oracle.model.forward_batch(
                x, oracle.edge_index, target, mastery_mask
            )
            planning_probability, _ = oracle.model.forward_batch(
                torch.zeros_like(x), oracle.edge_index, target, mastery_mask
            )
            full_probabilities.append(full_probability.cpu())
            planning_probabilities.append(planning_probability.cpu())
            labels.append(label)

    y_true = torch.cat(labels).numpy()
    full = torch.cat(full_probabilities).numpy()
    planning = torch.cat(planning_probabilities).numpy()
    binary_samples = int(np.isin(y_true, [0.0, 1.0]).sum())
    return {
        "samples": len(y_true),
        "binary_samples": binary_samples,
        **_metrics(y_true, full, "full_feature"),
        **_metrics(y_true, planning, "planning_mode"),
        "planning_mode_x": "all_zero",
        "inference_backend": "cpu",
    }


def _closure_graph(closure: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(closure["nodes"])
    graph.add_edges_from(tuple(edge) for edge in closure["edges"])
    return graph


def generate_records(
    manifest: dict,
    oracle: FrozenMonotonicOracle,
) -> list[SequenceRecord]:
    records = []
    initial_state = set(manifest["initial_state"])
    planner_config = {"planner": {"base_cost": manifest["base_cost"]}}
    for closure in manifest["closures"]:
        graph = _closure_graph(closure)
        planner = GreedyPlanner(
            oracle=oracle,
            nx_graph=graph,
            config=planner_config,
            edge_index=oracle.edge_index,
            num_nodes=oracle.model.num_nodes,
        )
        started = time.perf_counter()
        internal_cost, sequence = planner.solve(
            set(initial_state), set(closure["nodes"])
        )
        planning_seconds = time.perf_counter() - started
        if set(sequence) != set(closure["sequence_nodes"]):
            raise RuntimeError(
                f"Greedy sequence does not cover target {closure['target_node']} closure"
            )
        records.append(
            SequenceRecord(
                method=Method.ARIADNE_GREEDY,
                target_node=closure["target_node"],
                run_id=0,
                sequence=sequence,
                internal_cost=internal_cost,
                metadata={
                    "closure_hash": closure["closure_hash"],
                    "path_length": len(sequence),
                    "planning_seconds": planning_seconds,
                    "oracle_state_dependence": True,
                    "inference_backend": "cpu",
                },
            )
        )
    return records


def _deterministic_signature(records: list[SequenceRecord]) -> list[tuple]:
    return [
        (record.target_node, record.sequence, record.internal_cost)
        for record in records
    ]


def main() -> None:
    manifest = load_manifest()
    first_oracle = FrozenMonotonicOracle.from_artifacts(
        base_cost=manifest["base_cost"], device="cpu"
    )
    second_oracle = FrozenMonotonicOracle.from_artifacts(
        base_cost=manifest["base_cost"], device="cpu"
    )
    first_records = generate_records(manifest, first_oracle)
    second_records = generate_records(manifest, second_oracle)
    if _deterministic_signature(first_records) != _deterministic_signature(second_records):
        raise AssertionError("Independent Ariadne + Greedy runs were not deterministic")

    with (PROCESSED / "graph.pkl").open("rb") as file:
        graph = pickle.load(file)
    metrics = validation_metrics(first_oracle, graph)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEQUENCES_PATH, first_records)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"sequences={SEQUENCES_PATH}")
    print(f"metrics={METRICS_PATH}")
    print(json.dumps({"targets": manifest["targets"], "records": len(first_records)}))


if __name__ == "__main__":
    main()
