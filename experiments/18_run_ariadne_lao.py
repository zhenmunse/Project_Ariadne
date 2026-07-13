"""Run Ariadne + LAO* under the deterministic shared protocol."""

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
OUTPUT = ROOT / "results" / "ariadne_lao"
SEQUENCES_PATH = OUTPUT / "sequences.jsonl"
METRICS_PATH = OUTPUT / "oracle_valid_metrics.csv"
sys.path.insert(0, str(ROOT))

from experiments.common.frozen_oracle import FrozenMonotonicOracle
from experiments.common.manifest import (
    load_manifest,
    manifest_hash,
    sha256_file,
)
from experiments.common.schema import Method, SequenceRecord, write_jsonl
from src.oracle_core.dataset import get_dataloader
from src.planner_engine.solver import DAGPlanner, DAGPlannerDP


DP_TOLERANCE = 1e-9


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


def validation_metrics(oracle: FrozenMonotonicOracle, graph: dict) -> dict:
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
    full_probabilities, planning_probabilities, labels = [], [], []
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
    return {
        "samples": len(y_true),
        "binary_samples": int(np.isin(y_true, [0.0, 1.0]).sum()),
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
    initial_frozen = frozenset(initial_state)
    protocol_hash = manifest_hash(manifest)
    evaluator_hash = sha256_file(ROOT / "experiments" / "common" / "evaluator.py")
    planner_config = {
        "planner": {"base_cost": manifest["base_cost"], "heuristic": "sum"}
    }

    for closure in manifest["closures"]:
        graph = _closure_graph(closure)
        goal = set(closure["nodes"])
        lao = DAGPlanner(
            oracle=oracle,
            nx_graph=graph,
            config=planner_config,
            edge_index=oracle.edge_index,
            num_nodes=oracle.model.num_nodes,
        )
        started = time.perf_counter()
        lao_result = lao.solve_result(set(initial_state), goal)
        planning_seconds = time.perf_counter() - started
        sequence = DAGPlanner._extract_path(
            initial_frozen, frozenset(goal), lao_result.policy
        )
        lao_cost = float(lao_result.values[initial_frozen])

        dp = DAGPlannerDP(
            oracle=oracle,
            nx_graph=graph,
            config=planner_config,
            edge_index=oracle.edge_index,
            num_nodes=oracle.model.num_nodes,
        )
        dp_cost, dp_sequence = dp.solve(set(initial_state), goal)
        gap = abs(lao_cost - dp_cost)
        if not lao_result.converged:
            raise AssertionError(f"LAO* did not converge for target {closure['target_node']}")
        if gap >= DP_TOLERANCE:
            raise AssertionError(
                f"LAO* and DP differ for target {closure['target_node']}: "
                f"LAO*={lao_cost}, DP={dp_cost}, gap={gap}"
            )
        if set(sequence) != set(closure["sequence_nodes"]):
            raise AssertionError(
                f"LAO* sequence does not cover target {closure['target_node']} closure"
            )

        records.append(
            SequenceRecord(
                method=Method.ARIADNE_LAO,
                target_node=closure["target_node"],
                run_id=0,
                sequence=sequence,
                internal_cost=lao_cost,
                metadata={
                    "closure_hash": closure["closure_hash"],
                    "manifest_hash": protocol_hash,
                    "evaluator_hash": evaluator_hash,
                    "oracle_checkpoint_hash": manifest["artifact_hashes"][
                        "oracle_checkpoint"
                    ],
                    "exact_dp_cost": float(dp_cost),
                    "exact_dp_sequence": list(dp_sequence),
                    "lao_dp_absolute_gap": gap,
                    "expanded_states": lao_result.expanded_count,
                    "iterations": lao_result.iterations,
                    "planning_seconds": planning_seconds,
                    "converged": lao_result.converged,
                    "heuristic": "sum_p_bar_1",
                    "inference_backend": "cpu",
                },
            )
        )
    return records


def _deterministic_signature(records: list[SequenceRecord]) -> list[tuple]:
    return [
        (
            record.target_node,
            record.sequence,
            record.internal_cost,
            record.metadata["exact_dp_cost"],
            record.metadata["expanded_states"],
            record.metadata["iterations"],
        )
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
        raise AssertionError("Independent Ariadne + LAO* runs were not deterministic")

    with (PROCESSED / "graph.pkl").open("rb") as file:
        graph = pickle.load(file)
    metrics = validation_metrics(first_oracle, graph)
    metrics.update(
        {
            "manifest_hash": manifest_hash(manifest),
            "dag_hash": manifest["artifact_hashes"]["dag"],
            "oracle_checkpoint_hash": manifest["artifact_hashes"][
                "oracle_checkpoint"
            ],
            "train_validation_split_hash": manifest["artifact_hashes"][
                "train_validation_split"
            ]["combined_hash"],
            "validation_artifact_hash": sha256_file(
                PROCESSED / "valid_sessions.pkl"
            ),
            "evaluator_hash": sha256_file(
                ROOT / "experiments" / "common" / "evaluator.py"
            ),
        }
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEQUENCES_PATH, first_records)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"sequences={SEQUENCES_PATH}")
    print(f"metrics={METRICS_PATH}")
    print(json.dumps({"targets": manifest["targets"], "records": len(first_records)}))


if __name__ == "__main__":
    main()
