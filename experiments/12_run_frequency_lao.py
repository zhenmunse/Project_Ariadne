"""Run FrequencyOracle + LAO* as a state-independent negative control."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "results" / "frequency_lao"
SEQUENCES_PATH = OUTPUT / "sequences.jsonl"
METRICS_PATH = OUTPUT / "oracle_valid_metrics.csv"
GREEDY_SEQUENCES_PATH = ROOT / "results" / "frequency_greedy" / "sequences.jsonl"
INTERNAL_COST_TOLERANCE = 1e-9
sys.path.insert(0, str(ROOT))

from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.reference import load_single_run_reference
from experiments.common.schema import Method, SequenceRecord, write_jsonl
from src.planner_engine.baselines import FrequencyOracle
from src.planner_engine.solver import DAGPlanner


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


def frequency_metrics(
    oracle: FrequencyOracle,
    valid_samples: list,
    manifest: dict,
) -> dict:
    labels = np.asarray([label for _, _, label in valid_samples], dtype=float)
    probabilities = np.asarray(
        [oracle.success_prob(target, frozenset()) for _, target, _ in valid_samples],
        dtype=float,
    )
    binary = np.isin(labels, [0.0, 1.0])
    binary_labels = labels[binary]
    binary_probabilities = probabilities[binary]
    squared_error = (labels - probabilities) ** 2
    return {
        "samples": len(labels),
        "binary_samples": int(binary.sum()),
        "mse": float(np.mean(squared_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mae": float(np.mean(np.abs(labels - probabilities))),
        "auc": auc_score(binary_labels, binary_probabilities),
        "accuracy": float(
            np.mean((binary_probabilities >= 0.5) == (binary_labels == 1.0))
        ),
        "probability_source": "frequency_train_session_mean",
        "oracle_state_dependence": False,
        "global_mean": oracle.global_mean,
        "manifest_hash": manifest_hash(manifest),
        "dag_hash": manifest["artifact_hashes"]["dag"],
        "train_validation_split_hash": manifest["artifact_hashes"][
            "train_validation_split"
        ]["combined_hash"],
        "train_artifact_hash": sha256_file(PROCESSED / "train_sessions.pkl"),
        "validation_artifact_hash": sha256_file(PROCESSED / "valid_sessions.pkl"),
        "evaluator_hash": sha256_file(
            ROOT / "experiments" / "common" / "evaluator.py"
        ),
    }


def _closure_graph(closure: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(closure["nodes"])
    graph.add_edges_from(tuple(edge) for edge in closure["edges"])
    return graph


def generate_records(
    manifest: dict,
    oracle: FrequencyOracle,
    greedy_internal_costs: dict[int, float],
) -> list[SequenceRecord]:
    protocol_hash = manifest_hash(manifest)
    evaluator_hash = sha256_file(ROOT / "experiments" / "common" / "evaluator.py")
    train_hash = sha256_file(PROCESSED / "train_sessions.pkl")
    initial_state = set(manifest["initial_state"])
    initial_frozen = frozenset(initial_state)
    config = {
        "planner": {"base_cost": manifest["base_cost"], "heuristic": "sum"}
    }
    records = []
    for closure in manifest["closures"]:
        target = closure["target_node"]
        goal = set(closure["nodes"])
        planner = DAGPlanner(
            oracle=oracle,
            nx_graph=_closure_graph(closure),
            config=config,
            edge_index=None,
            num_nodes=oracle.num_nodes,
        )
        result = planner.solve_result(set(initial_state), goal)
        sequence = DAGPlanner._extract_path(
            initial_frozen, frozenset(goal), result.policy
        )
        internal_cost = float(result.values[initial_frozen])
        greedy_cost = greedy_internal_costs[target]
        internal_gap = abs(internal_cost - greedy_cost)
        if not result.converged:
            raise AssertionError(f"Frequency LAO* did not converge for target {target}")
        if internal_gap >= INTERNAL_COST_TOLERANCE:
            raise AssertionError(
                f"State-independent internal costs differ for target {target}: "
                f"LAO*={internal_cost}, Greedy={greedy_cost}, gap={internal_gap}"
            )
        if set(sequence) != set(closure["sequence_nodes"]):
            raise AssertionError(f"Frequency LAO* does not cover target {target}")

        records.append(
            SequenceRecord(
                method=Method.FREQUENCY_LAO,
                target_node=target,
                run_id=0,
                sequence=sequence,
                internal_cost=internal_cost,
                metadata={
                    "closure_hash": closure["closure_hash"],
                    "manifest_hash": protocol_hash,
                    "evaluator_hash": evaluator_hash,
                    "frequency_train_artifact_hash": train_hash,
                    "frequency_global_mean": oracle.global_mean,
                    "oracle_state_dependence": False,
                    "negative_control": True,
                    "greedy_internal_cost": greedy_cost,
                    "greedy_lao_internal_gap": internal_gap,
                    "expanded_states": result.expanded_count,
                    "iterations": result.iterations,
                    "converged": result.converged,
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
            record.metadata["expanded_states"],
            record.metadata["iterations"],
        )
        for record in records
    ]


def load_greedy_internal_costs(manifest: dict) -> dict[int, float]:
    """Load a strictly provenance-matched Frequency Greedy reference."""
    expected_train_hash = sha256_file(PROCESSED / "train_sessions.pkl")
    expected_evaluator_hash = sha256_file(
        ROOT / "experiments" / "common" / "evaluator.py"
    )
    records = load_single_run_reference(
        GREEDY_SEQUENCES_PATH,
        expected_method=Method.FREQUENCY_GREEDY,
        manifest=manifest,
        expected_metadata={
            "frequency_train_artifact_hash": expected_train_hash,
            "evaluator_hash": expected_evaluator_hash,
        },
        require_internal_cost=True,
    )
    return {
        target: float(record.internal_cost)
        for target, record in records.items()
    }


def main() -> None:
    manifest = load_manifest()
    with (PROCESSED / "graph.pkl").open("rb") as file:
        graph = pickle.load(file)
    with (PROCESSED / "train_sessions.pkl").open("rb") as file:
        train_samples = pickle.load(file)
    with (PROCESSED / "valid_sessions.pkl").open("rb") as file:
        valid_samples = pickle.load(file)

    greedy_internal_costs = load_greedy_internal_costs(manifest)

    first_oracle = FrequencyOracle(
        train_samples,
        len(graph["node_ids"]),
        graph["node_id_to_idx"],
        t_base=manifest["base_cost"],
    )
    second_oracle = FrequencyOracle(
        train_samples,
        len(graph["node_ids"]),
        graph["node_id_to_idx"],
        t_base=manifest["base_cost"],
    )
    first = generate_records(manifest, first_oracle, greedy_internal_costs)
    second = generate_records(manifest, second_oracle, greedy_internal_costs)
    if _deterministic_signature(first) != _deterministic_signature(second):
        raise AssertionError("Independent Frequency LAO* runs were not deterministic")

    metrics = frequency_metrics(first_oracle, valid_samples, manifest)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEQUENCES_PATH, first)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False, lineterminator="\n")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"records={len(first)}")
    print(f"max_greedy_lao_internal_gap={max(r.metadata['greedy_lao_internal_gap'] for r in first)}")
    print(f"sequences={SEQUENCES_PATH}")
    print(f"metrics={METRICS_PATH}")


if __name__ == "__main__":
    main()
