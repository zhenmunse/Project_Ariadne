"""Run DKT-derived Set Oracle + LAO* with exact-DP verification."""

from __future__ import annotations

import csv
import json
import struct
import sys
import time
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "dkt_set"
OUTPUT = ROOT / "results" / "dkt_set_lao"
SEQUENCES_PATH = OUTPUT / "sequences.jsonl"
SCORED_PATH = OUTPUT / "scored_sequences.csv"
COMPARISON_PATH = ROOT / "results" / "dkt_set" / "planner_comparison.csv"
GREEDY_PATH = ROOT / "results" / "dkt_set_greedy" / "sequences.jsonl"
CONFIG_PATH = ARTIFACTS / "surrogate_config.json"
CHECKPOINT_PATH = ARTIFACTS / "surrogate_checkpoint.pt"
METRICS_PATH = ARTIFACTS / "surrogate_metrics.json"
EXPECTED_SURROGATE_CHECKPOINT_HASH = (
    "74ee76f29e852b77f3116a6840386342ec256088a245a871a67ff4f4142c012a"
)
TOLERANCE = 1e-9
sys.path.insert(0, str(ROOT))

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.reference import load_single_run_reference
from experiments.common.schema import Method, SequenceRecord, write_jsonl
from src.oracle_core.dkt_set_oracle import DKTSetOracle
from src.planner_engine.solver import DAGPlanner, DAGPlannerDP


def _graph(closure: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(closure["nodes"])
    graph.add_edges_from(tuple(edge) for edge in closure["edges"])
    return graph


def load_frozen_oracle() -> tuple[DKTSetOracle, dict]:
    actual = sha256_file(CHECKPOINT_PATH)
    if actual != EXPECTED_SURROGATE_CHECKPOINT_HASH:
        raise ValueError(
            f"DKT-set LAO* expected checkpoint {EXPECTED_SURROGATE_CHECKPOINT_HASH}, found {actual}"
        )
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    if not metrics.get("go"):
        raise ValueError("DKT-derived Set Oracle failed its go/no-go gate")
    return DKTSetOracle.from_artifacts(
        config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH, device="cpu"
    ), config


def reference_metadata(config: dict, oracle: DKTSetOracle) -> dict[str, object]:
    return {
        "evaluator_hash": config["evaluator_hash"],
        "split_hash": config["split_hash"],
        "compression_config_hash": config["compression_config_hash"],
        "zero_observation_nodes_hash": config["zero_observation_nodes_hash"],
        "training_observed_nodes_hash": config["training_observed_nodes_hash"],
        "teacher_tensor_hash": config["teacher_tensor_hash"],
        "teacher_checkpoint_hash": config["teacher_checkpoint_hash"],
        "teacher_config_hash": config["teacher_config_hash"],
        "distillation_table_hash": config["tuple_collection_hash"],
        "surrogate_config_hash": oracle.config_hash,
        "surrogate_checkpoint_hash": oracle.checkpoint_hash,
        "oracle_state_dependence": True,
    }


def load_greedy_reference(
    manifest: dict, config: dict, oracle: DKTSetOracle
) -> dict[int, SequenceRecord]:
    return load_single_run_reference(
        GREEDY_PATH,
        expected_method=Method.DKT_SET_GREEDY,
        manifest=manifest,
        expected_metadata=reference_metadata(config, oracle),
        require_internal_cost=True,
    )


def generate_records(
    manifest: dict,
    oracle: DKTSetOracle,
    config: dict,
    greedy: dict[int, SequenceRecord],
) -> tuple[list[SequenceRecord], list[dict]]:
    initial = set(manifest["initial_state"])
    initial_frozen = frozenset(initial)
    planner_config = {"planner": {"base_cost": manifest["base_cost"], "heuristic": "sum"}}
    records, comparisons = [], []
    for closure in manifest["closures"]:
        target = closure["target_node"]
        goal = set(closure["nodes"])
        graph = _graph(closure)
        lao = DAGPlanner(oracle, graph, planner_config, None, len(oracle.node_order))
        started = time.perf_counter()
        result = lao.solve_result(set(initial), goal)
        elapsed = time.perf_counter() - started
        sequence = DAGPlanner._extract_path(initial_frozen, frozenset(goal), result.policy)
        lao_cost = float(result.values[initial_frozen])
        dp = DAGPlannerDP(oracle, graph, planner_config, None, len(oracle.node_order))
        dp_cost, dp_sequence = dp.solve(set(initial), goal)
        dp_cost = float(dp_cost)
        dp_gap = abs(lao_cost - dp_cost)
        greedy_cost = float(greedy[target].internal_cost)
        lao_minus_greedy = lao_cost - greedy_cost
        if not result.converged:
            raise AssertionError(f"DKT-set LAO* did not converge for target {target}")
        if dp_gap >= TOLERANCE:
            raise AssertionError(f"DKT-set LAO*/DP mismatch for target {target}: {dp_gap}")
        if lao_minus_greedy > TOLERANCE:
            raise AssertionError(f"DKT-set LAO* exceeds Greedy for target {target}")
        for name, path in (("LAO*", sequence), ("DP", dp_sequence)):
            if set(path) != set(closure["sequence_nodes"]) or not path or path[-1] != target:
                raise AssertionError(f"DKT-set {name} path is invalid for target {target}")
        metadata = {
            "condition_name": "DKT-derived Set Oracle",
            "closure_hash": closure["closure_hash"],
            "manifest_hash": manifest_hash(manifest),
            **reference_metadata(config, oracle),
            "inference_backend": "cpu",
            "heuristic": "sum_p_bar_1",
            "exact_dp_cost": dp_cost,
            "exact_dp_sequence": list(dp_sequence),
            "lao_dp_absolute_gap": dp_gap,
            "greedy_internal_cost": greedy_cost,
            "greedy_lao_internal_gap": lao_minus_greedy,
            "greedy_sequence": list(greedy[target].sequence),
            "sequence_matches_greedy": sequence == list(greedy[target].sequence),
            "expanded_states": result.expanded_count,
            "iterations": result.iterations,
            "converged": result.converged,
            "path_length": len(sequence),
        }
        records.append(SequenceRecord(
            method=Method.DKT_SET_LAO,
            target_node=target,
            run_id=0,
            sequence=sequence,
            internal_cost=lao_cost,
            metadata=metadata,
        ))
        comparisons.append({
            "target_node": target,
            "greedy_internal_cost": greedy_cost,
            "lao_internal_cost": lao_cost,
            "exact_dp_cost": dp_cost,
            "greedy_minus_lao": greedy_cost - lao_cost,
            "lao_dp_absolute_gap": dp_gap,
            "sequence_matches_greedy": sequence == list(greedy[target].sequence),
            "expanded_states": result.expanded_count,
            "iterations": result.iterations,
            "planning_seconds": elapsed,
            "converged": result.converged,
        })
    return records, comparisons


def signature(records: list[SequenceRecord]) -> list[tuple]:
    return [(
        record.target_node,
        record.sequence,
        struct.pack("!d", float(record.internal_cost)),
        struct.pack("!d", float(record.metadata["exact_dp_cost"])),
        record.metadata["expanded_states"],
        record.metadata["iterations"],
    ) for record in records]


def write_outputs(records: list[SequenceRecord], comparisons: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEQUENCES_PATH, records)
    scored = SequenceEvaluator.from_artifacts().score_records(records)
    if not all(value.valid for value in scored):
        raise AssertionError("Public evaluator rejected a DKT-set LAO* record")
    fields = ["method", "target_node", "run_id", "valid", "evaluation_cost",
              "optimal_cost", "normalized_regret", "sequence_hash", "invalid_reason"]
    with SCORED_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(value.to_dict() for value in scored)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)


def main() -> None:
    manifest = load_manifest()
    first_oracle, first_config = load_frozen_oracle()
    first_ref = load_greedy_reference(manifest, first_config, first_oracle)
    first, comparisons = generate_records(manifest, first_oracle, first_config, first_ref)
    second_oracle, second_config = load_frozen_oracle()
    second_ref = load_greedy_reference(manifest, second_config, second_oracle)
    second, _ = generate_records(manifest, second_oracle, second_config, second_ref)
    if first_config != second_config or signature(first) != signature(second):
        raise AssertionError("Independent DKT-set LAO* runs differ")
    write_outputs(first, comparisons)
    print(f"records={len(first)}")
    print(f"checkpoint_hash={EXPECTED_SURROGATE_CHECKPOINT_HASH}")
    print(f"max_lao_dp_gap={max(r.metadata['lao_dp_absolute_gap'] for r in first)}")
    print(f"max_lao_minus_greedy={max(r.metadata['greedy_lao_internal_gap'] for r in first)}")
    print(f"sequences={SEQUENCES_PATH}")
    print(f"scored={SCORED_PATH}")
    print(f"comparison={COMPARISON_PATH}")


if __name__ == "__main__":
    main()
