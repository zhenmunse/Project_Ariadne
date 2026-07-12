"""Run BKT-derived Set Oracle + LAO* under the shared protocol."""

from __future__ import annotations

import csv
import json
import struct
import sys
import time
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "bkt_set"
OUTPUT = ROOT / "results" / "bkt_set_lao"
SEQUENCES_PATH = OUTPUT / "sequences.jsonl"
SCORED_PATH = OUTPUT / "scored_sequences.csv"
COMPARISON_PATH = ROOT / "results" / "bkt_set" / "planner_comparison.csv"
GREEDY_SEQUENCES_PATH = ROOT / "results" / "bkt_set_greedy" / "sequences.jsonl"
CONFIG_PATH = ARTIFACTS / "surrogate_config.json"
CHECKPOINT_PATH = ARTIFACTS / "surrogate_checkpoint.pt"
METRICS_PATH = ARTIFACTS / "surrogate_metrics.json"
EXPECTED_SURROGATE_CHECKPOINT_HASH = (
    "d285d7666e658c8f10637deffc986e408e5a15bbb2d3dcff50856cff7250d4f4"
)
INTERNAL_COST_TOLERANCE = 1e-9
DP_TOLERANCE = 1e-9
sys.path.insert(0, str(ROOT))

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.reference import load_single_run_reference
from experiments.common.schema import Method, SequenceRecord, write_jsonl
from src.oracle_core.bkt_set_oracle import BKTSetOracle
from src.planner_engine.solver import DAGPlanner, DAGPlannerDP


def _closure_graph(closure: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(closure["nodes"])
    graph.add_edges_from(tuple(edge) for edge in closure["edges"])
    return graph


def load_frozen_oracle() -> tuple[BKTSetOracle, dict]:
    """Load, never train, the checkpoint already frozen by Commit 12-3."""
    actual_hash = sha256_file(CHECKPOINT_PATH)
    if actual_hash != EXPECTED_SURROGATE_CHECKPOINT_HASH:
        raise ValueError(
            "BKT-set LAO* requires the approved surrogate checkpoint; "
            f"expected {EXPECTED_SURROGATE_CHECKPOINT_HASH}, found {actual_hash}"
        )
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    if not metrics.get("go"):
        raise ValueError("BKT-derived Set Oracle did not pass its go/no-go gate")
    oracle = BKTSetOracle.from_artifacts(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )
    if oracle.checkpoint_hash != EXPECTED_SURROGATE_CHECKPOINT_HASH:
        raise AssertionError("Loaded Oracle checkpoint identity changed")
    return oracle, config


def _reference_metadata(config: dict, oracle: BKTSetOracle) -> dict[str, object]:
    return {
        "evaluator_hash": config["evaluator_hash"],
        "split_hash": config["split_hash"],
        "compression_config_hash": config["compression_config_hash"],
        "parameter_values_hash": config["parameter_values_hash"],
        "bkt_parameter_artifact_hash": config["bkt_parameter_artifact_hash"],
        "pooled_parameter_vector_hash": config["pooled_parameter_vector_hash"],
        "pooled_parameter_artifact_hash": config["pooled_parameter_artifact_hash"],
        "pooled_backoff_nodes_hash": config["pooled_backoff_nodes_hash"],
        "distillation_table_hash": config["tuple_collection_hash"],
        "surrogate_config_hash": oracle.config_hash,
        "surrogate_checkpoint_hash": oracle.checkpoint_hash,
        "oracle_state_dependence": True,
    }


def load_greedy_reference(
    manifest: dict,
    config: dict,
    oracle: BKTSetOracle,
) -> dict[int, SequenceRecord]:
    """Load one provenance-identical BKT-set Greedy record per target."""
    return load_single_run_reference(
        GREEDY_SEQUENCES_PATH,
        expected_method=Method.BKT_SET_GREEDY,
        manifest=manifest,
        expected_metadata=_reference_metadata(config, oracle),
        require_internal_cost=True,
    )


def generate_records(
    manifest: dict,
    oracle: BKTSetOracle,
    surrogate_config: dict,
    greedy_reference: dict[int, SequenceRecord],
) -> tuple[list[SequenceRecord], list[dict]]:
    protocol_hash = manifest_hash(manifest)
    evaluator_hash = sha256_file(ROOT / "experiments" / "common" / "evaluator.py")
    initial_state = set(manifest["initial_state"])
    initial_frozen = frozenset(initial_state)
    planner_config = {
        "planner": {"base_cost": manifest["base_cost"], "heuristic": "sum"}
    }
    records = []
    comparisons = []

    for closure in manifest["closures"]:
        target = closure["target_node"]
        graph = _closure_graph(closure)
        goal = set(closure["nodes"])
        lao = DAGPlanner(
            oracle=oracle,
            nx_graph=graph,
            config=planner_config,
            edge_index=None,
            num_nodes=len(oracle.node_order),
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
            edge_index=None,
            num_nodes=len(oracle.node_order),
        )
        dp_cost, dp_sequence = dp.solve(set(initial_state), goal)
        dp_cost = float(dp_cost)
        dp_gap = abs(lao_cost - dp_cost)
        greedy_record = greedy_reference[target]
        greedy_cost = float(greedy_record.internal_cost)
        greedy_lao_gap = lao_cost - greedy_cost

        if not lao_result.converged:
            raise AssertionError(f"BKT-set LAO* did not converge for target {target}")
        if dp_gap >= DP_TOLERANCE:
            raise AssertionError(
                f"BKT-set LAO* and exact DP differ for target {target}: "
                f"LAO*={lao_cost}, DP={dp_cost}, gap={dp_gap}"
            )
        if lao_cost > greedy_cost + INTERNAL_COST_TOLERANCE:
            raise AssertionError(
                f"BKT-set LAO* is worse than Greedy for target {target}: "
                f"LAO*={lao_cost}, Greedy={greedy_cost}"
            )
        if set(sequence) != set(closure["sequence_nodes"]):
            raise AssertionError(f"BKT-set LAO* does not cover target {target}")
        if not sequence or sequence[-1] != target:
            raise AssertionError("BKT-set LAO* target must be the final action")
        if set(dp_sequence) != set(closure["sequence_nodes"]):
            raise AssertionError(f"Exact DP does not cover target {target}")
        if not dp_sequence or dp_sequence[-1] != target:
            raise AssertionError("Exact DP target must be the final action")

        metadata = {
            "condition_name": "BKT-derived Set Oracle",
            "closure_hash": closure["closure_hash"],
            "manifest_hash": protocol_hash,
            "evaluator_hash": evaluator_hash,
            "split_hash": surrogate_config["split_hash"],
            "compression_config_hash": surrogate_config["compression_config_hash"],
            "parameter_values_hash": surrogate_config["parameter_values_hash"],
            "bkt_parameter_artifact_hash": surrogate_config[
                "bkt_parameter_artifact_hash"
            ],
            "pooled_parameter_vector_hash": surrogate_config[
                "pooled_parameter_vector_hash"
            ],
            "pooled_parameter_artifact_hash": surrogate_config[
                "pooled_parameter_artifact_hash"
            ],
            "pooled_backoff_nodes_hash": surrogate_config[
                "pooled_backoff_nodes_hash"
            ],
            "distillation_table_hash": surrogate_config["tuple_collection_hash"],
            "surrogate_config_hash": oracle.config_hash,
            "surrogate_checkpoint_hash": oracle.checkpoint_hash,
            "oracle_state_dependence": True,
            "inference_backend": "cpu",
            "heuristic": "sum_p_bar_1",
            "exact_dp_cost": dp_cost,
            "exact_dp_sequence": list(dp_sequence),
            "lao_dp_absolute_gap": dp_gap,
            "greedy_internal_cost": greedy_cost,
            "greedy_lao_internal_gap": greedy_lao_gap,
            "greedy_sequence": list(greedy_record.sequence),
            "sequence_matches_greedy": sequence == list(greedy_record.sequence),
            "expanded_states": lao_result.expanded_count,
            "iterations": lao_result.iterations,
            "converged": lao_result.converged,
            "path_length": len(sequence),
        }
        records.append(
            SequenceRecord(
                method=Method.BKT_SET_LAO,
                target_node=target,
                run_id=0,
                sequence=sequence,
                internal_cost=lao_cost,
                metadata=metadata,
            )
        )
        comparisons.append(
            {
                "target_node": target,
                "greedy_internal_cost": greedy_cost,
                "lao_internal_cost": lao_cost,
                "exact_dp_cost": dp_cost,
                "greedy_minus_lao": greedy_cost - lao_cost,
                "lao_dp_absolute_gap": dp_gap,
                "sequence_matches_greedy": sequence == list(greedy_record.sequence),
                "expanded_states": lao_result.expanded_count,
                "iterations": lao_result.iterations,
                "planning_seconds": planning_seconds,
                "converged": lao_result.converged,
            }
        )
    return records, comparisons


def _signature(records: list[SequenceRecord]) -> list[tuple]:
    return [
        (
            record.target_node,
            record.sequence,
            struct.pack("!d", float(record.internal_cost)),
            struct.pack("!d", float(record.metadata["exact_dp_cost"])),
            record.metadata["expanded_states"],
            record.metadata["iterations"],
        )
        for record in records
    ]


def score_and_write(records: list[SequenceRecord], output_path: Path) -> None:
    evaluator = SequenceEvaluator.from_artifacts()
    scored = evaluator.score_records(records)
    if not all(result.valid for result in scored):
        invalid = [result.to_dict() for result in scored if not result.valid]
        raise AssertionError(f"Public evaluator rejected BKT-set LAO*: {invalid}")
    fieldnames = [
        "method",
        "target_node",
        "run_id",
        "valid",
        "evaluation_cost",
        "optimal_cost",
        "normalized_regret",
        "sequence_hash",
        "invalid_reason",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result.to_dict() for result in scored)


def write_comparison(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest = load_manifest()
    first_oracle, first_config = load_frozen_oracle()
    greedy_reference = load_greedy_reference(manifest, first_config, first_oracle)
    first, comparisons = generate_records(
        manifest, first_oracle, first_config, greedy_reference
    )

    second_oracle, second_config = load_frozen_oracle()
    if first_config != second_config:
        raise AssertionError("Independent Oracle loads returned different configs")
    second_reference = load_greedy_reference(manifest, second_config, second_oracle)
    second, _ = generate_records(
        manifest, second_oracle, second_config, second_reference
    )
    if _signature(first) != _signature(second):
        raise AssertionError(
            "Independent BKT-derived Set Oracle LAO* runs were not bitwise deterministic"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEQUENCES_PATH, first)
    score_and_write(first, SCORED_PATH)
    write_comparison(comparisons, COMPARISON_PATH)
    print(f"records={len(first)}")
    print(f"checkpoint_hash={EXPECTED_SURROGATE_CHECKPOINT_HASH}")
    print(
        "max_lao_dp_gap="
        f"{max(record.metadata['lao_dp_absolute_gap'] for record in first)}"
    )
    print(
        "max_lao_minus_greedy="
        f"{max(record.metadata['greedy_lao_internal_gap'] for record in first)}"
    )
    print(f"sequences={SEQUENCES_PATH}")
    print(f"scored={SCORED_PATH}")
    print(f"comparison={COMPARISON_PATH}")


if __name__ == "__main__":
    main()
