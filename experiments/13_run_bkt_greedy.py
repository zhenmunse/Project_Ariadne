"""Run BKT-derived Set Oracle + Greedy under the shared protocol."""

from __future__ import annotations

import csv
import json
import struct
import sys
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "bkt_set"
OUTPUT = ROOT / "results" / "bkt_set_greedy"
SEQUENCES_PATH = OUTPUT / "sequences.jsonl"
SCORED_PATH = OUTPUT / "scored_sequences.csv"
CONFIG_PATH = ARTIFACTS / "surrogate_config.json"
CHECKPOINT_PATH = ARTIFACTS / "surrogate_checkpoint.pt"
METRICS_PATH = ARTIFACTS / "surrogate_metrics.json"
EXPECTED_SURROGATE_CHECKPOINT_HASH = (
    "b00a8184babd0280f979af41d1403c7c0ea0fe4b4bb70c05c71be3fb5ccff920"
)
sys.path.insert(0, str(ROOT))

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, SequenceRecord, write_jsonl
from src.oracle_core.bkt_set_oracle import BKTSetOracle
from src.planner_engine.baselines import GreedyPlanner


def _closure_graph(closure: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(closure["nodes"])
    graph.add_edges_from(tuple(edge) for edge in closure["edges"])
    return graph


def load_frozen_oracle() -> tuple[BKTSetOracle, dict]:
    """Load, never train, the exact checkpoint approved in Commit 12-3."""
    actual_hash = sha256_file(CHECKPOINT_PATH)
    if actual_hash != EXPECTED_SURROGATE_CHECKPOINT_HASH:
        raise ValueError(
            "BKT-set Greedy requires the approved surrogate checkpoint; "
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


def generate_records(
    manifest: dict,
    oracle: BKTSetOracle,
    surrogate_config: dict,
) -> list[SequenceRecord]:
    protocol_hash = manifest_hash(manifest)
    evaluator_hash = sha256_file(ROOT / "experiments" / "common" / "evaluator.py")
    initial_state = set(manifest["initial_state"])
    planner_config = {"planner": {"base_cost": manifest["base_cost"]}}
    records = []
    for closure in manifest["closures"]:
        planner = GreedyPlanner(
            oracle=oracle,
            nx_graph=_closure_graph(closure),
            config=planner_config,
            edge_index=None,
            num_nodes=len(oracle.node_order),
        )
        internal_cost, sequence = planner.solve(
            set(initial_state), set(closure["nodes"])
        )
        if set(sequence) != set(closure["sequence_nodes"]):
            raise AssertionError(
                f"BKT-set Greedy did not cover target {closure['target_node']} closure"
            )
        if not sequence or sequence[-1] != closure["target_node"]:
            raise AssertionError("BKT-set Greedy target must be the final action")
        records.append(
            SequenceRecord(
                method=Method.BKT_SET_GREEDY,
                target_node=closure["target_node"],
                run_id=0,
                sequence=sequence,
                internal_cost=internal_cost,
                metadata={
                    "condition_name": "BKT-derived Set Oracle",
                    "closure_hash": closure["closure_hash"],
                    "manifest_hash": protocol_hash,
                    "evaluator_hash": evaluator_hash,
                    "split_hash": surrogate_config["split_hash"],
                    "compression_config_hash": surrogate_config[
                        "compression_config_hash"
                    ],
                    "parameter_values_hash": surrogate_config[
                        "parameter_values_hash"
                    ],
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
                    "distillation_table_hash": surrogate_config[
                        "tuple_collection_hash"
                    ],
                    "surrogate_config_hash": oracle.config_hash,
                    "surrogate_checkpoint_hash": oracle.checkpoint_hash,
                    "oracle_state_dependence": True,
                    "inference_backend": "cpu",
                    "path_length": len(sequence),
                },
            )
        )
    return records


def _signature(records: list[SequenceRecord]) -> list[tuple[int, tuple[int, ...], bytes]]:
    return [
        (
            record.target_node,
            record.sequence,
            struct.pack("!d", float(record.internal_cost)),
        )
        for record in records
    ]


def score_and_write(records: list[SequenceRecord], output_path: Path) -> None:
    """Run the method output through the independent frozen evaluator."""
    evaluator = SequenceEvaluator.from_artifacts()
    scored = evaluator.score_records(records)
    if not all(result.valid for result in scored):
        invalid = [result.to_dict() for result in scored if not result.valid]
        raise AssertionError(f"Public evaluator rejected BKT-set records: {invalid}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    manifest = load_manifest()
    first_oracle, first_config = load_frozen_oracle()
    second_oracle, second_config = load_frozen_oracle()
    if first_config != second_config:
        raise AssertionError("Independent Oracle loads returned different configs")
    first = generate_records(manifest, first_oracle, first_config)
    second = generate_records(manifest, second_oracle, second_config)
    if _signature(first) != _signature(second):
        raise AssertionError(
            "Independent BKT-derived Set Oracle Greedy runs were not bitwise deterministic"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEQUENCES_PATH, first)
    score_and_write(first, SCORED_PATH)
    print(f"records={len(first)}")
    print(f"checkpoint_hash={EXPECTED_SURROGATE_CHECKPOINT_HASH}")
    print(f"sequences={SEQUENCES_PATH}")
    print(f"scored={SCORED_PATH}")


if __name__ == "__main__":
    main()

