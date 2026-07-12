"""Run DKT-derived Set Oracle + Greedy under the shared protocol."""

from __future__ import annotations

import csv
import json
import struct
import sys
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "dkt_set"
OUTPUT = ROOT / "results" / "dkt_set_greedy"
SEQUENCES_PATH = OUTPUT / "sequences.jsonl"
SCORED_PATH = OUTPUT / "scored_sequences.csv"
CONFIG_PATH = ARTIFACTS / "surrogate_config.json"
CHECKPOINT_PATH = ARTIFACTS / "surrogate_checkpoint.pt"
METRICS_PATH = ARTIFACTS / "surrogate_metrics.json"
EXPECTED_SURROGATE_CHECKPOINT_HASH = (
    "74ee76f29e852b77f3116a6840386342ec256088a245a871a67ff4f4142c012a"
)
sys.path.insert(0, str(ROOT))

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, SequenceRecord, write_jsonl
from src.oracle_core.dkt_set_oracle import DKTSetOracle
from src.planner_engine.baselines import GreedyPlanner


def _closure_graph(closure: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(closure["nodes"])
    graph.add_edges_from(tuple(edge) for edge in closure["edges"])
    return graph


def load_frozen_oracle() -> tuple[DKTSetOracle, dict]:
    actual = sha256_file(CHECKPOINT_PATH)
    if actual != EXPECTED_SURROGATE_CHECKPOINT_HASH:
        raise ValueError(
            "DKT-set Greedy requires the approved surrogate checkpoint; "
            f"expected {EXPECTED_SURROGATE_CHECKPOINT_HASH}, found {actual}"
        )
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    if not metrics.get("go"):
        raise ValueError("DKT-derived Set Oracle failed its go/no-go gate")
    oracle = DKTSetOracle.from_artifacts(
        config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH, device="cpu"
    )
    return oracle, config


def generate_records(
    manifest: dict, oracle: DKTSetOracle, config: dict
) -> list[SequenceRecord]:
    initial = set(manifest["initial_state"])
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
        internal_cost, sequence = planner.solve(set(initial), set(closure["nodes"]))
        if set(sequence) != set(closure["sequence_nodes"]):
            raise AssertionError("DKT-set Greedy did not cover the manifest closure")
        if not sequence or sequence[-1] != closure["target_node"]:
            raise AssertionError("DKT-set Greedy target must be the final action")
        records.append(SequenceRecord(
            method=Method.DKT_SET_GREEDY,
            target_node=closure["target_node"],
            run_id=0,
            sequence=sequence,
            internal_cost=internal_cost,
            metadata={
                "condition_name": "DKT-derived Set Oracle",
                "closure_hash": closure["closure_hash"],
                "manifest_hash": manifest_hash(manifest),
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
                "inference_backend": "cpu",
                "path_length": len(sequence),
            },
        ))
    return records


def _signature(records: list[SequenceRecord]) -> list[tuple]:
    return [
        (record.target_node, record.sequence, struct.pack("!d", float(record.internal_cost)))
        for record in records
    ]


def score_and_write(records: list[SequenceRecord]) -> None:
    scored = SequenceEvaluator.from_artifacts().score_records(records)
    if not all(result.valid for result in scored):
        raise AssertionError("Public evaluator rejected a DKT-set Greedy record")
    fields = ["method", "target_node", "run_id", "valid", "evaluation_cost",
              "optimal_cost", "normalized_regret", "sequence_hash", "invalid_reason"]
    with SCORED_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result.to_dict() for result in scored)


def main() -> None:
    manifest = load_manifest()
    first_oracle, first_config = load_frozen_oracle()
    second_oracle, second_config = load_frozen_oracle()
    if first_config != second_config:
        raise AssertionError("Independent DKT Oracle configs differ")
    first = generate_records(manifest, first_oracle, first_config)
    second = generate_records(manifest, second_oracle, second_config)
    if _signature(first) != _signature(second):
        raise AssertionError("Independent DKT-set Greedy plans are not bitwise identical")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEQUENCES_PATH, first)
    score_and_write(first)
    print(f"records={len(first)}")
    print(f"checkpoint_hash={EXPECTED_SURROGATE_CHECKPOINT_HASH}")
    print(f"sequences={SEQUENCES_PATH}")
    print(f"scored={SCORED_PATH}")


if __name__ == "__main__":
    main()
