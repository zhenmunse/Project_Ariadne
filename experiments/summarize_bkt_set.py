"""Build the deterministic Task 12 summary from frozen artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "bkt_set"
RESULTS = ROOT / "results" / "bkt_set"
GREEDY_RESULTS = ROOT / "results" / "bkt_set_greedy"
LAO_RESULTS = ROOT / "results" / "bkt_set_lao"
OUTPUT_PATH = RESULTS / "task12_summary.json"
sys.path.insert(0, str(ROOT))

from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, read_jsonl


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _validated_records(
    path: Path,
    scored_path: Path,
    *,
    expected_method: Method,
    targets: set[int],
    expected_metadata: dict[str, object],
) -> tuple[list, list[dict[str, str]]]:
    records = read_jsonl(path)
    scored = _load_csv(scored_path)
    if len(records) != len(targets) or len(scored) != len(targets):
        raise ValueError(
            f"Expected exactly one {expected_method.value} record and score per target"
        )
    if {record.target_node for record in records} != targets:
        raise ValueError(f"{expected_method.value} target set mismatch")
    if {int(row["target_node"]) for row in scored} != targets:
        raise ValueError(f"{expected_method.value} scored target set mismatch")
    if any(record.method is not expected_method for record in records):
        raise ValueError(f"Unexpected method in {path}")
    if any(row["method"] != expected_method.value for row in scored):
        raise ValueError(f"Unexpected scored method in {scored_path}")
    if any(row["valid"] != "True" for row in scored):
        raise ValueError(f"Public evaluator rejected a record in {scored_path}")
    for record in records:
        for key, expected in expected_metadata.items():
            if record.metadata.get(key) != expected:
                raise ValueError(
                    f"{expected_method.value} metadata mismatch for {key}"
                )
    return records, scored


def build_summary() -> dict:
    manifest = load_manifest()
    config = _load_json(ARTIFACTS / "surrogate_config.json")
    metrics = _load_json(ARTIFACTS / "surrogate_metrics.json")
    targets = set(manifest["targets"])
    protocol_hash = manifest_hash(manifest)
    checkpoint_hash = sha256_file(ARTIFACTS / "surrogate_checkpoint.pt")
    evaluator_hash = sha256_file(ROOT / "experiments" / "common" / "evaluator.py")

    if metrics.get("config_hash") != sha256_file(
        ARTIFACTS / "surrogate_config.json"
    ):
        raise ValueError("Surrogate metrics/config hash mismatch")
    if metrics.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("Surrogate metrics/checkpoint hash mismatch")
    if config.get("evaluator_hash") != evaluator_hash:
        raise ValueError("Surrogate/public evaluator hash mismatch")

    expected_metadata = {
        "manifest_hash": protocol_hash,
        "evaluator_hash": evaluator_hash,
        "split_hash": config["split_hash"],
        "parameter_values_hash": config["parameter_values_hash"],
        "distillation_table_hash": config["tuple_collection_hash"],
        "surrogate_config_hash": sha256_file(
            ARTIFACTS / "surrogate_config.json"
        ),
        "surrogate_checkpoint_hash": checkpoint_hash,
        "oracle_state_dependence": True,
    }
    greedy_records, greedy_scored = _validated_records(
        GREEDY_RESULTS / "sequences.jsonl",
        GREEDY_RESULTS / "scored_sequences.csv",
        expected_method=Method.BKT_SET_GREEDY,
        targets=targets,
        expected_metadata=expected_metadata,
    )
    lao_records, lao_scored = _validated_records(
        LAO_RESULTS / "sequences.jsonl",
        LAO_RESULTS / "scored_sequences.csv",
        expected_method=Method.BKT_SET_LAO,
        targets=targets,
        expected_metadata=expected_metadata,
    )

    comparisons = _load_csv(RESULTS / "planner_comparison.csv")
    if len(comparisons) != len(targets):
        raise ValueError("Planner comparison must contain one row per target")
    if {int(row["target_node"]) for row in comparisons} != targets:
        raise ValueError("Planner comparison target set mismatch")
    if any(row["converged"] != "True" for row in comparisons):
        raise ValueError("A BKT-set LAO* run did not converge")
    lao_dp_max_gap = max(float(row["lao_dp_absolute_gap"]) for row in comparisons)
    max_lao_minus_greedy = max(
        float(record.internal_cost)
        - float(next(
            greedy.internal_cost
            for greedy in greedy_records
            if greedy.target_node == record.target_node
        ))
        for record in lao_records
    )
    if lao_dp_max_gap >= 1e-9:
        raise ValueError("LAO*/DP consistency gate failed")
    if max_lao_minus_greedy > 1e-9:
        raise ValueError("LAO* is worse than Greedy beyond tolerance")

    state = metrics["state_dependence"]
    status = "go" if bool(metrics.get("go")) and bool(state.get("go")) else "no-go"
    return {
        "schema_version": 1,
        "status": status,
        "condition_name": "BKT-derived Set Oracle",
        "manifest_hash": protocol_hash,
        "split_hash": config["split_hash"],
        "teacher_hash": config["parameter_values_hash"],
        "teacher_artifact_hash": config["bkt_parameter_artifact_hash"],
        "pooled_teacher_vector_hash": config["pooled_parameter_vector_hash"],
        "distillation_hash": config["tuple_collection_hash"],
        "surrogate_config_hash": sha256_file(
            ARTIFACTS / "surrogate_config.json"
        ),
        "surrogate_checkpoint_hash": checkpoint_hash,
        "max_state_effect": float(state["max_state_effect"]),
        "state_effect_threshold": float(state["minimum_required_effect"]),
        "greedy_valid_targets": sum(row["valid"] == "True" for row in greedy_scored),
        "lao_valid_targets": sum(row["valid"] == "True" for row in lao_scored),
        "lao_dp_max_gap": lao_dp_max_gap,
        "lao_minus_greedy_max_gap": max_lao_minus_greedy,
        "public_evaluator_hash": evaluator_hash,
        "greedy_sequences_hash": sha256_file(GREEDY_RESULTS / "sequences.jsonl"),
        "lao_sequences_hash": sha256_file(LAO_RESULTS / "sequences.jsonl"),
        "generation_command": "python experiments/summarize_bkt_set.py",
    }


def main() -> None:
    summary = build_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
