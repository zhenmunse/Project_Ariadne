"""Aggregate every approved standard condition through the public evaluator."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final"
sys.path.insert(0, str(ROOT))

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, SequenceRecord, read_jsonl, write_jsonl


CONDITIONS = (
    (Method.ARIADNE_GREEDY, ROOT / "results/ariadne_greedy/sequences.jsonl", 1),
    (Method.ARIADNE_LAO, ROOT / "results/ariadne_lao/sequences.jsonl", 1),
    (Method.FREQUENCY_GREEDY, ROOT / "results/frequency_greedy/sequences.jsonl", 1),
    (Method.FREQUENCY_LAO, ROOT / "results/frequency_lao/sequences.jsonl", 1),
    (Method.BKT_SET_GREEDY, ROOT / "results/bkt_set_greedy/sequences.jsonl", 1),
    (Method.BKT_SET_LAO, ROOT / "results/bkt_set_lao/sequences.jsonl", 1),
    (Method.DKT_SET_GREEDY, ROOT / "results/dkt_set_greedy/sequences.jsonl", 1),
    (Method.DKT_SET_LAO, ROOT / "results/dkt_set_lao/sequences.jsonl", 1),
    (Method.RANDOM_FRONTIER, ROOT / "results/random_frontier/sequences.jsonl", 100),
    (Method.LINEAR_SYLLABUS, ROOT / "results/linear_syllabus_order/sequences.jsonl", 1),
)

ORACLE_METRIC_SOURCES = (
    ("FrozenMonotonicOracle", ROOT / "results/ariadne_greedy/oracle_valid_metrics.csv", None),
    ("FrequencyOracle", ROOT / "results/frequency_greedy/oracle_valid_metrics.csv", None),
    ("BKT-derived Set Oracle", ROOT / "results/bkt_set/oracle_metrics.csv", "validation"),
    ("DKT-derived Set Oracle", ROOT / "results/dkt_set/oracle_metrics.csv", "validation"),
)

SCORED_FIELDS = (
    "method", "target_node", "run_id", "valid", "evaluation_cost",
    "optimal_cost", "normalized_regret", "sequence_hash", "invalid_reason",
)


def _write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_and_validate_records(
    conditions=CONDITIONS,
    *,
    manifest: dict | None = None,
) -> list[SequenceRecord]:
    manifest = load_manifest() if manifest is None else manifest
    expected_manifest = manifest_hash(manifest)
    targets = set(manifest["targets"])
    closures = {item["target_node"]: item["closure_hash"] for item in manifest["closures"]}
    combined: list[SequenceRecord] = []

    for method, path, runs_per_target in conditions:
        records = read_jsonl(path)
        expected_count = len(targets) * runs_per_target
        if len(records) != expected_count:
            raise ValueError(f"{method.value}: expected {expected_count} records, got {len(records)}")
        identities = {(r.target_node, r.run_id) for r in records}
        expected_identities = {
            (target, run_id) for target in targets for run_id in range(runs_per_target)
        }
        if identities != expected_identities:
            raise ValueError(f"{method.value}: target/run identity grid mismatch")
        for record in records:
            if record.method is not method:
                raise ValueError(f"{path}: unexpected method {record.method.value}")
            if record.metadata.get("manifest_hash") != expected_manifest:
                raise ValueError(f"{method.value}: manifest hash mismatch")
            if record.metadata.get("closure_hash") != closures[record.target_node]:
                raise ValueError(f"{method.value}: closure hash mismatch for {record.target_node}")
        combined.extend(records)

    identities = {(r.method.value, r.target_node, r.run_id) for r in combined}
    if len(identities) != len(combined):
        raise ValueError("Duplicate aggregate (method, target_node, run_id) identity")
    method_order = {
        method: index for index, (method, _, _) in enumerate(conditions)
    }
    combined.sort(
        key=lambda record: (
            method_order[record.method],
            record.target_node,
            record.run_id,
        )
    )
    return combined


def build_per_target(scored: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in scored:
        groups.setdefault((row["method"], row["target_node"]), []).append(row)
    output = []
    for (method, target), rows in sorted(groups.items()):
        regrets = [float(row["normalized_regret"]) for row in rows]
        costs = [float(row["evaluation_cost"]) for row in rows]
        optimal = {float(row["optimal_cost"]) for row in rows}
        if len(optimal) != 1:
            raise ValueError(f"Optimal cost mismatch for {method}, target {target}")
        output.append({
            "method": method,
            "target_node": target,
            "records": len(rows),
            "valid_records": sum(bool(row["valid"]) for row in rows),
            "mean_evaluation_cost": statistics.fmean(costs),
            "optimal_cost": next(iter(optimal)),
            "mean_normalized_regret": statistics.fmean(regrets),
            "median_normalized_regret": statistics.median(regrets),
            "std_normalized_regret": statistics.pstdev(regrets),
            "p05_normalized_regret": _percentile(regrets, 0.05),
            "p95_normalized_regret": _percentile(regrets, 0.95),
            "min_normalized_regret": min(regrets),
            "max_normalized_regret": max(regrets),
        })
    return output


def build_main_table(per_target: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in per_target:
        groups.setdefault(row["method"], []).append(row)
    output = []
    order = {method.value: index for index, (method, _, _) in enumerate(CONDITIONS)}
    for method, rows in sorted(groups.items(), key=lambda item: order[item[0]]):
        target_regrets = [float(row["mean_normalized_regret"]) for row in rows]
        target_costs = [float(row["mean_evaluation_cost"]) for row in rows]
        output.append({
            "method": method,
            "targets": len(rows),
            "records": sum(int(row["records"]) for row in rows),
            "valid_records": sum(int(row["valid_records"]) for row in rows),
            "mean_evaluation_cost_across_targets": statistics.fmean(target_costs),
            "mean_normalized_regret_across_targets": statistics.fmean(target_regrets),
            "median_normalized_regret_across_targets": statistics.median(target_regrets),
            "std_normalized_regret_across_targets": statistics.pstdev(target_regrets),
            "min_normalized_regret_across_targets": min(target_regrets),
            "max_normalized_regret_across_targets": max(target_regrets),
        })
    return output


def build_oracle_metrics() -> list[dict]:
    manifest = manifest_hash(load_manifest())
    evaluator = sha256_file(ROOT / "experiments/common/evaluator.py")
    identity_fields = {
        "manifest_hash", "evaluator_hash", "dag_hash", "oracle_checkpoint_hash",
        "train_validation_split_hash", "train_artifact_hash", "validation_artifact_hash",
        "config_hash", "checkpoint_hash", "parameter_values_hash",
        "pooled_parameter_vector_hash", "bkt_parameter_artifact_hash",
        "pooled_parameter_artifact_hash", "tuple_collection_hash", "split_hash",
        "compression_config_hash", "teacher_tensor_hash", "teacher_checkpoint_hash",
        "coverage", "per_target_raw_count", "per_target_grouped_count",
        "probability_source", "planning_mode_x", "inference_backend", "split",
    }
    output = []
    for oracle, path, selected_split in ORACLE_METRIC_SOURCES:
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if selected_split is not None:
            rows = [row for row in rows if row.get("split") == selected_split]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one selected metrics row in {path}")
        row = rows[0]
        if row.get("manifest_hash", manifest) != manifest:
            raise ValueError(f"Oracle metrics manifest mismatch: {path}")
        source_evaluator = row.get("evaluator_hash", "")
        split = row.get("split") or "validation"
        for metric, raw_value in row.items():
            if metric in identity_fields or raw_value in (None, ""):
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            output.append({
                "oracle": oracle,
                "split": split,
                "metric": metric,
                "value": value,
                "source_path": path.relative_to(ROOT).as_posix(),
                "source_sha256": sha256_file(path),
                "manifest_hash": manifest,
                "source_evaluator_hash": source_evaluator,
                "aggregation_evaluator_hash": evaluator,
            })
    return output


def build_aggregation_manifest(
    records: list[SequenceRecord],
    output_rows: dict[str, int],
) -> dict:
    manifest = load_manifest()
    inputs = {}
    by_method: dict[Method, list[SequenceRecord]] = {}
    for record in records:
        by_method.setdefault(record.method, []).append(record)
    for method, path, runs_per_target in CONDITIONS:
        method_records = by_method[method]
        inputs[method.value] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "records": len(method_records),
            "runs_per_target": runs_per_target,
            "source_evaluator_hashes": sorted(
                {
                    str(record.metadata.get("evaluator_hash"))
                    for record in method_records
                }
            ),
        }
    outputs = {}
    for name, rows in output_rows.items():
        path = OUTPUT / name
        outputs[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "rows": rows,
        }
    return {
        "schema_version": 1,
        "manifest_hash": manifest_hash(manifest),
        "evaluator_hash": sha256_file(ROOT / "experiments/common/evaluator.py"),
        "sorting": ["method_order", "target_node", "run_id"],
        "inputs": inputs,
        "outputs": outputs,
        "generation_command": "python experiments/aggregate_results.py",
    }


def main() -> None:
    records = load_and_validate_records()
    evaluator = SequenceEvaluator.from_artifacts()
    scored_objects = evaluator.score_records(records)
    if any(not row.valid for row in scored_objects):
        invalid = [row for row in scored_objects if not row.valid]
        raise ValueError(f"Public evaluator rejected {len(invalid)} aggregate records")
    scored = [asdict(row) for row in scored_objects]
    per_target = build_per_target(scored)
    main_table = build_main_table(per_target)
    oracle_metrics = build_oracle_metrics()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "all_sequences.jsonl", records)
    _write_csv(OUTPUT / "scored_sequences.csv", scored, SCORED_FIELDS)
    _write_csv(OUTPUT / "per_target.csv", per_target, tuple(per_target[0]))
    _write_csv(OUTPUT / "main_table.csv", main_table, tuple(main_table[0]))
    _write_csv(OUTPUT / "oracle_metrics.csv", oracle_metrics, tuple(oracle_metrics[0]))
    aggregation_manifest = build_aggregation_manifest(
        records,
        {
            "all_sequences.jsonl": len(records),
            "scored_sequences.csv": len(scored),
            "per_target.csv": len(per_target),
            "main_table.csv": len(main_table),
            "oracle_metrics.csv": len(oracle_metrics),
        },
    )
    (OUTPUT / "aggregation_manifest.json").write_text(
        json.dumps(aggregation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"records={len(records)}")
    print(f"valid={sum(row.valid for row in scored_objects)}")
    print(f"per_target_rows={len(per_target)}")
    print(f"main_table_rows={len(main_table)}")
    print(f"oracle_metric_rows={len(oracle_metrics)}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
