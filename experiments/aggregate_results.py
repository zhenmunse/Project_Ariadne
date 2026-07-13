"""Aggregate all 14 approved conditions through the public evaluator.

Task 18 extends the Task 14 aggregate with the four formal LLM conditions.  It
never regenerates condition-level artifacts: it reads the frozen sequence and
run-status files, validates them, and writes only ``results/final`` outputs.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final"
sys.path.insert(0, str(ROOT))

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, SequenceRecord, read_jsonl, write_jsonl


# The tuple shape is retained for the Task 14 validation API and its regression
# fixtures: method, frozen sequence artifact, planned runs per target.
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

LLM_SEQUENCE_PATH = ROOT / "results/llm/valid_sequences.jsonl"
LLM_STATUS_PATH = ROOT / "results/llm/run_status.jsonl"
LLM_CONDITIONS = (
    (Method.GPT56_SOL_ZERO, 20),
    (Method.GPT56_SOL_FULL, 20),
    (Method.DEEPSEEK_V4_ZERO, 20),
    (Method.DEEPSEEK_V4_FULL, 20),
)
ALL_METHODS = tuple(item[0] for item in CONDITIONS) + tuple(item[0] for item in LLM_CONDITIONS)
METHOD_ORDER = {method: index for index, method in enumerate(ALL_METHODS)}
DISPLAY_NAMES = {
    Method.ARIADNE_GREEDY: "Ariadne Greedy",
    Method.ARIADNE_LAO: "Ariadne LAO*",
    Method.FREQUENCY_GREEDY: "Frequency Greedy",
    Method.FREQUENCY_LAO: "Frequency LAO*",
    Method.BKT_SET_GREEDY: "BKT-set Greedy",
    Method.BKT_SET_LAO: "BKT-set LAO*",
    Method.DKT_SET_GREEDY: "DKT-set Greedy",
    Method.DKT_SET_LAO: "DKT-set LAO*",
    Method.RANDOM_FRONTIER: "Random Frontier",
    Method.LINEAR_SYLLABUS: "Linear Syllabus",
    Method.GPT56_SOL_ZERO: "GPT-5.6 SOL Zero",
    Method.GPT56_SOL_FULL: "GPT-5.6 SOL Full",
    Method.DEEPSEEK_V4_ZERO: "DeepSeek V4 Pro Zero",
    Method.DEEPSEEK_V4_FULL: "DeepSeek V4 Pro Full",
}

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
PER_TARGET_FIELDS = (
    "method", "display_name", "target_node", "planned_runs", "provider_responses",
    "valid_runs", "model_invalid_runs", "transport_ambiguous_runs",
    "model_validity_rate", "validity_rate", "pipeline_yield", "mean_evaluation_cost",
    "median_evaluation_cost", "optimal_cost", "mean_normalized_regret",
    "median_normalized_regret", "std_normalized_regret", "p05_normalized_regret",
    "p95_normalized_regret", "min_normalized_regret", "max_normalized_regret",
)
MAIN_FIELDS = (
    "method", "display_name", "targets", "planned_runs", "provider_responses",
    "valid_runs", "model_invalid_runs", "transport_ambiguous_runs",
    "model_validity_rate", "validity_rate", "pipeline_yield", "mean_evaluation_cost",
    "mean_normalized_regret", "median_normalized_regret", "std_normalized_regret",
    "p05_normalized_regret", "p95_normalized_regret",
)
VALIDITY_FIELDS = (
    "method", "display_name", "planned_runs", "provider_responses", "valid_runs",
    "model_invalid_runs", "transport_ambiguous_runs", "model_validity_rate",
    "validity_rate", "pipeline_yield", "model_validity_denominator",
    "validity_denominator", "cost_regret_conditioning",
)
STATISTICAL_FIELDS = (
    "comparison", "full_method", "zero_method", "sample_unit", "targets",
    "difference_direction", "paired_mean_difference", "bootstrap_ci_low",
    "bootstrap_ci_high", "bootstrap_replicates", "bootstrap_seed",
    "cohens_dz", "matched_pairs_rank_biserial", "permutation_p_value",
    "permutation_p_value_holm", "wilcoxon_statistic", "wilcoxon_p_value",
    "wilcoxon_p_value_holm", "target_differences_json",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_dict_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _read_dict_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty collection")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_and_validate_records(
    conditions=CONDITIONS,
    *,
    manifest: dict | None = None,
) -> list[SequenceRecord]:
    """Load the ten frozen non-LLM conditions with an exact identity grid."""
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
        expected_identities = {(target, run_id) for target in targets for run_id in range(runs_per_target)}
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
    method_order = {method: index for index, (method, _, _) in enumerate(conditions)}
    combined.sort(key=lambda r: (method_order[r.method], r.target_node, r.run_id))
    return combined


def load_llm_records(*, manifest: dict | None = None) -> tuple[list[SequenceRecord], list[dict[str, Any]]]:
    """Load the 792 valid LLM sequences and all 800 frozen terminal statuses."""
    manifest = load_manifest() if manifest is None else manifest
    expected_manifest = manifest_hash(manifest)
    targets = set(manifest["targets"])
    closures = {item["target_node"]: item["closure_hash"] for item in manifest["closures"]}
    llm_methods = {method for method, _ in LLM_CONDITIONS}
    records = read_jsonl(LLM_SEQUENCE_PATH)
    statuses = _read_dict_jsonl(LLM_STATUS_PATH)
    if len(statuses) != 800:
        raise ValueError(f"Expected 800 LLM terminal statuses, got {len(statuses)}")
    counts = {
        "valid": sum(row["terminal_status"] == "valid" for row in statuses),
        "model_invalid": sum(row["terminal_status"] == "model_invalid" for row in statuses),
        "transport_ambiguous": sum(row["terminal_status"] == "transport_ambiguous" for row in statuses),
    }
    if counts != {"valid": 792, "model_invalid": 7, "transport_ambiguous": 1}:
        raise ValueError(f"Unexpected Task 17 terminal counts: {counts}")
    status_ids = {(Method(row["method"]), int(row["target_node"]), int(row["run_id"])) for row in statuses}
    expected_status_ids = {
        (method, target, run_id)
        for method, runs in LLM_CONDITIONS
        for target in targets
        for run_id in range(runs)
    }
    if status_ids != expected_status_ids or len(status_ids) != len(statuses):
        raise ValueError("LLM terminal status identity grid mismatch")
    valid_status_ids = {
        (Method(row["method"]), int(row["target_node"]), int(row["run_id"]))
        for row in statuses if row["terminal_status"] == "valid"
    }
    record_ids = {(r.method, r.target_node, r.run_id) for r in records}
    if len(records) != 792 or len(record_ids) != len(records) or record_ids != valid_status_ids:
        raise ValueError("LLM valid sequence/status identity mismatch")
    for record in records:
        if record.method not in llm_methods:
            raise ValueError(f"Unexpected LLM method: {record.method.value}")
        if record.metadata.get("manifest_hash") != expected_manifest:
            raise ValueError(f"{record.method.value}: manifest hash mismatch")
        if record.metadata.get("closure_hash") != closures[record.target_node]:
            raise ValueError(f"{record.method.value}: closure hash mismatch for {record.target_node}")
    records.sort(key=lambda r: (METHOD_ORDER[r.method], r.target_node, r.run_id))
    statuses.sort(key=lambda row: (METHOD_ORDER[Method(row["method"])], int(row["target_node"]), int(row["run_id"])))
    return records, statuses


def load_all_records() -> tuple[list[SequenceRecord], list[dict[str, Any]]]:
    manifest = load_manifest()
    standard = load_and_validate_records(manifest=manifest)
    llm, llm_statuses = load_llm_records(manifest=manifest)
    records = standard + llm
    records.sort(key=lambda r: (METHOD_ORDER[r.method], r.target_node, r.run_id))
    identities = {(r.method, r.target_node, r.run_id) for r in records}
    if len(identities) != len(records) or len(records) != 1882:
        raise ValueError("Final valid sequence identity/count mismatch")

    standard_statuses = [
        {
            "schema_version": 1,
            "method": record.method.value,
            "target_node": record.target_node,
            "run_id": record.run_id,
            "terminal_status": "valid",
            "provider_response_obtained": True,
            "structurally_valid": True,
            "source": "frozen_condition_sequence",
        }
        for record in standard
    ]
    statuses = standard_statuses + llm_statuses
    statuses.sort(key=lambda row: (METHOD_ORDER[Method(row["method"])], int(row["target_node"]), int(row["run_id"])))
    if len(statuses) != 1890:
        raise ValueError("Final all-run status count mismatch")
    return records, statuses


def build_per_target(scored: list[dict[str, Any]], statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    status_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in scored:
        score_groups.setdefault((row["method"], int(row["target_node"])), []).append(row)
    for row in statuses:
        status_groups.setdefault((row["method"], int(row["target_node"])), []).append(row)

    output: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        for target in load_manifest()["targets"]:
            run_rows = status_groups.get((method.value, target), [])
            valid_rows = score_groups.get((method.value, target), [])
            if not run_rows or not valid_rows:
                raise ValueError(f"Missing final rows for {method.value}, target {target}")
            costs = [float(row["evaluation_cost"]) for row in valid_rows]
            regrets = [float(row["normalized_regret"]) for row in valid_rows]
            optimal = {float(row["optimal_cost"]) for row in valid_rows}
            if len(optimal) != 1:
                raise ValueError(f"Optimal cost mismatch for {method.value}, target {target}")
            provider_responses = sum(bool(row["provider_response_obtained"]) for row in run_rows)
            valid_count = sum(row["terminal_status"] == "valid" for row in run_rows)
            invalid_count = sum(row["terminal_status"] == "model_invalid" for row in run_rows)
            transport_count = sum(row["terminal_status"] == "transport_ambiguous" for row in run_rows)
            if valid_count != len(valid_rows):
                raise ValueError(f"Status/score valid count mismatch for {method.value}, target {target}")
            output.append({
                "method": method.value,
                "display_name": DISPLAY_NAMES[method],
                "target_node": target,
                "planned_runs": len(run_rows),
                "provider_responses": provider_responses,
                "valid_runs": valid_count,
                "model_invalid_runs": invalid_count,
                "transport_ambiguous_runs": transport_count,
                "model_validity_rate": valid_count / provider_responses,
                "validity_rate": valid_count / len(run_rows),
                "pipeline_yield": valid_count / len(run_rows),
                "mean_evaluation_cost": statistics.fmean(costs),
                "median_evaluation_cost": statistics.median(costs),
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


def build_main_table(per_target: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        rows = [row for row in per_target if row["method"] == method.value]
        if len(rows) != 10:
            raise ValueError(f"Expected ten target rows for {method.value}")
        target_costs = [float(row["mean_evaluation_cost"]) for row in rows]
        target_regrets = [float(row["mean_normalized_regret"]) for row in rows]
        planned = sum(int(row["planned_runs"]) for row in rows)
        responses = sum(int(row["provider_responses"]) for row in rows)
        valid = sum(int(row["valid_runs"]) for row in rows)
        output.append({
            "method": method.value,
            "display_name": DISPLAY_NAMES[method],
            "targets": 10,
            "planned_runs": planned,
            "provider_responses": responses,
            "valid_runs": valid,
            "model_invalid_runs": sum(int(row["model_invalid_runs"]) for row in rows),
            "transport_ambiguous_runs": sum(int(row["transport_ambiguous_runs"]) for row in rows),
            "model_validity_rate": valid / responses,
            "validity_rate": valid / planned,
            "pipeline_yield": valid / planned,
            "mean_evaluation_cost": statistics.fmean(target_costs),
            "mean_normalized_regret": statistics.fmean(target_regrets),
            "median_normalized_regret": statistics.median(target_regrets),
            "std_normalized_regret": statistics.pstdev(target_regrets),
            "p05_normalized_regret": _percentile(target_regrets, 0.05),
            "p95_normalized_regret": _percentile(target_regrets, 0.95),
        })
    return output


def build_validity_table(main_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        key: row[key] for key in (
            "method", "display_name", "planned_runs", "provider_responses", "valid_runs",
            "model_invalid_runs", "transport_ambiguous_runs", "model_validity_rate",
            "validity_rate", "pipeline_yield",
        )
    } | {
        "model_validity_denominator": "obtained_provider_responses",
        "validity_denominator": "all_planned_runs",
        "cost_regret_conditioning": "structurally_valid_runs_only",
    } for row in main_table]


def _exact_sign_permutation_p_value(differences: list[float]) -> float:
    observed = abs(statistics.fmean(differences))
    extreme = 0
    total = 0
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(statistics.fmean(sign * value for sign, value in zip(signs, differences)))
        extreme += permuted + tolerance >= observed
        total += 1
    return extreme / total


def _rank_biserial(differences: list[float]) -> float:
    nonzero = [(abs(value), value) for value in differences if value != 0.0]
    if not nonzero:
        return 0.0
    ordered = sorted(nonzero)
    ranks = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        for position in range(index, end):
            ranks[position] = rank
        index = end
    positive = sum(rank for rank, (_, value) in zip(ranks, ordered) if value > 0)
    negative = sum(rank for rank, (_, value) in zip(ranks, ordered) if value < 0)
    return (positive - negative) / (positive + negative)


def _average_signed_ranks(differences: list[float]) -> list[tuple[float, float]]:
    """Return (average absolute rank, signed difference), dropping exact zeros."""
    ordered = sorted((abs(value), value) for value in differences if value != 0.0)
    ranked: list[tuple[float, float]] = []
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        ranked.extend((rank, ordered[position][1]) for position in range(index, end))
        index = end
    return ranked


def _exact_wilcoxon(differences: list[float]) -> tuple[float, float]:
    """Two-sided exact signed-rank test with average ranks for ties."""
    ranked = _average_signed_ranks(differences)
    if not ranked:
        return 0.0, 1.0
    total_rank = sum(rank for rank, _ in ranked)
    observed_positive = sum(rank for rank, value in ranked if value > 0)
    observed = min(observed_positive, total_rank - observed_positive)
    extreme = 0
    total = 0
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(ranked)):
        positive = sum(rank for sign, (rank, _) in zip(signs, ranked) if sign > 0)
        statistic = min(positive, total_rank - positive)
        extreme += statistic <= observed + tolerance
        total += 1
    return observed, extreme / total


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def build_statistical_tests(per_target: list[dict[str, Any]], *, seed: int = 42, replicates: int = 100_000) -> list[dict[str, Any]]:
    comparisons = (
        ("GPT Full vs GPT Zero", Method.GPT56_SOL_FULL, Method.GPT56_SOL_ZERO),
        ("DeepSeek Full vs DeepSeek Zero", Method.DEEPSEEK_V4_FULL, Method.DEEPSEEK_V4_ZERO),
    )
    means = {
        (row["method"], int(row["target_node"])): float(row["mean_normalized_regret"])
        for row in per_target
    }
    rows: list[dict[str, Any]] = []
    for comparison_index, (label, full, zero) in enumerate(comparisons):
        differences = [
            means[(full.value, target)] - means[(zero.value, target)]
            for target in load_manifest()["targets"]
        ]
        rng = random.Random(seed + comparison_index)
        bootstrap = [
            statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences)
            for _ in range(replicates)
        ]
        sample_std = statistics.stdev(differences)
        statistic, wilcoxon_p = _exact_wilcoxon(differences)
        rows.append({
            "comparison": label,
            "full_method": full.value,
            "zero_method": zero.value,
            "sample_unit": "target_level_mean_normalized_regret",
            "targets": len(differences),
            "difference_direction": "full_minus_zero; negative favors Full",
            "paired_mean_difference": statistics.fmean(differences),
            "bootstrap_ci_low": _percentile(bootstrap, 0.025),
            "bootstrap_ci_high": _percentile(bootstrap, 0.975),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed + comparison_index,
            "cohens_dz": statistics.fmean(differences) / sample_std if sample_std else 0.0,
            "matched_pairs_rank_biserial": _rank_biserial(differences),
            "permutation_p_value": _exact_sign_permutation_p_value(differences),
            "permutation_p_value_holm": None,
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_p_value": float(wilcoxon_p),
            "wilcoxon_p_value_holm": None,
            "target_differences_json": json.dumps(differences, separators=(",", ":")),
        })
    permutation_adjusted = _holm([float(row["permutation_p_value"]) for row in rows])
    wilcoxon_adjusted = _holm([float(row["wilcoxon_p_value"]) for row in rows])
    for row, permutation, signed_rank in zip(rows, permutation_adjusted, wilcoxon_adjusted):
        row["permutation_p_value_holm"] = permutation
        row["wilcoxon_p_value_holm"] = signed_rank
    return rows


def build_oracle_metrics() -> list[dict[str, Any]]:
    current_manifest = manifest_hash(load_manifest())
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
        if row.get("manifest_hash", current_manifest) != current_manifest:
            raise ValueError(f"Oracle metrics manifest mismatch: {path}")
        for metric, raw_value in row.items():
            if metric in identity_fields or raw_value in (None, ""):
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            output.append({
                "oracle": oracle,
                "split": row.get("split") or "validation",
                "metric": metric,
                "value": value,
                "source_path": path.relative_to(ROOT).as_posix(),
                "source_sha256": sha256_file(path),
                "manifest_hash": current_manifest,
                "source_evaluator_hash": row.get("evaluator_hash", ""),
                "aggregation_evaluator_hash": evaluator,
            })
    return output


def build_aggregation_manifest(records: list[SequenceRecord], output_rows: dict[str, int]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    by_method: dict[Method, list[SequenceRecord]] = {}
    for record in records:
        by_method.setdefault(record.method, []).append(record)
    for method, path, runs in CONDITIONS:
        inputs[method.value] = {
            "path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path),
            "valid_records": len(by_method[method]), "planned_runs_per_target": runs,
        }
    for method, runs in LLM_CONDITIONS:
        inputs[method.value] = {
            "path": LLM_SEQUENCE_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(LLM_SEQUENCE_PATH), "valid_records": len(by_method[method]),
            "planned_runs_per_target": runs,
            "terminal_status_path": LLM_STATUS_PATH.relative_to(ROOT).as_posix(),
            "terminal_status_sha256": sha256_file(LLM_STATUS_PATH),
        }
    outputs = {
        name: {
            "path": (OUTPUT / name).relative_to(ROOT).as_posix(),
            "sha256": sha256_file(OUTPUT / name), "rows": rows,
        }
        for name, rows in output_rows.items()
    }
    materialized = load_manifest()
    return {
        "schema_version": 2,
        "manifest_hash": manifest_hash(materialized),
        "evaluator_hash": sha256_file(ROOT / "experiments/common/evaluator.py"),
        "sorting": ["method_order", "target_node", "run_id"],
        "aggregation": {
            "cost_regret_conditioning": "structurally_valid_runs_only",
            "within_target": "mean over valid repetitions",
            "across_targets": "equal weight over ten target-level means",
            "distribution_columns": "distribution of ten target-level mean regrets",
        },
        "inputs": inputs,
        "outputs": outputs,
        "generation_command": "python experiments/finalize_all_results.py",
    }


def main() -> None:
    records, statuses = load_all_records()
    evaluator = SequenceEvaluator.from_artifacts()
    scored_objects = evaluator.score_records(records)
    if any(not row.valid for row in scored_objects):
        raise ValueError(f"Public evaluator rejected {sum(not row.valid for row in scored_objects)} final records")
    scored = [asdict(row) for row in scored_objects]
    per_target = build_per_target(scored, statuses)
    main_table = build_main_table(per_target)
    validity = build_validity_table(main_table)
    statistical_tests = build_statistical_tests(per_target)
    oracle_metrics = build_oracle_metrics()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "all_sequences.jsonl", records)
    _write_dict_jsonl(OUTPUT / "all_run_status.jsonl", statuses)
    _write_csv(OUTPUT / "scored_sequences.csv", scored, SCORED_FIELDS)
    _write_csv(OUTPUT / "per_target.csv", per_target, PER_TARGET_FIELDS)
    _write_csv(OUTPUT / "main_table.csv", main_table, MAIN_FIELDS)
    _write_csv(OUTPUT / "validity_table.csv", validity, VALIDITY_FIELDS)
    _write_csv(OUTPUT / "oracle_metrics.csv", oracle_metrics, tuple(oracle_metrics[0]))
    _write_csv(OUTPUT / "statistical_tests.csv", statistical_tests, STATISTICAL_FIELDS)
    output_rows = {
        "all_sequences.jsonl": len(records),
        "all_run_status.jsonl": len(statuses),
        "scored_sequences.csv": len(scored),
        "per_target.csv": len(per_target),
        "main_table.csv": len(main_table),
        "validity_table.csv": len(validity),
        "oracle_metrics.csv": len(oracle_metrics),
        "statistical_tests.csv": len(statistical_tests),
    }
    aggregate = build_aggregation_manifest(records, output_rows)
    (OUTPUT / "aggregation_manifest.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"planned_runs={len(statuses)}")
    print(f"valid_sequences={len(records)}")
    print(f"per_target_rows={len(per_target)}")
    print(f"main_table_rows={len(main_table)}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
