"""Finalize Task 17 LLM artifacts, score valid sequences, and aggregate results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
if SCRIPT_DIRECTORY in sys.path:
    sys.path.remove(SCRIPT_DIRECTORY)
sys.path.insert(0, str(ROOT))

import statistics

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash
from experiments.common.schema import Method, SequenceRecord, write_jsonl
from experiments.llm.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_file,
    value_hash,
)


LLM = ROOT / "experiments" / "llm"
RESULTS = ROOT / "results" / "llm"
MODEL_ORDER = {"closed_frontier": 0, "open_weight": 1}
CONDITION_ORDER = {"zero": 0, "full": 1}
SCORED_FIELDS = (
    "method", "model_key", "condition", "target_node", "run_id", "valid",
    "evaluation_cost", "optimal_cost", "normalized_regret", "sequence_hash",
    "invalid_reason",
)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        for row in rows:
            file.write(canonical_json_bytes(row))
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _file_ref(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _run_sort(run: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        MODEL_ORDER[run["model_key"]],
        CONDITION_ORDER[run["condition"]],
        int(run["target_node"]),
        int(run["run_id"]),
    )


def collect_statuses() -> tuple[list[dict[str, Any]], list[SequenceRecord], list[dict], list[dict]]:
    run_manifest = load_json(LLM / "generated" / "run_manifest.json")
    shared_manifest = load_manifest()
    closures = {row["target_node"]: row["closure_hash"] for row in shared_manifest["closures"]}
    statuses: list[dict[str, Any]] = []
    records: list[SequenceRecord] = []
    invalid: list[dict[str, Any]] = []
    transport: list[dict[str, Any]] = []

    for run in sorted(run_manifest["runs"], key=_run_sort):
        key = run["logical_run_key"]
        request_paths = sorted((RESULTS / "requests" / key).glob("*.json"))
        raw_paths = sorted((RESULTS / "raw" / key).glob("*.json"))
        parsed_paths = sorted((RESULTS / "parsed" / key).glob("*.json"))
        if len(request_paths) != 1:
            raise ValueError(f"{key}: expected exactly one request artifact")
        request_path = request_paths[0]
        request = load_json(request_path)
        base = {
            "schema_version": 1,
            "logical_run_key": key,
            "model_key": run["model_key"],
            "method": run["method"],
            "condition": run["condition"],
            "target_node": run["target_node"],
            "run_id": run["run_id"],
            "attempt": request["attempt"],
            "request_artifact": _file_ref(request_path),
        }

        if request.get("status") == "transport_ambiguous_terminal":
            if raw_paths or parsed_paths:
                raise ValueError(f"{key}: terminal transport-ambiguous has raw/parsed")
            row = {
                **base,
                "terminal_status": "transport_ambiguous",
                "provider_response_obtained": False,
                "structurally_valid": None,
                "error_code": "transport_ambiguous",
                "transport_error": request["transport_error"],
                "raw_artifact": None,
                "parsed_artifact": None,
            }
            statuses.append(row)
            transport.append(row)
            continue

        if len(raw_paths) != 1 or len(parsed_paths) != 1:
            raise ValueError(f"{key}: nonterminal or incomplete artifact set")
        raw_path, parsed_path = raw_paths[0], parsed_paths[0]
        raw, parsed = load_json(raw_path), load_json(parsed_path)
        response = raw["provider_response"]
        valid = bool(parsed["sequence_validation"]["valid"])
        error_code = (
            parsed["sequence_validation"].get("error_code")
            or parsed["parse_result"].get("parse_error_code")
        )
        row = {
            **base,
            "terminal_status": "valid" if valid else "model_invalid",
            "provider_response_obtained": True,
            "structurally_valid": valid,
            "error_code": error_code,
            "finish_reason": response["finish_reason"],
            "requested_model_id": response["requested_model_id"],
            "response_model_id": response["response_model_id"],
            "provider_request_id_sha256": hashlib.sha256(
                response["provider_request_id"].encode("utf-8")
            ).hexdigest(),
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "reasoning_tokens": response["reasoning_tokens"],
            "latency_seconds": response["latency_seconds"],
            "raw_artifact": _file_ref(raw_path),
            "parsed_artifact": _file_ref(parsed_path),
        }
        statuses.append(row)
        if not valid:
            invalid.append({
                **row,
                "parse_result": parsed["parse_result"],
                "sequence_validation": parsed["sequence_validation"],
            })
            continue

        metadata = {
            "logical_run_key": key,
            "model_key": run["model_key"],
            "condition": run["condition"],
            "manifest_hash": run_manifest["manifest_hash"],
            "closure_hash": closures[run["target_node"]],
            "protocol_hash": run["protocol_hash"],
            "run_config_hash": parsed["provenance"]["run_config_hash"],
            "provider_config_hash": run["provider_config_hash"],
            "prompt_hash": run["prompt_hash"],
            "mapping_hash": run["mapping_hash"],
            "requested_model_id": response["requested_model_id"],
            "response_model_id": response["response_model_id"],
            "provider_request_id_sha256": row["provider_request_id_sha256"],
            "finish_reason": response["finish_reason"],
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "reasoning_tokens": response["reasoning_tokens"],
            "latency_seconds": response["latency_seconds"],
            "request_artifact_sha256": row["request_artifact"]["sha256"],
            "raw_artifact_sha256": row["raw_artifact"]["sha256"],
            "parsed_artifact_sha256": row["parsed_artifact"]["sha256"],
        }
        records.append(SequenceRecord(
            method=Method(run["method"]),
            target_node=int(run["target_node"]),
            run_id=int(run["run_id"]),
            sequence=tuple(parsed["sequence_validation"]["real_sequence"]),
            internal_cost=None,
            metadata=metadata,
        ))

    counts = {name: sum(row["terminal_status"] == name for row in statuses) for name in (
        "valid", "model_invalid", "transport_ambiguous"
    )}
    if len(statuses) != 800 or counts != {
        "valid": 792, "model_invalid": 7, "transport_ambiguous": 1,
    }:
        raise ValueError(f"Unexpected terminal counts: {counts}, total={len(statuses)}")
    return statuses, records, invalid, transport


def build_summaries(
    statuses: list[dict[str, Any]], scored: list[dict[str, Any]],
) -> tuple[list[dict], list[dict], list[dict]]:
    scored_by_key = {
        (row["method"], row["target_node"], row["run_id"]): row for row in scored
    }
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for status in statuses:
        groups.setdefault(
            (status["model_key"], status["condition"], status["target_node"]), []
        ).append(status)

    per_target = []
    for (model, condition, target), rows in sorted(
        groups.items(), key=lambda item: (
            MODEL_ORDER[item[0][0]], CONDITION_ORDER[item[0][1]], item[0][2]
        )
    ):
        valid_rows = [row for row in rows if row["terminal_status"] == "valid"]
        response_count = sum(row["provider_response_obtained"] for row in rows)
        method = valid_rows[0]["method"] if valid_rows else rows[0]["method"]
        scored_rows = [
            scored_by_key[(method, row["target_node"], row["run_id"])]
            for row in valid_rows
        ]
        costs = [float(row["evaluation_cost"]) for row in scored_rows]
        regrets = [float(row["normalized_regret"]) for row in scored_rows]
        per_target.append({
            "model_key": model,
            "condition": condition,
            "method": method,
            "target_node": target,
            "planned_runs": len(rows),
            "provider_responses": response_count,
            "valid_runs": len(valid_rows),
            "model_invalid_runs": sum(row["terminal_status"] == "model_invalid" for row in rows),
            "transport_ambiguous_runs": sum(row["terminal_status"] == "transport_ambiguous" for row in rows),
            "model_validity_rate": len(valid_rows) / response_count,
            "pipeline_yield": len(valid_rows) / len(rows),
            "mean_evaluation_cost_valid_runs": statistics.fmean(costs),
            "median_evaluation_cost_valid_runs": statistics.median(costs),
            "mean_normalized_regret_valid_runs": statistics.fmean(regrets),
            "median_normalized_regret_valid_runs": statistics.median(regrets),
            "std_normalized_regret_valid_runs": statistics.pstdev(regrets),
            "p05_normalized_regret_valid_runs": _percentile(regrets, 0.05),
            "p95_normalized_regret_valid_runs": _percentile(regrets, 0.95),
        })

    main_table = []
    validity_summary = []
    for model in MODEL_ORDER:
        for condition in CONDITION_ORDER:
            target_rows = [
                row for row in per_target
                if row["model_key"] == model and row["condition"] == condition
            ]
            status_rows = [
                row for row in statuses
                if row["model_key"] == model and row["condition"] == condition
            ]
            responses = sum(row["provider_response_obtained"] for row in status_rows)
            valid = sum(row["terminal_status"] == "valid" for row in status_rows)
            summary = {
                "model_key": model,
                "condition": condition,
                "method": target_rows[0]["method"],
                "planned_runs": len(status_rows),
                "provider_responses": responses,
                "valid_runs": valid,
                "model_invalid_runs": sum(row["terminal_status"] == "model_invalid" for row in status_rows),
                "transport_ambiguous_runs": sum(row["terminal_status"] == "transport_ambiguous" for row in status_rows),
                "model_validity_rate_over_responses": valid / responses,
                "pipeline_yield_over_planned": valid / len(status_rows),
                "target_equal_mean_model_validity_rate": statistics.fmean(
                    row["model_validity_rate"] for row in target_rows
                ),
                "target_equal_mean_pipeline_yield": statistics.fmean(
                    row["pipeline_yield"] for row in target_rows
                ),
            }
            validity_summary.append(summary)
            main_table.append({
                **summary,
                "targets": len(target_rows),
                "target_equal_mean_evaluation_cost_valid_runs": statistics.fmean(
                    row["mean_evaluation_cost_valid_runs"] for row in target_rows
                ),
                "target_equal_mean_normalized_regret_valid_runs": statistics.fmean(
                    row["mean_normalized_regret_valid_runs"] for row in target_rows
                ),
                "target_equal_median_normalized_regret_valid_runs": statistics.median(
                    row["mean_normalized_regret_valid_runs"] for row in target_rows
                ),
                "target_equal_std_normalized_regret_valid_runs": statistics.pstdev(
                    row["mean_normalized_regret_valid_runs"] for row in target_rows
                ),
            })
    return validity_summary, per_target, main_table


def _collection(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*.json") if path.is_file())
    entries = [_file_ref(path) for path in files]
    return {"files": entries, "count": len(entries), "collection_hash": value_hash(entries)}


def main() -> None:
    statuses, records, invalid, transport = collect_statuses()
    evaluator = SequenceEvaluator.from_artifacts()
    scored_objects = evaluator.score_records(records)
    if any(not row.valid for row in scored_objects):
        raise ValueError("A structurally valid LLM record failed the public evaluator")
    scored = []
    metadata = {(r.method.value, r.target_node, r.run_id): r.metadata for r in records}
    for row in scored_objects:
        payload = asdict(row)
        source = metadata[(row.method, row.target_node, row.run_id)]
        payload["model_key"] = source["model_key"]
        payload["condition"] = source["condition"]
        scored.append(payload)
    scored.sort(key=lambda row: (
        MODEL_ORDER[row["model_key"]], CONDITION_ORDER[row["condition"]],
        row["target_node"], row["run_id"],
    ))
    validity, per_target, main_table = build_summaries(statuses, scored)

    outputs = {
        "run_status.jsonl": len(statuses),
        "valid_sequences.jsonl": len(records),
        "invalid_runs.jsonl": len(invalid),
        "transport_failures.jsonl": len(transport),
        "validity_summary.csv": len(validity),
        "scored_valid_sequences.csv": len(scored),
        "per_target.csv": len(per_target),
        "main_table.csv": len(main_table),
    }
    _write_jsonl(RESULTS / "run_status.jsonl", statuses)
    write_jsonl(RESULTS / "valid_sequences.jsonl", records)
    _write_jsonl(RESULTS / "invalid_runs.jsonl", invalid)
    _write_jsonl(RESULTS / "transport_failures.jsonl", transport)
    _write_csv(RESULTS / "validity_summary.csv", validity, validity[0].keys())
    _write_csv(RESULTS / "scored_valid_sequences.csv", scored, SCORED_FIELDS)
    _write_csv(RESULTS / "per_target.csv", per_target, per_target[0].keys())
    _write_csv(RESULTS / "main_table.csv", main_table, main_table[0].keys())

    output_refs = {
        name: {**_file_ref(RESULTS / name), "rows": rows}
        for name, rows in outputs.items()
    }
    source_files = [
        LLM / "protocol.json",
        LLM / "run_config.json",
        LLM / "generated" / "prompt_manifest.json",
        LLM / "generated" / "run_manifest.json",
        LLM / "generated" / "provider_preflight.json",
        ROOT / "experiments" / "common" / "manifest.json",
        ROOT / "experiments" / "common" / "evaluator.py",
        Path(__file__),
    ]
    formal_manifest = {
        "schema_version": 1,
        "status": "complete_with_declared_transport_failure",
        "planned_runs": 800,
        "provider_responses": 799,
        "valid_sequences": 792,
        "model_invalid_responses": 7,
        "transport_ambiguous_runs": 1,
        "model_validity_definition": "valid_sequences / obtained_provider_responses",
        "pipeline_yield_definition": "valid_sequences / planned_runs",
        "overall_model_validity": 792 / 799,
        "overall_pipeline_yield": 792 / 800,
        "aggregation_order": ["within_target_repetitions", "equal_weight_across_ten_targets"],
        "cost_and_regret_conditioning": "structurally_valid_runs_only",
        "sorting": ["model_key", "condition", "target_node", "run_id"],
        "materialized_manifest_hash": manifest_hash(load_manifest()),
        "inputs": {
            "source_files": [_file_ref(path) for path in source_files],
            "requests": _collection(RESULTS / "requests"),
            "raw": _collection(RESULTS / "raw"),
            "parsed": _collection(RESULTS / "parsed"),
        },
        "outputs": output_refs,
        "generation_command": "python experiments/llm/finalize_results.py",
    }
    formal_manifest["manifest_payload_hash"] = value_hash(formal_manifest)
    atomic_write_json(RESULTS / "formal_run_manifest.json", formal_manifest)
    print(json.dumps({
        "planned": len(statuses),
        "provider_responses": sum(row["provider_response_obtained"] for row in statuses),
        "valid": len(records),
        "model_invalid": len(invalid),
        "transport_ambiguous": len(transport),
        "scored": len(scored),
        "per_target_rows": len(per_target),
        "main_table_rows": len(main_table),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
