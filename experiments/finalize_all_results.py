"""Create the Task 18 final tables and the single provenance root.

This script is intentionally read-only outside ``results/final``.  Approved
condition artifacts from Tasks 6--17 are inputs; only the unified aggregate and
``final_freeze_manifest.json`` are regenerated.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
sys.path.insert(0, str(ROOT))

from experiments import aggregate_results
from experiments.common.manifest import artifact_collection, load_manifest, manifest_hash, sha256_file


STANDARD_RESULT_DIRECTORIES = (
    "ariadne_greedy", "ariadne_lao", "frequency_greedy", "frequency_lao",
    "bkt_set_greedy", "bkt_set_lao", "dkt_set_greedy", "dkt_set_lao",
    "random_frontier", "linear_syllabus_order", "bkt_set", "dkt_set",
)
SEQUENCE_INPUTS = tuple(path for _, path, _ in aggregate_results.CONDITIONS) + (
    ROOT / "results/llm/valid_sequences.jsonl",
)
FINAL_TABLES = (
    "scored_sequences.csv", "per_target.csv", "main_table.csv",
    "validity_table.csv", "oracle_metrics.csv", "statistical_tests.csv",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _value_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _file_ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _relative(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _collection(paths: Iterable[Path]) -> dict[str, Any]:
    files = [_file_ref(path) for path in sorted(set(paths), key=lambda item: _relative(item))]
    return {"count": len(files), "collection_hash": _value_hash(files), "files": files}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Invalid Git commit SHA: {value!r}")
    return value


def _validate_formal_llm_scope(statuses: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "planned_runs": len(statuses),
        "provider_responses": sum(bool(row["provider_response_obtained"]) for row in statuses),
        "valid_sequences": sum(row["terminal_status"] == "valid" for row in statuses),
        "model_invalid_responses": sum(row["terminal_status"] == "model_invalid" for row in statuses),
        "transport_ambiguous_runs": sum(row["terminal_status"] == "transport_ambiguous" for row in statuses),
    }
    expected = {
        "planned_runs": 800, "provider_responses": 799, "valid_sequences": 792,
        "model_invalid_responses": 7, "transport_ambiguous_runs": 1,
    }
    if counts != expected:
        raise ValueError(f"Task 17 terminal contract changed: {counts}")
    for row in statuses:
        for artifact_key in ("request_artifact", "raw_artifact", "parsed_artifact"):
            artifact = row.get(artifact_key)
            if artifact is None:
                continue
            path = str(artifact["path"])
            if "/smoke/" in f"/{path}/" or "/pilot_" in f"/{path}/":
                raise ValueError(f"Excluded smoke/pilot artifact entered formal scope: {path}")
            actual = ROOT / path
            if sha256_file(actual) != artifact["sha256"]:
                raise ValueError(f"Task 17 artifact hash mismatch: {path}")
    return counts


def _status_artifact_collection(statuses: list[dict[str, Any]], key: str) -> dict[str, Any]:
    paths = [ROOT / row[key]["path"] for row in statuses if row.get(key) is not None]
    return _collection(paths)


def _official_prior_results() -> dict[str, Any]:
    paths: list[Path] = []
    for directory in STANDARD_RESULT_DIRECTORIES:
        paths.extend(path for path in (ROOT / "results" / directory).iterdir() if path.is_file())
    return _collection(paths)


def _artifact_group(directory: Path) -> dict[str, Any]:
    return _collection(path for path in directory.iterdir() if path.is_file())


def _model_freeze(statuses: list[dict[str, Any]]) -> dict[str, Any]:
    config = json.loads((ROOT / "experiments/llm/run_config.json").read_text(encoding="utf-8"))
    models = {}
    for model_key, model_config in config["models"].items():
        rows = [row for row in statuses if row["model_key"] == model_key and row["provider_response_obtained"]]
        requested = sorted({row["requested_model_id"] for row in rows})
        response = sorted({row["response_model_id"] for row in rows})
        if requested != [model_config["requested_model_id"]] or response != requested:
            raise ValueError(f"Requested/response model identity mismatch for {model_key}")
        models[model_key] = {
            "display_name": "GPT-5.6 SOL" if model_key == "closed_frontier" else "DeepSeek V4 Pro",
            "provider": model_config["provider"],
            "endpoint": model_config["endpoint"],
            "requested_model_ids": requested,
            "response_model_ids": response,
            "reasoning": model_config["reasoning"],
            "max_output_tokens": model_config["max_output_tokens"],
            "temperature": model_config["temperature"],
            "top_p": model_config["top_p"],
            "repetitions": config["repetitions"],
        }
    return models


def build_manifest() -> dict[str, Any]:
    shared = load_manifest()
    statuses = _jsonl(ROOT / "results/llm/run_status.jsonl")
    llm_counts = _validate_formal_llm_scope(statuses)
    run_config = ROOT / "experiments/llm/run_config.json"
    prompt_manifest = ROOT / "experiments/llm/generated/prompt_manifest.json"
    provider_preflight = ROOT / "experiments/llm/generated/provider_preflight.json"
    formal_run_manifest = ROOT / "results/llm/formal_run_manifest.json"

    sequence_paths = list(SEQUENCE_INPUTS) + [FINAL / "all_sequences.jsonl"]
    table_paths = [FINAL / name for name in FINAL_TABLES]
    table_paths.extend(sorted((ROOT / "results/llm").glob("*.csv")))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen_complete_with_declared_transport_ambiguity",
        "provenance_root": "results/final/final_freeze_manifest.json",
        "generation_command": "python experiments/finalize_all_results.py",
        "code_snapshot": {
            "source_code_commit_sha": _git_commit(),
            "aggregator": _file_ref(ROOT / "experiments/aggregate_results.py"),
            "public_evaluator": _file_ref(ROOT / "experiments/common/evaluator.py"),
        },
        "freeze_generator": _file_ref(ROOT / "experiments/finalize_all_results.py"),
        "shared_protocol": {
            "dag": _file_ref(ROOT / "data/ecs32a_dag_required_full_v1.json"),
            "source_manifest": _file_ref(ROOT / "experiments/common/manifest.json"),
            "materialized_manifest_hash": manifest_hash(shared),
            "frozen_monotonic_oracle_checkpoint": _file_ref(ROOT / "data/processed/oracle_ckpt.pt"),
            "base_cost": shared["base_cost"],
        },
        "splits": {
            "canonical_session_artifacts": {
                "train": _file_ref(ROOT / "data/processed/train_sessions.pkl"),
                "validation": _file_ref(ROOT / "data/processed/valid_sessions.pkl"),
                "test": _file_ref(ROOT / "data/processed/test_sessions.pkl"),
                "train_validation_collection": artifact_collection((
                    ROOT / "data/processed/train_sessions.pkl",
                    ROOT / "data/processed/valid_sessions.pkl",
                )),
            },
            "kt_student_split": _file_ref(ROOT / "data/kt_set/student_split.json"),
            "kt_preprocessing_manifest": _file_ref(ROOT / "data/kt_set/preprocessing_manifest.json"),
        },
        "mapping_hash_collection": _collection((
            ROOT / "data/question_concept_mapping_final.csv",
            ROOT / "experiments/llm/generated/mappings.json",
        )),
        "bkt_set": {
            "teacher_and_surrogate_artifacts": _artifact_group(ROOT / "artifacts/bkt_set"),
            "teacher_parameters": _file_ref(ROOT / "artifacts/bkt_set/bkt_parameters.json"),
            "pooled_teacher_parameters": _file_ref(ROOT / "artifacts/bkt_set/pooled_bkt_parameters.json"),
            "teacher_metadata": _file_ref(ROOT / "artifacts/bkt_set/bkt_teacher_metadata.json"),
            "distillation_metadata": _file_ref(ROOT / "artifacts/bkt_set/distillation_metadata.json"),
            "surrogate_config": _file_ref(ROOT / "artifacts/bkt_set/surrogate_config.json"),
            "surrogate_checkpoint": _file_ref(ROOT / "artifacts/bkt_set/surrogate_checkpoint.pt"),
        },
        "dkt_set": {
            "teacher_and_surrogate_artifacts": _artifact_group(ROOT / "artifacts/dkt_set"),
            "teacher_config": _file_ref(ROOT / "artifacts/dkt_set/dkt_config.json"),
            "teacher_checkpoint": _file_ref(ROOT / "artifacts/dkt_set/dkt_checkpoint.pt"),
            "teacher_metadata": _file_ref(ROOT / "artifacts/dkt_set/dkt_teacher_metadata.json"),
            "distillation_metadata": _file_ref(ROOT / "artifacts/dkt_set/distillation_metadata.json"),
            "surrogate_config": _file_ref(ROOT / "artifacts/dkt_set/surrogate_config.json"),
            "surrogate_checkpoint": _file_ref(ROOT / "artifacts/dkt_set/surrogate_checkpoint.pt"),
        },
        "llm": {
            "protocol": _file_ref(ROOT / "experiments/llm/protocol.json"),
            "run_config": _file_ref(run_config),
            "prompt_manifest": _file_ref(prompt_manifest),
            "run_manifest": _file_ref(ROOT / "experiments/llm/generated/run_manifest.json"),
            "provider_preflight": _file_ref(provider_preflight),
            "formal_run_manifest": _file_ref(formal_run_manifest),
            "models": _model_freeze(statuses),
            "terminal_counts": llm_counts,
            "formal_requests": _status_artifact_collection(statuses, "request_artifact"),
            "formal_raw_outputs": _status_artifact_collection(statuses, "raw_artifact"),
            "formal_parsed_outputs": _status_artifact_collection(statuses, "parsed_artifact"),
            "analysis_exclusions": {
                "smoke": "results/llm/smoke",
                "pilots": ["results/llm/pilot_4096_excluded", "results/llm/pilot_65536_excluded"],
                "included_in_formal_analysis": False,
            },
        },
        "approved_prior_condition_results": _official_prior_results(),
        "sequence_jsonl_hashes": _collection(sequence_paths),
        "generated_table_hashes": _collection(table_paths),
        "final_outputs": _collection(
            path for path in FINAL.iterdir()
            if path.is_file() and path.name != "final_freeze_manifest.json"
        ),
        "final_aggregation_contract": {
            "methods": [method.value for method in aggregate_results.ALL_METHODS],
            "method_count": 14,
            "planned_runs": 1890,
            "valid_sequences": 1882,
            "llm_cost_regret_conditioning": "structurally_valid_runs_only",
            "within_target": "aggregate repetitions first",
            "across_targets": "equal weight across ten targets",
            "statistical_sample_unit": "ten paired target-level mean regrets",
        },
    }
    payload["manifest_payload_hash"] = _value_hash(payload)
    return payload


def main() -> None:
    aggregate_results.main()
    manifest = build_manifest()
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "final_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"final_methods={manifest['final_aggregation_contract']['method_count']}")
    print(f"final_planned_runs={manifest['final_aggregation_contract']['planned_runs']}")
    print(f"final_valid_sequences={manifest['final_aggregation_contract']['valid_sequences']}")
    print(f"freeze_payload_hash={manifest['manifest_payload_hash']}")


if __name__ == "__main__":
    main()
