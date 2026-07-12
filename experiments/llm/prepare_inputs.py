"""Generate deterministic Task 16-1 mappings, statistics, prompts and run plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.common.manifest import load_manifest, manifest_hash
from experiments.llm.anonymize import build_mappings
from experiments.llm.artifacts import atomic_write_json, load_json, sha256_file, value_hash
from experiments.llm.prompts import build_prompt
from experiments.llm.statistics import build_full_statistics


ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "experiments" / "llm"
GENERATED = LLM / "generated"
PROTOCOL_PATH = LLM / "protocol.json"
RUN_CONFIG_PATH = LLM / "run_config.json"


def _source_hashes() -> dict[str, str]:
    names = ["artifacts.py", "anonymize.py", "statistics.py", "prompts.py", "prepare_inputs.py"]
    return {name: sha256_file(LLM / name) for name in names}


def prepare_inputs() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_PATH)
    run_config = load_json(RUN_CONFIG_PATH)
    if protocol["base_cost"] != 60.0:
        raise ValueError("Task 15 base_cost must remain 60.0")
    if run_config["repetitions"] != 20 or not run_config["targets_from_manifest"]:
        raise ValueError("Task 16 run matrix contract mismatch")
    manifest = load_manifest()
    materialized_hash = manifest_hash(manifest)

    mappings = build_mappings()
    atomic_write_json(GENERATED / "mappings.json", mappings)
    statistics = build_full_statistics()
    atomic_write_json(GENERATED / "full_statistics.json", statistics)
    statistics_artifact_hash = sha256_file(GENERATED / "full_statistics.json")

    prompt_rows = []
    prompt_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for bundle in mappings["targets"]:
        target = int(bundle["target_node"])
        for condition in run_config["conditions"]:
            prompt = build_prompt(
                bundle,
                condition,
                statistics,
                manifest_hash=materialized_hash,
                statistics_artifact_hash=statistics_artifact_hash,
            )
            path = GENERATED / "prompts" / str(target) / f"{condition}.json"
            atomic_write_json(path, prompt)
            row = {
                "target_node": target,
                "condition": condition,
                "path": path.relative_to(ROOT).as_posix(),
                "artifact_hash": sha256_file(path),
                "prompt_hash": prompt["prompt_hash"],
                "system_prompt_hash": prompt["system_prompt_hash"],
                "template_hash": prompt["template_hash"],
                "shared_curriculum_hash": prompt["shared_curriculum_hash"],
                "mapping_hash": prompt["mapping_hash"],
                "statistics_hash": prompt["statistics_hash"],
            }
            prompt_rows.append(row)
            prompt_lookup[(target, condition)] = row
    prompt_manifest = {
        "schema_version": 1,
        "manifest_hash": materialized_hash,
        "protocol_hash": sha256_file(PROTOCOL_PATH),
        "run_config_hash": sha256_file(RUN_CONFIG_PATH),
        "generation_source_hash": sha256_file(LLM / "prompts.py"),
        "prompts": prompt_rows,
    }
    atomic_write_json(GENERATED / "prompt_manifest.json", prompt_manifest)

    runs = []
    for model_key, model in run_config["models"].items():
        provider_config_hash = value_hash(model)
        for condition in run_config["conditions"]:
            method = model["logical_condition_ids"][condition]
            for target in manifest["targets"]:
                prompt = prompt_lookup[(int(target), condition)]
                mapping = next(item for item in mappings["targets"] if item["target_node"] == target)
                for run_id in range(run_config["repetitions"]):
                    runs.append({
                        "logical_run_key": f"{model_key}/{condition}/{target}/{run_id:02d}",
                        "method": method,
                        "model_key": model_key,
                        "condition": condition,
                        "target_node": int(target),
                        "run_id": run_id,
                        "prompt_hash": prompt["prompt_hash"],
                        "mapping_hash": mapping["mapping_hash"],
                        "protocol_hash": sha256_file(PROTOCOL_PATH),
                        "provider_config_hash": provider_config_hash,
                        "status": "pending",
                    })
    runs.sort(key=lambda item: (item["model_key"], item["condition"], item["target_node"], item["run_id"]))
    if len(runs) != 800 or len({item["logical_run_key"] for item in runs}) != 800:
        raise ValueError("Expected exactly 800 unique logical runs")
    run_manifest = {
        "schema_version": 1,
        "manifest_hash": materialized_hash,
        "protocol_hash": sha256_file(PROTOCOL_PATH),
        "run_config_hash": sha256_file(RUN_CONFIG_PATH),
        "source_hashes": _source_hashes(),
        "sorting": ["model_key", "condition", "target_node", "run_id"],
        "logical_run_count": len(runs),
        "runs": runs,
    }
    atomic_write_json(GENERATED / "run_manifest.json", run_manifest)
    return {
        "mappings": len(mappings["targets"]),
        "statistics_nodes": len(statistics["nodes"]),
        "prompts": len(prompt_rows),
        "logical_runs": len(runs),
    }
