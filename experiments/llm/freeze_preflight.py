"""Create a sanitized machine-readable Task 17 preflight record from smoke logs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.llm.artifacts import atomic_write_json, load_json, sha256_file, value_hash


LLM = ROOT / "experiments" / "llm"
SMOKE_ROOT = ROOT / "results" / "llm" / "smoke"
OUTPUT = LLM / "generated" / "provider_preflight.json"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _latest_verified(provider: str) -> dict | None:
    directory = SMOKE_ROOT / provider
    candidates = sorted(directory.glob("*.json")) if directory.exists() else []
    for path in reversed(candidates):
        artifact = load_json(path)
        response = artifact.get("provider_response") or {}
        if (
            artifact.get("smoke_test") is True
            and artifact.get("excluded_from_analysis") is True
            and artifact.get("prompt_scope") == "generic_provider_preflight"
            and response.get("response_model_id")
            and response.get("provider_request_id")
            and response.get("finish_reason")
            and response.get("raw_provider_payload")
        ):
            return {"path": path, "artifact": artifact}
    return None


def build_preflight() -> dict:
    config_path = LLM / "run_config.json"
    prompt_manifest_path = LLM / "generated" / "prompt_manifest.json"
    run_manifest_path = LLM / "generated" / "run_manifest.json"
    config = load_json(config_path)
    providers = {}
    for model_key in ("closed_frontier", "open_weight"):
        model = config["models"][model_key]
        verified = _latest_verified(model_key)
        row = {
            "provider": model["provider"],
            "endpoint": model["endpoint"],
            "requested_model_id": model["requested_model_id"],
            "reasoning": model["reasoning"],
            "temperature": model["temperature"],
            "top_p": model["top_p"],
            "max_output_tokens": model["max_output_tokens"],
            "model_id_status": model["model_id_status"],
            "smoke_verified": verified is not None,
        }
        if verified is not None:
            path = verified["path"]
            artifact = verified["artifact"]
            response = artifact["provider_response"]
            row["smoke"] = {
                "smoke_test": True,
                "excluded_from_analysis": True,
                "prompt_scope": artifact["prompt_scope"],
                "restricted_artifact_path": path.relative_to(ROOT).as_posix(),
                "restricted_artifact_sha256": sha256_file(path),
                "requested_model_id": response["requested_model_id"],
                "response_model_id": response["response_model_id"],
                "provider_request_id_present": True,
                "provider_request_id_sha256": _hash_text(response["provider_request_id"]),
                "created_at_utc": response["created_at_utc"],
                "finish_reason": response["finish_reason"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "reasoning_tokens": response["reasoning_tokens"],
                "raw_provider_payload_present": True,
                "raw_provider_payload_hash": value_hash(response["raw_provider_payload"]),
                "response_text_hash": _hash_text(response["response_text"]),
                "parse_valid": artifact["parse_result"]["parse_valid"],
                "schema_valid": artifact["parse_result"]["schema_valid"],
            }
        providers[model_key] = row
    return {
        "schema_version": 1,
        "providers": providers,
        "formal_execution_ready": all(row["smoke_verified"] for row in providers.values()),
        "run_config_hash": sha256_file(config_path),
        "prompt_manifest_hash": sha256_file(prompt_manifest_path),
        "run_manifest_hash": sha256_file(run_manifest_path),
        "protocol_hash": sha256_file(LLM / "protocol.json"),
        "freeze_script_hash": sha256_file(Path(__file__)),
        "official_documentation": {
            "closed_frontier": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
            "open_weight": "https://api-docs.deepseek.com/guides/thinking_mode",
        },
    }


def main() -> None:
    artifact = build_preflight()
    atomic_write_json(OUTPUT, artifact)
    print(json.dumps({
        "formal_execution_ready": artifact["formal_execution_ready"],
        "closed_frontier_smoke_verified": artifact["providers"]["closed_frontier"]["smoke_verified"],
        "open_weight_smoke_verified": artifact["providers"]["open_weight"]["smoke_verified"],
        "output": str(OUTPUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
