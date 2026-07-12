"""Run one explicitly excluded provider smoke test outside the formal run grid."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.llm.artifacts import atomic_write_json, load_json, sha256_file
from experiments.llm.harness import assert_secret_free
from experiments.llm.models import ProviderRequest
from experiments.llm.parse_output import parse_output
from experiments.llm.providers.mock import MockProvider
from experiments.llm.providers.factory import build_provider
from experiments.llm.validate_sequence import validate_sequence


LLM = ROOT / "experiments" / "llm"


def run_smoke(
    provider_name: str,
    provider,
    *,
    model_key: str,
    output_root: Path,
    timestamp_utc: str | None = None,
    generic_preflight: bool = False,
) -> Path:
    config = load_json(LLM / "run_config.json")
    model = config["models"][model_key]
    target = 6
    condition = "zero"
    prompt_path = LLM / "generated" / "prompts" / str(target) / f"{condition}.json"
    prompt = load_json(prompt_path)
    timestamp = timestamp_utc or datetime.now(timezone.utc).isoformat()
    safe_timestamp = timestamp.replace(":", "-").replace("+", "_")
    logical_id = f"smoke/{provider_name}/{safe_timestamp}"
    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]
    if generic_preflight:
        system_prompt = "Return only the exact JSON object requested by the user."
        user_prompt = 'Return exactly {"sequence": []} and no other text.'
    request = ProviderRequest(
        logical_run_id=logical_id,
        requested_model_id=(model["requested_model_id"] if provider_name != "mock" else "mock-model-request"),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning_config={"effort": model["reasoning"]},
        sampling_config={"temperature": model["temperature"], "top_p": model["top_p"]},
        max_output_tokens=int(model["max_output_tokens"]),
        tools_disabled=True,
    )
    response = provider.complete(request)
    parsed = parse_output(response.response_text)
    mappings = load_json(LLM / "generated" / "mappings.json")
    bundle = next(item for item in mappings["targets"] if item["target_node"] == target)
    validation = None if generic_preflight else validate_sequence(parsed, bundle)
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "smoke_test": True,
        "excluded_from_analysis": True,
        "prompt_scope": (
            "generic_provider_preflight" if generic_preflight else "frozen_curriculum_smoke"
        ),
        "logical_smoke_key": logical_id,
        "formal_run_id": None,
        "provider": provider_name,
        "model_key": model_key,
        "target_node": target,
        "condition": condition,
        "request": request.to_dict(),
        "provider_response": response.to_dict(),
        "parse_result": parsed.to_dict(),
        "sequence_validation": validation.to_dict() if validation is not None else None,
        "protocol_hash": sha256_file(LLM / "protocol.json"),
        "run_config_hash": sha256_file(LLM / "run_config.json"),
        "prompt_hash": prompt["prompt_hash"],
        "mapping_hash": bundle["mapping_hash"],
        "smoke_runner_source_hash": sha256_file(Path(__file__)),
    }
    assert_secret_free(artifact)
    output = output_root / provider_name / f"{safe_timestamp}.json"
    atomic_write_json(output, artifact)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("closed_frontier", "open_weight"), required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "llm" / "smoke")
    parser.add_argument("--generic-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(LLM / "run_config.json")
    provider = build_provider(args.provider, config, "valid")
    output = run_smoke(
        args.provider,
        provider,
        model_key=args.provider,
        output_root=args.output_root,
        generic_preflight=args.generic_preflight,
    )
    artifact = load_json(output)
    print(json.dumps({
        "smoke_test": artifact["smoke_test"],
        "excluded_from_analysis": artifact["excluded_from_analysis"],
        "provider": artifact["provider"],
        "response_model_id": artifact["provider_response"]["response_model_id"],
        "provider_request_id_present": bool(artifact["provider_response"]["provider_request_id"]),
        "finish_reason": artifact["provider_response"]["finish_reason"],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
