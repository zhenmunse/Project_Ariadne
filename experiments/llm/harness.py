"""Idempotent request execution, artifact persistence, parsing and recovery."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.llm.artifacts import atomic_write_json, load_json, sha256_file
from experiments.llm.models import ProviderRequest, ProviderResponse
from experiments.llm.parse_output import parse_output
from experiments.llm.providers.base import LLMProvider, TransportError
from experiments.llm.validate_sequence import validate_sequence


ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "experiments" / "llm"
PROTOCOL_PATH = LLM / "protocol.json"
RUN_CONFIG_PATH = LLM / "run_config.json"
MAPPINGS_PATH = LLM / "generated" / "mappings.json"
STATISTICS_PATH = LLM / "generated" / "full_statistics.json"

SENSITIVE_KEYS = {
    "authorization", "api_key", "apikey", "cookie", "set-cookie",
    "x-api-key", "proxy-authorization",
}


class AmbiguousRunState(RuntimeError):
    """A request may have reached the provider but no raw response is durable."""


class SimulatedCrash(RuntimeError):
    pass


def repository_commit_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def assert_secret_free(value: Any, path: str = "artifact") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                raise ValueError(f"Sensitive field is forbidden in {path}: {key}")
            assert_secret_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_secret_free(item, f"{path}[{index}]")


def _write(path: Path, value: dict[str, Any]) -> None:
    assert_secret_free(value)
    atomic_write_json(path, value)


def _paths(root: Path, logical_key: str, attempt: int) -> tuple[Path, Path, Path]:
    suffix = f"{attempt:03d}.json"
    return (
        root / "requests" / logical_key / suffix,
        root / "raw" / logical_key / suffix,
        root / "parsed" / logical_key / suffix,
    )


def _attempt_numbers(root: Path, logical_key: str) -> list[int]:
    numbers = set()
    for kind in ("requests", "raw", "parsed"):
        directory = root / kind / logical_key
        if directory.exists():
            numbers.update(int(path.stem) for path in directory.glob("*.json"))
    return sorted(numbers)


def _provenance(run: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
    adapter_path = Path(__import__(provider.__class__.__module__, fromlist=["x"]).__file__)
    return {
        "run_identity": {
            "method": run["method"],
            "model_key": run["model_key"],
            "condition": run["condition"],
            "target_node": run["target_node"],
            "run_id": run["run_id"],
            "logical_run_key": run["logical_run_key"],
        },
        "repository_commit_sha": repository_commit_sha(),
        "protocol_hash": sha256_file(PROTOCOL_PATH),
        "run_config_hash": sha256_file(RUN_CONFIG_PATH),
        "manifest_hash": load_json(LLM / "generated" / "run_manifest.json")["manifest_hash"],
        "dag_hash": load_json(MAPPINGS_PATH)["dag_hash"],
        "mapping_hash": run["mapping_hash"],
        "prompt_hash": run["prompt_hash"],
        "provider_config_hash": run["provider_config_hash"],
        "statistics_hash": sha256_file(STATISTICS_PATH) if run["condition"] == "full" else None,
        "provider_adapter_source_hash": sha256_file(adapter_path),
        "harness_source_hash": sha256_file(Path(__file__)),
        "parser_source_hash": sha256_file(LLM / "parse_output.py"),
        "validator_source_hash": sha256_file(LLM / "validate_sequence.py"),
    }


def _parse_raw(
    *,
    run: dict[str, Any],
    raw_path: Path,
    parsed_path: Path,
    bundle: dict[str, Any],
    provenance: dict[str, Any],
    selected_for_analysis: bool,
) -> dict[str, Any]:
    raw = load_json(raw_path)
    response = raw["provider_response"]
    parsed = parse_output(response["response_text"])
    validation = validate_sequence(parsed, bundle)
    artifact = {
        "schema_version": 1,
        "logical_run_key": run["logical_run_key"],
        "attempt": int(raw["attempt"]),
        "raw_response_path": (
            f"raw/{run['logical_run_key']}/{int(raw['attempt']):03d}.json"
        ),
        "raw_response_hash": sha256_file(raw_path),
        "parse_result": parsed.to_dict(),
        "sequence_validation": validation.to_dict(),
        "selected_for_analysis": selected_for_analysis,
        "provenance": provenance,
    }
    _write(parsed_path, artifact)
    return artifact


def execute_run(
    run: dict[str, Any],
    provider: LLMProvider,
    *,
    output_root: str | Path,
    force_rerun: bool = False,
    max_transport_attempts: int = 3,
    crash_at: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    numbers = _attempt_numbers(output_root, run["logical_run_key"])
    mappings = load_json(MAPPINGS_PATH)
    bundle = next(item for item in mappings["targets"] if item["target_node"] == run["target_node"])
    provenance = _provenance(run, provider)
    prior_transport_attempts = 0

    if not force_rerun:
        for attempt in numbers:
            request_path, raw_path, parsed_path = _paths(output_root, run["logical_run_key"], attempt)
            if parsed_path.exists():
                return {"status": "skipped_completed", "attempt": attempt, "parsed": load_json(parsed_path)}
            if raw_path.exists():
                parsed = _parse_raw(
                    run=run,
                    raw_path=raw_path,
                    parsed_path=parsed_path,
                    bundle=bundle,
                    provenance=provenance,
                    selected_for_analysis=not any(
                        _paths(output_root, run["logical_run_key"], earlier)[1].exists()
                        for earlier in numbers if earlier < attempt
                    ),
                )
                return {"status": "recovered_parse", "attempt": attempt, "parsed": parsed}
            if request_path.exists():
                request_artifact = load_json(request_path)
                if request_artifact.get("status") == "transport_error":
                    prior_transport_attempts += 1
                    transport_error = request_artifact.get("transport_error") or {}
                    if not bool(transport_error.get("retryable")):
                        return {
                            "status": "transport_failed_nonretryable",
                            "attempt": attempt,
                        }
                    continue
                else:
                    raise AmbiguousRunState(
                        f"Request {run['logical_run_key']} attempt {attempt} may have reached provider"
                    )
        if prior_transport_attempts >= max_transport_attempts:
            return {
                "status": "transport_failed_exhausted",
                "attempt": max(numbers),
            }

    attempt = (max(numbers) + 1) if numbers else 0
    prompt_path = LLM / "generated" / "prompts" / str(run["target_node"]) / f"{run['condition']}.json"
    prompt = load_json(prompt_path)
    if prompt["prompt_hash"] != run["prompt_hash"]:
        raise ValueError("Run manifest/prompt hash mismatch")
    config = load_json(RUN_CONFIG_PATH)
    model = config["models"][run["model_key"]]
    requested_model = model["requested_model_id"] or "mock-model-request"
    request = ProviderRequest(
        logical_run_id=run["logical_run_key"],
        requested_model_id=requested_model,
        system_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        reasoning_config={"effort": model["reasoning"]},
        sampling_config={"temperature": model["temperature"], "top_p": model["top_p"]},
        max_output_tokens=int(model["max_output_tokens"]),
        tools_disabled=True,
    )
    successful_raw_already_exists = any(
        _paths(output_root, run["logical_run_key"], prior)[1].exists() for prior in numbers
    )

    transport_count = 0 if force_rerun else prior_transport_attempts
    while transport_count < max_transport_attempts:
        request_path, raw_path, parsed_path = _paths(output_root, run["logical_run_key"], attempt)
        request_artifact = {
            "schema_version": 1,
            "logical_run_key": run["logical_run_key"],
            "attempt": attempt,
            "status": "request_prepared",
            "request_attempt_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": provider.provider_name,
            "request": request.to_dict(),
            "provenance": provenance,
        }
        if crash_at == "before_request":
            raise SimulatedCrash("before_request")
        _write(request_path, request_artifact)
        request_artifact["status"] = "request_dispatched"
        _write(request_path, request_artifact)
        try:
            response: ProviderResponse = provider.complete(request)
        except TransportError as error:
            request_artifact.update({
                "status": "transport_error",
                "transport_error": {
                    "message": str(error),
                    "retryable": error.retryable,
                    "status_code": error.status_code,
                },
            })
            _write(request_path, request_artifact)
            transport_count += 1
            if not error.retryable or transport_count >= max_transport_attempts:
                return {"status": "transport_failed", "attempt": attempt}
            attempt += 1
            continue
        if crash_at == "after_provider_before_raw":
            raise SimulatedCrash("after_provider_before_raw")
        raw_artifact = {
            "schema_version": 1,
            "logical_run_key": run["logical_run_key"],
            "attempt": attempt,
            "request_artifact_path": (
                f"requests/{run['logical_run_key']}/{attempt:03d}.json"
            ),
            "request_artifact_hash": sha256_file(request_path),
            "provider_response": response.to_dict(),
            "provenance": provenance,
        }
        _write(raw_path, raw_artifact)
        if crash_at == "after_raw_before_parse":
            raise SimulatedCrash("after_raw_before_parse")
        parsed = _parse_raw(
            run=run,
            raw_path=raw_path,
            parsed_path=parsed_path,
            bundle=bundle,
            provenance=provenance,
            selected_for_analysis=not successful_raw_already_exists,
        )
        if crash_at == "after_parse_before_status":
            raise SimulatedCrash("after_parse_before_status")
        return {"status": "completed", "attempt": attempt, "parsed": parsed}
    raise AssertionError("Unreachable transport loop")


def select_runs(
    runs: list[dict[str, Any]],
    *,
    model: str | None = None,
    condition: str | None = None,
    target: int | None = None,
    run_id: int | None = None,
) -> list[dict[str, Any]]:
    return [
        run for run in runs
        if (model is None or run["model_key"] == model)
        and (condition is None or run["condition"] == condition)
        and (target is None or run["target_node"] == target)
        and (run_id is None or run["run_id"] == run_id)
    ]
