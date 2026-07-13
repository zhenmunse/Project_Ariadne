"""Typed records shared by deterministic LLM input and parsing stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderRequest:
    logical_run_id: str
    requested_model_id: str
    system_prompt: str
    user_prompt: str
    reasoning_config: dict[str, Any]
    sampling_config: dict[str, Any]
    max_output_tokens: int
    tools_disabled: bool = True

    def __post_init__(self) -> None:
        if not self.logical_run_id or not self.requested_model_id:
            raise ValueError("Provider request requires logical run and model IDs")
        if not self.tools_disabled:
            raise ValueError("LLM experiment provider requests must disable tools")

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_run_id": self.logical_run_id,
            "requested_model_id": self.requested_model_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "reasoning_config": self.reasoning_config,
            "sampling_config": self.sampling_config,
            "max_output_tokens": self.max_output_tokens,
            "tools_disabled": self.tools_disabled,
        }


@dataclass(frozen=True)
class ProviderResponse:
    response_text: str
    requested_model_id: str
    response_model_id: str
    provider_request_id: str
    created_at_utc: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    latency_seconds: float
    raw_provider_payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.requested_model_id or not self.response_model_id:
            raise ValueError("Provider response must preserve requested and actual model IDs")
        if not self.provider_request_id or not self.created_at_utc:
            raise ValueError("Provider response requires request ID and UTC timestamp")
        if not self.finish_reason:
            raise ValueError("Provider response requires a nonempty finish reason/status")
        if not isinstance(self.raw_provider_payload, dict) or not self.raw_provider_payload:
            raise ValueError("Provider response requires a nonempty raw provider payload")
        if self.latency_seconds < 0:
            raise ValueError("Provider latency must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_text": self.response_text,
            "requested_model_id": self.requested_model_id,
            "response_model_id": self.response_model_id,
            "provider_request_id": self.provider_request_id,
            "created_at_utc": self.created_at_utc,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "latency_seconds": self.latency_seconds,
            "raw_provider_payload": self.raw_provider_payload,
        }


@dataclass(frozen=True)
class ParseResult:
    parse_valid: bool
    schema_valid: bool
    opaque_sequence: tuple[str, ...] | None
    parse_error_code: str | None
    parse_error_detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_valid": self.parse_valid,
            "schema_valid": self.schema_valid,
            "opaque_sequence": (
                list(self.opaque_sequence) if self.opaque_sequence is not None else None
            ),
            "parse_error_code": self.parse_error_code,
            "parse_error_detail": self.parse_error_detail,
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    real_sequence: tuple[int, ...] | None
    error_code: str | None
    error_detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "real_sequence": (
                list(self.real_sequence) if self.real_sequence is not None else None
            ),
            "error_code": self.error_code,
            "error_detail": self.error_detail,
        }
