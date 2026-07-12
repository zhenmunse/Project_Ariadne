"""Open-weight chat-completions-style adapter with native reasoning config."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from experiments.llm.models import ProviderRequest, ProviderResponse
from experiments.llm.providers.base import CapabilityReport, ProviderConfigurationError
from experiments.llm.providers.http_base import JSONTransport, authorization_headers, urllib_json_transport


class OpenWeightProvider:
    provider_name = "open_weight"

    def __init__(
        self,
        *,
        endpoint: str | None,
        requested_model_id: str | None,
        reasoning: str | None,
        thinking_enabled: bool = True,
        api_key_env: str = "DEEPSEEK_API_KEY",
        transport: JSONTransport = urllib_json_transport,
    ) -> None:
        self.endpoint = endpoint
        self.requested_model_id = requested_model_id
        self.reasoning = reasoning
        self.thinking_enabled = thinking_enabled
        self.api_key_env = api_key_env
        self.transport = transport

    def capability_report(self) -> CapabilityReport:
        return CapabilityReport(True, False, False, bool(self.reasoning), bool(self.requested_model_id), bool(self.endpoint))

    def require_ready(self) -> None:
        if not self.capability_report().ready:
            raise ProviderConfigurationError("Open-weight provider is not frozen and capability-ready")

    def build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        if request.requested_model_id != self.requested_model_id:
            raise ProviderConfigurationError("Request model ID differs from frozen adapter model ID")
        payload: dict[str, Any] = {
            "model": request.requested_model_id,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "reasoning_effort": request.reasoning_config.get("effort"),
            "thinking": {"type": "enabled" if self.thinking_enabled else "disabled"},
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        payload.update({key: value for key, value in request.sampling_config.items() if value is not None})
        return payload

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.require_ready()
        started = time.perf_counter()
        payload = self.transport(self.endpoint or "", authorization_headers(self.api_key_env), self.build_payload(request))
        latency = time.perf_counter() - started
        choices = payload.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        usage = payload.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        created = payload.get("created")
        created_at = datetime.fromtimestamp(created, timezone.utc).isoformat() if isinstance(created, (int, float)) else datetime.now(timezone.utc).isoformat()
        return ProviderResponse(
            response_text=str(message.get("content") or ""),
            requested_model_id=request.requested_model_id,
            response_model_id=str(payload.get("model") or ""),
            provider_request_id=str(payload.get("id") or ""),
            created_at_utc=created_at,
            finish_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            latency_seconds=latency,
            raw_provider_payload=payload,
        )
