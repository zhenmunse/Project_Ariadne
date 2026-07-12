"""Closed-frontier Responses-style adapter with no tools or prior response."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from experiments.llm.models import ProviderRequest, ProviderResponse
from experiments.llm.providers.base import CapabilityReport, ProviderConfigurationError
from experiments.llm.providers.http_base import JSONTransport, authorization_headers, urllib_json_transport


class ClosedFrontierProvider:
    provider_name = "closed_frontier"

    def __init__(
        self,
        *,
        endpoint: str | None,
        requested_model_id: str | None,
        reasoning: str | None,
        api_key_env: str = "OPENAI_API_KEY",
        transport: JSONTransport = urllib_json_transport,
    ) -> None:
        self.endpoint = endpoint
        self.requested_model_id = requested_model_id
        self.reasoning = reasoning
        self.api_key_env = api_key_env
        self.transport = transport

    def capability_report(self) -> CapabilityReport:
        return CapabilityReport(True, False, False, bool(self.reasoning), bool(self.requested_model_id), bool(self.endpoint))

    def require_ready(self) -> None:
        if not self.capability_report().ready:
            raise ProviderConfigurationError("Closed-frontier provider is not frozen and capability-ready")

    def build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        if request.requested_model_id != self.requested_model_id:
            raise ProviderConfigurationError("Request model ID differs from frozen adapter model ID")
        payload: dict[str, Any] = {
            "model": request.requested_model_id,
            "input": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "reasoning": request.reasoning_config,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
            "tools": [],
        }
        payload.update({key: value for key, value in request.sampling_config.items() if value is not None})
        return payload

    @staticmethod
    def _response_text(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        texts = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if isinstance(content.get("text"), str):
                    texts.append(content["text"])
        return "".join(texts)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.require_ready()
        started = time.perf_counter()
        payload = self.transport(self.endpoint or "", authorization_headers(self.api_key_env), self.build_payload(request))
        latency = time.perf_counter() - started
        usage = payload.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        return ProviderResponse(
            response_text=self._response_text(payload),
            requested_model_id=request.requested_model_id,
            response_model_id=str(payload.get("model") or ""),
            provider_request_id=str(payload.get("id") or ""),
            created_at_utc=str(payload.get("created_at") or datetime.now(timezone.utc).isoformat()),
            finish_reason=payload.get("status"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            latency_seconds=latency,
            raw_provider_payload=payload,
        )
