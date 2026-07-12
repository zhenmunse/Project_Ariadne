"""Provider payload, capability, provenance and secret-isolation tests."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from experiments.llm.models import ProviderRequest
from experiments.llm.providers.base import ProviderConfigurationError
from experiments.llm.providers.closed_frontier import ClosedFrontierProvider
from experiments.llm.providers.open_weight import OpenWeightProvider


def request(model: str = "frozen-model") -> ProviderRequest:
    return ProviderRequest(
        logical_run_id="closed_frontier/zero/6/00",
        requested_model_id=model,
        system_prompt="system",
        user_prompt="user",
        reasoning_config={"effort": "high"},
        sampling_config={"temperature": None, "top_p": None},
        max_output_tokens=4096,
    )


class LLMProviderContractTests(unittest.TestCase):
    def test_unfrozen_formal_adapters_fail_capability_gate(self) -> None:
        for provider in (
            ClosedFrontierProvider(endpoint=None, requested_model_id=None, reasoning="high"),
            OpenWeightProvider(endpoint=None, requested_model_id=None, reasoning="high"),
        ):
            self.assertFalse(provider.capability_report().ready)
            with self.assertRaises(ProviderConfigurationError):
                provider.require_ready()

    def test_closed_payload_is_single_turn_toolless_and_stateless(self) -> None:
        provider = ClosedFrontierProvider(
            endpoint="https://provider.invalid/responses",
            requested_model_id="frozen-model",
            reasoning="high",
        )
        payload = provider.build_payload(request())
        self.assertEqual([item["role"] for item in payload["input"]], ["system", "user"])
        self.assertEqual(payload["tools"], [])
        self.assertFalse(payload["store"])
        self.assertIn("reasoning", payload)
        self.assertNotIn("previous_response_id", payload)
        self.assertNotIn("conversation", payload)

    def test_open_payload_is_single_turn_toolless_and_stateless(self) -> None:
        provider = OpenWeightProvider(
            endpoint="https://provider.invalid/chat/completions",
            requested_model_id="frozen-model",
            reasoning="high",
        )
        payload = provider.build_payload(request())
        self.assertEqual([item["role"] for item in payload["messages"]], ["system", "user"])
        self.assertNotIn("tools", payload)
        self.assertNotIn("previous_response_id", payload)
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_closed_response_preserves_requested_and_actual_model_metadata(self) -> None:
        captured = {}
        def transport(endpoint, headers, payload):
            captured.update({"endpoint": endpoint, "headers": headers, "payload": payload})
            return {
                "id": "req-1", "model": "actual-snapshot", "created_at": "2026-07-12T00:00:00Z",
                "status": "completed", "output_text": '{"sequence":[]}',
                "usage": {"input_tokens": 10, "output_tokens": 4, "output_tokens_details": {"reasoning_tokens": 2}},
            }
        provider = ClosedFrontierProvider(
            endpoint="https://provider.invalid/responses",
            requested_model_id="frozen-model",
            reasoning="high",
            transport=transport,
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "TOP-SECRET"}):
            response = provider.complete(request())
        self.assertEqual(response.requested_model_id, "frozen-model")
        self.assertEqual(response.response_model_id, "actual-snapshot")
        self.assertEqual(response.provider_request_id, "req-1")
        self.assertEqual(response.reasoning_tokens, 2)
        self.assertNotIn("TOP-SECRET", json.dumps(response.to_dict()))
        self.assertNotIn("TOP-SECRET", json.dumps(captured["payload"]))

    def test_open_response_preserves_raw_payload_and_usage(self) -> None:
        raw = {
            "id": "deep-1", "model": "actual-open-snapshot", "created": 1,
            "choices": [{"message": {"content": '{"sequence":[]}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 1}},
        }
        provider = OpenWeightProvider(
            endpoint="https://provider.invalid/chat/completions",
            requested_model_id="frozen-model",
            reasoning="high",
            transport=lambda endpoint, headers, payload: raw,
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "TOP-SECRET"}):
            response = provider.complete(request())
        self.assertEqual(response.response_model_id, "actual-open-snapshot")
        self.assertEqual(response.raw_provider_payload, raw)
        self.assertEqual(response.input_tokens, 7)
        self.assertEqual(response.output_tokens, 3)
        self.assertNotIn("TOP-SECRET", json.dumps(response.to_dict()))


if __name__ == "__main__":
    unittest.main()
