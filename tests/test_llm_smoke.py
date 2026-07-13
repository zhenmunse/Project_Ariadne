"""Excluded smoke-test identity and metadata contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.llm.artifacts import load_json
from experiments.llm.providers.mock import MockProvider
from experiments.llm.smoke_test import run_smoke


class LLMSmokeTests(unittest.TestCase):
    def test_mock_smoke_is_excluded_and_uses_no_formal_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = run_smoke(
                "mock",
                MockProvider(),
                model_key="closed_frontier",
                output_root=Path(directory),
                timestamp_utc="2000-01-01T00:00:00+00:00",
            )
            artifact = load_json(output)
        self.assertTrue(artifact["smoke_test"])
        self.assertTrue(artifact["excluded_from_analysis"])
        self.assertIsNone(artifact["formal_run_id"])
        self.assertTrue(artifact["logical_smoke_key"].startswith("smoke/mock/"))
        self.assertEqual(artifact["provider_response"]["response_model_id"], "mock-model-v1")
        self.assertTrue(artifact["provider_response"]["provider_request_id"])
        self.assertTrue(artifact["provider_response"]["finish_reason"])
        self.assertTrue(artifact["provider_response"]["raw_provider_payload"])

    def test_generic_preflight_contains_no_workspace_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = run_smoke(
                "mock",
                MockProvider(fixture="empty_response"),
                model_key="open_weight",
                output_root=Path(directory),
                timestamp_utc="2000-01-01T00:00:00+00:00",
                generic_preflight=True,
            )
            artifact = load_json(output)
        self.assertEqual(artifact["prompt_scope"], "generic_provider_preflight")
        self.assertEqual(artifact["request"]["user_prompt"], 'Return exactly {"sequence": []} and no other text.')
        self.assertNotIn("Prerequisite", artifact["request"]["user_prompt"])
        self.assertIsNone(artifact["sequence_validation"])

    def test_curriculum_smoke_target_and_condition_are_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = run_smoke(
                "mock",
                MockProvider(),
                model_key="open_weight",
                output_root=Path(directory),
                timestamp_utc="2000-01-01T00:00:00+00:00",
                target=39,
                condition="full",
            )
            artifact = load_json(output)
        self.assertEqual(artifact["target_node"], 39)
        self.assertEqual(artifact["condition"], "full")
        self.assertTrue(artifact["sequence_validation"]["valid"])


if __name__ == "__main__":
    unittest.main()
