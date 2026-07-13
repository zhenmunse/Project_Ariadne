"""Sanitized Task 17 provider preflight manifest tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.llm.artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[1]
LLM = ROOT / "experiments/llm"
PREFLIGHT = LLM / "generated/provider_preflight.json"


class LLMPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))

    def test_deepseek_32768_inherits_completed_lower_ceiling_smoke(self) -> None:
        deepseek = self.preflight["providers"]["open_weight"]
        self.assertTrue(deepseek["smoke_verified"])
        self.assertEqual(deepseek["requested_model_id"], "deepseek-v4-pro")
        self.assertEqual(deepseek["model_id_status"], "smoke_verified")
        self.assertEqual(
            deepseek["smoke"]["verification_basis"],
            "inherited_nonbinding_ceiling_increase",
        )
        self.assertEqual(deepseek["smoke"]["verified_request_max_output_tokens"], 16384)

    def test_openai_32768_configuration_is_verified(self) -> None:
        openai = self.preflight["providers"]["closed_frontier"]
        self.assertTrue(openai["smoke_verified"])
        self.assertEqual(openai["requested_model_id"], "gpt-5.6-sol")
        self.assertEqual(openai["smoke"]["response_model_id"], "gpt-5.6-sol")
        self.assertTrue(openai["smoke"]["provider_request_id_present"])
        self.assertTrue(openai["smoke"]["raw_provider_payload_present"])
        self.assertEqual(openai["model_id_status"], "smoke_verified")
        self.assertTrue(self.preflight["formal_execution_ready"])

    def test_freeze_hashes_bind_current_machine_readable_inputs(self) -> None:
        self.assertEqual(self.preflight["run_config_hash"], sha256_file(LLM / "run_config.json"))
        self.assertEqual(
            self.preflight["prompt_manifest_hash"],
            sha256_file(LLM / "generated/prompt_manifest.json"),
        )
        self.assertEqual(
            self.preflight["run_manifest_hash"],
            sha256_file(LLM / "generated/run_manifest.json"),
        )


if __name__ == "__main__":
    unittest.main()
