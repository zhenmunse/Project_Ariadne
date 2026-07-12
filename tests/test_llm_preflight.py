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

    def test_deepseek_smoke_is_verified_and_excluded(self) -> None:
        deepseek = self.preflight["providers"]["open_weight"]
        self.assertTrue(deepseek["smoke_verified"])
        self.assertEqual(deepseek["requested_model_id"], "deepseek-v4-pro")
        self.assertEqual(deepseek["smoke"]["response_model_id"], "deepseek-v4-pro")
        self.assertTrue(deepseek["smoke"]["smoke_test"])
        self.assertTrue(deepseek["smoke"]["excluded_from_analysis"])
        self.assertTrue(deepseek["smoke"]["provider_request_id_present"])
        self.assertTrue(deepseek["smoke"]["raw_provider_payload_present"])

    def test_openai_remains_explicitly_pending(self) -> None:
        self.assertFalse(self.preflight["providers"]["closed_frontier"]["smoke_verified"])
        self.assertFalse(self.preflight["formal_execution_ready"])

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
