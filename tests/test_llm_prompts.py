"""Zero/Full prompt equivalence and leakage controls."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from experiments.llm.artifacts import sha256_file, value_hash
from experiments.llm.prompts import OBJECTIVE, OUTPUT_CONTRACT


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "experiments/llm/generated"


class LLMPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((GENERATED / "prompt_manifest.json").read_text(encoding="utf-8"))

    def _prompt(self, target: int, condition: str) -> dict:
        return json.loads((GENERATED / "prompts" / str(target) / f"{condition}.json").read_text(encoding="utf-8"))

    def test_twenty_prompts_and_artifact_hashes(self) -> None:
        self.assertEqual(len(self.manifest["prompts"]), 20)
        for row in self.manifest["prompts"]:
            self.assertEqual(row["artifact_hash"], sha256_file(ROOT / row["path"]))

    def test_zero_full_share_mapping_curriculum_objective_and_output(self) -> None:
        for target in (6, 7, 12, 18, 29, 36, 39, 42, 46, 52):
            zero = self._prompt(target, "zero")
            full = self._prompt(target, "full")
            self.assertEqual(zero["mapping_hash"], full["mapping_hash"])
            self.assertEqual(zero["shared_curriculum_hash"], full["shared_curriculum_hash"])
            self.assertEqual(zero["system_prompt_hash"], full["system_prompt_hash"])
            self.assertIn(OBJECTIVE, zero["user_prompt"])
            self.assertIn(OBJECTIVE, full["user_prompt"])
            self.assertIn(OUTPUT_CONTRACT, zero["user_prompt"])
            self.assertIn(OUTPUT_CONTRACT, full["user_prompt"])

    def test_zero_has_no_statistics_and_full_has_only_locked_fields(self) -> None:
        for target in (6, 7, 12, 18, 29, 36, 39, 42, 46, 52):
            zero = self._prompt(target, "zero")["user_prompt"]
            full = self._prompt(target, "full")["user_prompt"]
            self.assertNotIn("attempt_count=", zero)
            self.assertNotIn("success_rate=", zero)
            self.assertIn("attempt_count=", full)
            self.assertIn("success_rate=", full)
            self.assertNotRegex(full.lower(), r"completion[_ -]?time|empirical[_ -]?duration")

    def test_rates_render_four_decimals_and_missing_is_null(self) -> None:
        full = self._prompt(42, "full")["user_prompt"]
        rendered = re.findall(r"success_rate=([^\n]+)", full)
        self.assertIn("null", rendered)
        self.assertTrue(all(value == "null" or re.fullmatch(r"\d\.\d{4}", value) for value in rendered))

    def test_no_forbidden_planner_or_evaluator_leakage(self) -> None:
        forbidden = ("ariadne", "lao*", "dynamic programming", "oracle", "evaluator")
        for row in self.manifest["prompts"]:
            prompt = self._prompt(row["target_node"], row["condition"])
            combined = (prompt["system_prompt"] + prompt["user_prompt"]).lower()
            self.assertTrue(all(term not in combined for term in forbidden))
            self.assertEqual(
                prompt["prompt_hash"],
                value_hash({"system_prompt": prompt["system_prompt"], "user_prompt": prompt["user_prompt"]}),
            )
