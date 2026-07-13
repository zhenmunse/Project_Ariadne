"""Contract tests for the frozen Task 15 LLM objective and input fields."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "llm" / "protocol.json"


class LLMProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with PROTOCOL_PATH.open(encoding="utf-8") as file:
            cls.protocol = json.load(file)

    def test_base_cost_is_uniform_sixty(self) -> None:
        self.assertEqual(self.protocol["base_cost"], 60.0)
        self.assertTrue(self.protocol["cost_semantics"]["uniform_for_all_concepts"])
        self.assertFalse(self.protocol["cost_semantics"]["empirical_duration"])

    def test_full_additional_fields_are_exactly_count_and_rate(self) -> None:
        self.assertEqual(
            self.protocol["conditions"]["full"]["additional_fields"],
            ["attempt_count", "success_rate"],
        )
        self.assertEqual(self.protocol["conditions"]["zero"]["additional_fields"], [])

    def test_completion_time_is_not_an_allowed_field(self) -> None:
        allowed = set()
        for condition in self.protocol["conditions"].values():
            allowed.update(condition["common_fields"])
            allowed.update(condition["additional_fields"])
        self.assertNotIn("median_completion_time", allowed)
        self.assertNotIn("mean_completion_time", allowed)
        self.assertNotIn("empirical_per_concept_duration", allowed)
        self.assertTrue(
            {"median_completion_time", "mean_completion_time"}.issubset(
                self.protocol["excluded_student_fields"]
            )
        )

    def test_zero_and_full_share_one_objective(self) -> None:
        objective = self.protocol["objective"]
        self.assertEqual(objective["id"], "expected_uniform_cost_attempts")
        self.assertEqual(set(objective["shared_by_conditions"]), {"zero", "full"})

    def test_success_rate_population_threshold_and_missing_value_are_frozen(self) -> None:
        statistics = self.protocol["statistics"]
        self.assertEqual(statistics["attempt_count"]["split"], "train")
        self.assertEqual(statistics["attempt_count"]["unit"], "canonical_concept_session")
        self.assertEqual(statistics["success_rate"]["correctness_threshold"], 0.8)
        self.assertEqual(statistics["success_rate"]["denominator"], "attempt_count")
        self.assertEqual(statistics["success_rate"]["render_decimal_places"], 4)
        self.assertEqual(statistics["zero_observation"]["attempt_count"], 0)
        self.assertIsNone(statistics["zero_observation"]["success_rate"])
        self.assertFalse(statistics["zero_observation"]["imputation_allowed"])

    def test_uniform_cost_applies_to_planners_and_public_evaluator(self) -> None:
        consumers = set(self.protocol["uniform_cost_applies_to"])
        self.assertIn("all_planners", consumers)
        self.assertIn("public_evaluator", consumers)


if __name__ == "__main__":
    unittest.main()
