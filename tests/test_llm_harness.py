"""Static 800-run matrix and deterministic Task 16-1 dry-run tests."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from experiments.llm.artifacts import sha256_file
from experiments.llm.prepare_inputs import prepare_inputs


ROOT = Path(__file__).resolve().parents[1]
LLM = ROOT / "experiments/llm"
GENERATED = LLM / "generated"


class LLMHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((LLM / "run_config.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((GENERATED / "run_manifest.json").read_text(encoding="utf-8"))

    def test_run_configuration_machine_locks_execution_policy(self) -> None:
        self.assertEqual(self.config["repetitions"], 20)
        self.assertEqual(self.config["conditions"], ["zero", "full"])
        methods = {
            value
            for model in self.config["models"].values()
            for value in model["logical_condition_ids"].values()
        }
        self.assertEqual(methods, {
            "gpt56_sol_zero", "gpt56_sol_full",
            "deepseek_v4_zero", "deepseek_v4_full",
        })
        for model in self.config["models"].values():
            self.assertIsNone(model["requested_model_id"])
            self.assertEqual(model["model_id_status"], "must_be_frozen_before_task17")
        policy = self.config["request_policy"]
        self.assertTrue(policy["single_turn"] and policy["fresh_request"])
        self.assertFalse(policy["tools"] or policy["search"] or policy["retrieval"])
        self.assertFalse(policy["code_execution"] or policy["conversation_memory"])
        self.assertFalse(policy["previous_response"] or policy["repair_invalid_output"])
        self.assertFalse(policy["experimental_retry_after_provider_response"])
        self.assertTrue(policy["transport_retry"]["allowed_before_provider_response"])

    def test_exactly_800_unique_logical_runs_and_balanced_cells(self) -> None:
        runs = self.manifest["runs"]
        self.assertEqual(len(runs), 800)
        self.assertEqual(len({run["logical_run_key"] for run in runs}), 800)
        model_condition = Counter((run["model_key"], run["condition"]) for run in runs)
        self.assertEqual(set(model_condition.values()), {200})
        cells = Counter((run["model_key"], run["condition"], run["target_node"]) for run in runs)
        self.assertEqual(len(cells), 40)
        self.assertEqual(set(cells.values()), {20})
        self.assertEqual({run["status"] for run in runs}, {"pending"})

    def test_zero_full_run_grids_and_mapping_are_aligned(self) -> None:
        runs = self.manifest["runs"]
        for model in self.config["models"]:
            zero = {(r["target_node"], r["run_id"]): r for r in runs if r["model_key"] == model and r["condition"] == "zero"}
            full = {(r["target_node"], r["run_id"]): r for r in runs if r["model_key"] == model and r["condition"] == "full"}
            self.assertEqual(set(zero), set(full))
            for identity in zero:
                self.assertEqual(zero[identity]["mapping_hash"], full[identity]["mapping_hash"])

    def test_dry_run_twice_is_byte_identical(self) -> None:
        paths = sorted(path for path in GENERATED.rglob("*.json"))
        before = {path.relative_to(GENERATED).as_posix(): sha256_file(path) for path in paths}
        first = prepare_inputs()
        middle = {path.relative_to(GENERATED).as_posix(): sha256_file(path) for path in paths}
        second = prepare_inputs()
        after = {path.relative_to(GENERATED).as_posix(): sha256_file(path) for path in paths}
        self.assertEqual(first, second)
        self.assertEqual(before, middle)
        self.assertEqual(middle, after)
        self.assertEqual(first["logical_runs"], 800)

    def test_no_formal_response_directories_are_created(self) -> None:
        results = ROOT / "results/llm"
        self.assertFalse((results / "raw").exists())
        self.assertFalse((results / "parsed").exists())
        self.assertFalse((results / "requests").exists())

    def test_provider_provenance_requirements_are_machine_locked(self) -> None:
        required = set(self.config["provider_response_provenance_required"])
        self.assertTrue({
            "requested_model_id", "response_model_id", "provider_request_id",
            "created_at_utc", "finish_reason", "raw_provider_payload",
        }.issubset(required))
        aggregation = self.config["aggregation"]
        self.assertEqual(aggregation["validity_denominator"], "all_requested_runs")
        self.assertEqual(aggregation["cost_conditioning"], "valid_runs_only")
        self.assertEqual(aggregation["target_weighting"], "equal")


if __name__ == "__main__":
    unittest.main()
