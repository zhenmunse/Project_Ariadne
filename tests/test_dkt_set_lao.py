"""End-to-end tests for DKT-derived Set Oracle + LAO*."""

from __future__ import annotations

import csv
import importlib
import json
import unittest
from pathlib import Path

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "dkt_set"
GREEDY = ROOT / "results" / "dkt_set_greedy"
LAO = ROOT / "results" / "dkt_set_lao"
COMPARISON = ROOT / "results" / "dkt_set" / "planner_comparison.csv"
SUMMARY = ROOT / "results" / "dkt_set" / "task13_summary.json"
RUNNER = importlib.import_module("experiments.16_run_dkt_set_lao")


class DKTSetLAOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.greedy = read_jsonl(GREEDY / "sequences.jsonl")
        cls.lao = read_jsonl(LAO / "sequences.jsonl")
        with (ARTIFACTS / "surrogate_config.json").open(encoding="utf-8") as file:
            cls.config = json.load(file)

    def test_both_conditions_use_the_same_frozen_checkpoint(self) -> None:
        actual = sha256_file(ARTIFACTS / "surrogate_checkpoint.pt")
        self.assertEqual(actual, RUNNER.EXPECTED_SURROGATE_CHECKPOINT_HASH)
        self.assertEqual(
            {record.metadata["surrogate_checkpoint_hash"] for record in self.greedy + self.lao},
            {actual},
        )
        self.assertEqual(
            subprocess_attribute("eol", "artifacts/dkt_set/surrogate_checkpoint.pt"),
            "lf",
        )

    def test_lao_records_are_complete_and_provenance_matched(self) -> None:
        self.assertEqual(len(self.lao), 10)
        self.assertEqual({record.target_node for record in self.lao}, set(self.manifest["targets"]))
        closures = {c["target_node"]: c for c in self.manifest["closures"]}
        for record in self.lao:
            closure = closures[record.target_node]
            self.assertIs(record.method, Method.DKT_SET_LAO)
            self.assertEqual(record.run_id, 0)
            self.assertEqual(set(record.sequence), set(closure["sequence_nodes"]))
            self.assertEqual(record.sequence[-1], record.target_node)
            self.assertEqual(record.metadata["manifest_hash"], manifest_hash(self.manifest))
            self.assertEqual(record.metadata["closure_hash"], closure["closure_hash"])
            self.assertTrue(record.metadata["converged"])

    def test_reference_loader_validates_full_dkt_identity(self) -> None:
        oracle, config = RUNNER.load_frozen_oracle()
        reference = RUNNER.load_greedy_reference(self.manifest, config, oracle)
        expected = RUNNER.reference_metadata(config, oracle)
        self.assertEqual(set(reference), set(self.manifest["targets"]))
        for record in reference.values():
            for key, value in expected.items():
                self.assertEqual(record.metadata[key], value)

    def test_lao_matches_dp_and_never_exceeds_greedy(self) -> None:
        greedy = {record.target_node: float(record.internal_cost) for record in self.greedy}
        for record in self.lao:
            cost = float(record.internal_cost)
            self.assertLess(abs(cost - float(record.metadata["exact_dp_cost"])), RUNNER.TOLERANCE)
            self.assertLessEqual(cost, greedy[record.target_node] + RUNNER.TOLERANCE)

    def test_independent_oracle_planning_signatures_are_identical(self) -> None:
        first_oracle, first_config = RUNNER.load_frozen_oracle()
        second_oracle, second_config = RUNNER.load_frozen_oracle()
        first_ref = RUNNER.load_greedy_reference(self.manifest, first_config, first_oracle)
        second_ref = RUNNER.load_greedy_reference(self.manifest, second_config, second_oracle)
        first, _ = RUNNER.generate_records(self.manifest, first_oracle, first_config, first_ref)
        second, _ = RUNNER.generate_records(self.manifest, second_oracle, second_config, second_ref)
        self.assertEqual(RUNNER.signature(first), RUNNER.signature(second))
        self.assertEqual(RUNNER.signature(first), RUNNER.signature(self.lao))

    def test_public_evaluator_and_comparison_outputs(self) -> None:
        self.assertTrue(all(result.valid for result in SequenceEvaluator.from_artifacts().score_records(self.lao)))
        with (LAO / "scored_sequences.csv").open(encoding="utf-8", newline="") as file:
            scored = list(csv.DictReader(file))
        with COMPARISON.open(encoding="utf-8", newline="") as file:
            comparison = list(csv.DictReader(file))
        self.assertEqual(len(scored), 10)
        self.assertEqual(len(comparison), 10)
        self.assertTrue(all(row["valid"] == "True" for row in scored))
        self.assertTrue(all(float(row["lao_dp_absolute_gap"]) < RUNNER.TOLERANCE for row in comparison))

    def test_generated_task13_summary_closes_acceptance_contract(self) -> None:
        with SUMMARY.open(encoding="utf-8") as file:
            summary = json.load(file)
        self.assertEqual(summary["status"], "go")
        self.assertEqual(summary["condition_name"], "DKT-derived Set Oracle")
        self.assertEqual(summary["greedy_valid_targets"], 10)
        self.assertEqual(summary["lao_valid_targets"], 10)
        self.assertEqual(summary["lao_dp_max_gap"], 0.0)
        self.assertLessEqual(summary["lao_minus_greedy_max_gap"], RUNNER.TOLERANCE)
        self.assertEqual(summary["lao_improved_targets"], 8)
        self.assertEqual(
            summary["surrogate_checkpoint_hash"],
            sha256_file(ARTIFACTS / "surrogate_checkpoint.pt"),
        )


def subprocess_attribute(attribute: str, path: str) -> str:
    import subprocess

    output = subprocess.run(
        ["git", "check-attr", attribute, "--", path],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return output.rsplit(": ", 1)[-1]


if __name__ == "__main__":
    unittest.main()
