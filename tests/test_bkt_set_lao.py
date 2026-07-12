"""End-to-end checks for BKT-derived Set Oracle + LAO*."""

from __future__ import annotations

import csv
import importlib
import json
import struct
import subprocess
import unittest
from pathlib import Path

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "bkt_set"
LAO_RESULTS = ROOT / "results" / "bkt_set_lao"
GREEDY_RESULTS = ROOT / "results" / "bkt_set_greedy"
COMPARISON_PATH = ROOT / "results" / "bkt_set" / "planner_comparison.csv"
RUNNER = importlib.import_module("experiments.14_run_bkt_set_lao")


def packed(value: float) -> bytes:
    return struct.pack("!d", value)


class BKTSetLAOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.records = read_jsonl(LAO_RESULTS / "sequences.jsonl")
        cls.greedy = read_jsonl(GREEDY_RESULTS / "sequences.jsonl")
        with (ARTIFACTS / "surrogate_config.json").open(encoding="utf-8") as file:
            cls.config = json.load(file)

    def test_runner_and_both_conditions_lock_the_same_checkpoint(self) -> None:
        checkpoint_hash = sha256_file(ARTIFACTS / "surrogate_checkpoint.pt")
        self.assertEqual(checkpoint_hash, RUNNER.EXPECTED_SURROGATE_CHECKPOINT_HASH)
        self.assertEqual(
            checkpoint_hash,
            "4a4ae471e06dbeeea46bf09f0502f39455576ccdd7f992e0184912cac7b60791",
        )
        self.assertEqual(
            {record.metadata["surrogate_checkpoint_hash"] for record in self.greedy},
            {checkpoint_hash},
        )
        source = (ROOT / "experiments" / "14_run_bkt_set_lao.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("train_bkt_set_oracle", source)

    def test_all_surrogate_source_texts_have_canonical_lf_attributes(self) -> None:
        text_sources = [
            "artifacts/bkt_set/distillation_metadata.json",
            "artifacts/bkt_set/bkt_teacher_metadata.json",
            "artifacts/bkt_set/bkt_parameters.json",
            "artifacts/bkt_set/pooled_bkt_parameters.json",
            "artifacts/bkt_set/bkt_coverage.json",
            "documents/kt_set_adapter_spec.md",
            "data/ecs32a_dag_required_full_v1.json",
            "experiments/common/evaluator.py",
            "data/kt_set/student_split.json",
            "data/kt_set/preprocessing_manifest.json",
            "data/processed/cleaned_interactions.csv",
            "data/question_concept_mapping_final.csv",
        ]
        binary_sources = [
            "artifacts/bkt_set/train_grouped_tuples.parquet",
            "artifacts/bkt_set/validation_grouped_tuples.parquet",
            "artifacts/bkt_set/train_prefix_examples.parquet",
            "artifacts/bkt_set/validation_prefix_examples.parquet",
        ]
        for path in text_sources:
            output = subprocess.run(
                ["git", "check-attr", "eol", "--", path],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(output, f"{path}: eol: lf")
        for path in binary_sources:
            output = subprocess.run(
                ["git", "check-attr", "text", "--", path],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(output, f"{path}: text: unset")

    def test_records_are_complete_and_provenance_matched(self) -> None:
        self.assertEqual(len(self.records), 10)
        self.assertEqual(
            {record.target_node for record in self.records},
            set(self.manifest["targets"]),
        )
        closures = {
            closure["target_node"]: closure for closure in self.manifest["closures"]
        }
        expected_config_hash = sha256_file(ARTIFACTS / "surrogate_config.json")
        for record in self.records:
            closure = closures[record.target_node]
            self.assertIs(record.method, Method.BKT_SET_LAO)
            self.assertEqual(record.run_id, 0)
            self.assertEqual(set(record.sequence), set(closure["sequence_nodes"]))
            self.assertEqual(record.sequence[-1], record.target_node)
            self.assertEqual(record.metadata["condition_name"], "BKT-derived Set Oracle")
            self.assertEqual(record.metadata["closure_hash"], closure["closure_hash"])
            self.assertEqual(record.metadata["manifest_hash"], manifest_hash(self.manifest))
            self.assertEqual(record.metadata["evaluator_hash"], self.config["evaluator_hash"])
            self.assertEqual(record.metadata["surrogate_config_hash"], expected_config_hash)
            self.assertEqual(
                record.metadata["surrogate_checkpoint_hash"],
                RUNNER.EXPECTED_SURROGATE_CHECKPOINT_HASH,
            )
            self.assertTrue(record.metadata["oracle_state_dependence"])
            self.assertTrue(record.metadata["converged"])
            self.assertEqual(record.metadata["heuristic"], "sum_p_bar_1")

    def test_reference_loader_requires_the_full_oracle_identity(self) -> None:
        oracle, config = RUNNER.load_frozen_oracle()
        reference = RUNNER.load_greedy_reference(self.manifest, config, oracle)
        self.assertEqual(set(reference), set(self.manifest["targets"]))
        expected = RUNNER._reference_metadata(config, oracle)
        for record in reference.values():
            self.assertIs(record.method, Method.BKT_SET_GREEDY)
            self.assertIsNotNone(record.internal_cost)
            for key, value in expected.items():
                self.assertEqual(record.metadata[key], value)

    def test_lao_matches_exact_dp_and_never_exceeds_greedy(self) -> None:
        greedy_by_target = {record.target_node: record for record in self.greedy}
        for record in self.records:
            lao_cost = float(record.internal_cost)
            dp_cost = float(record.metadata["exact_dp_cost"])
            greedy_cost = float(greedy_by_target[record.target_node].internal_cost)
            self.assertLess(abs(lao_cost - dp_cost), RUNNER.DP_TOLERANCE)
            self.assertLessEqual(
                lao_cost,
                greedy_cost + RUNNER.INTERNAL_COST_TOLERANCE,
            )
            self.assertEqual(
                packed(greedy_cost),
                packed(float(record.metadata["greedy_internal_cost"])),
            )

    def test_independent_oracles_generate_identical_lao_signatures(self) -> None:
        first_oracle, first_config = RUNNER.load_frozen_oracle()
        second_oracle, second_config = RUNNER.load_frozen_oracle()
        first_ref = RUNNER.load_greedy_reference(
            self.manifest, first_config, first_oracle
        )
        second_ref = RUNNER.load_greedy_reference(
            self.manifest, second_config, second_oracle
        )
        first, _ = RUNNER.generate_records(
            self.manifest, first_oracle, first_config, first_ref
        )
        second, _ = RUNNER.generate_records(
            self.manifest, second_oracle, second_config, second_ref
        )
        self.assertEqual(RUNNER._signature(first), RUNNER._signature(second))
        self.assertEqual(RUNNER._signature(first), RUNNER._signature(self.records))

    def test_public_evaluator_and_comparison_outputs_are_complete(self) -> None:
        scored = SequenceEvaluator.from_artifacts().score_records(self.records)
        self.assertTrue(all(result.valid for result in scored))
        with (LAO_RESULTS / "scored_sequences.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            scored_rows = list(csv.DictReader(file))
        with COMPARISON_PATH.open(encoding="utf-8", newline="") as file:
            comparison_rows = list(csv.DictReader(file))
        self.assertEqual(len(scored_rows), 10)
        self.assertEqual(len(comparison_rows), 10)
        self.assertTrue(all(row["valid"] == "True" for row in scored_rows))
        self.assertEqual(
            {row["method"] for row in scored_rows}, {Method.BKT_SET_LAO.value}
        )
        self.assertTrue(all(row["converged"] == "True" for row in comparison_rows))
        self.assertTrue(
            all(float(row["lao_dp_absolute_gap"]) < RUNNER.DP_TOLERANCE for row in comparison_rows)
        )


if __name__ == "__main__":
    unittest.main()
