"""Planning-level tests for BKT-derived Set Oracle + Greedy."""

from __future__ import annotations

import csv
import importlib
import json
import struct
import unittest
from pathlib import Path

from experiments.common.evaluator import SequenceEvaluator
from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "bkt_set"
RESULTS = ROOT / "results" / "bkt_set_greedy"
RUNNER = importlib.import_module("experiments.13_run_bkt_greedy")


def packed(value: float) -> bytes:
    return struct.pack("!d", value)


class BKTSetGreedyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.records = read_jsonl(RESULTS / "sequences.jsonl")
        with (ARTIFACTS / "surrogate_config.json").open(encoding="utf-8") as file:
            cls.config = json.load(file)

    def test_runner_is_locked_to_the_approved_checkpoint(self) -> None:
        actual = sha256_file(ARTIFACTS / "surrogate_checkpoint.pt")
        self.assertEqual(actual, RUNNER.EXPECTED_SURROGATE_CHECKPOINT_HASH)
        self.assertEqual(
            actual,
            "d285d7666e658c8f10637deffc986e408e5a15bbb2d3dcff50856cff7250d4f4",
        )
        source = (ROOT / "experiments" / "13_run_bkt_greedy.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("train_bkt_set_oracle", source)

    def test_records_are_complete_and_provenance_matched(self) -> None:
        self.assertEqual(len(self.records), 10)
        self.assertEqual({record.target_node for record in self.records}, set(self.manifest["targets"]))
        closure_hashes = {
            closure["target_node"]: closure["closure_hash"]
            for closure in self.manifest["closures"]
        }
        for record in self.records:
            self.assertIs(record.method, Method.BKT_SET_GREEDY)
            self.assertEqual(record.run_id, 0)
            self.assertIsNotNone(record.internal_cost)
            self.assertEqual(record.metadata["condition_name"], "BKT-derived Set Oracle")
            self.assertEqual(record.metadata["closure_hash"], closure_hashes[record.target_node])
            self.assertEqual(record.metadata["manifest_hash"], manifest_hash(self.manifest))
            self.assertEqual(
                record.metadata["surrogate_checkpoint_hash"],
                RUNNER.EXPECTED_SURROGATE_CHECKPOINT_HASH,
            )
            for key in (
                "evaluator_hash",
                "split_hash",
                "compression_config_hash",
                "parameter_values_hash",
                "bkt_parameter_artifact_hash",
                "pooled_parameter_vector_hash",
                "pooled_parameter_artifact_hash",
                "pooled_backoff_nodes_hash",
                "distillation_table_hash",
                "surrogate_config_hash",
            ):
                self.assertEqual(record.metadata[key], self._expected_metadata(key))
            self.assertTrue(record.metadata["oracle_state_dependence"])

    def _expected_metadata(self, key: str) -> str:
        mapping = {
            "evaluator_hash": self.config["evaluator_hash"],
            "split_hash": self.config["split_hash"],
            "compression_config_hash": self.config["compression_config_hash"],
            "parameter_values_hash": self.config["parameter_values_hash"],
            "bkt_parameter_artifact_hash": self.config["bkt_parameter_artifact_hash"],
            "pooled_parameter_vector_hash": self.config["pooled_parameter_vector_hash"],
            "pooled_parameter_artifact_hash": self.config["pooled_parameter_artifact_hash"],
            "pooled_backoff_nodes_hash": self.config["pooled_backoff_nodes_hash"],
            "distillation_table_hash": self.config["tuple_collection_hash"],
            "surrogate_config_hash": sha256_file(ARTIFACTS / "surrogate_config.json"),
        }
        return mapping[key]

    def test_independent_oracles_generate_bitwise_identical_plans(self) -> None:
        first_oracle, first_config = RUNNER.load_frozen_oracle()
        second_oracle, second_config = RUNNER.load_frozen_oracle()
        first = RUNNER.generate_records(self.manifest, first_oracle, first_config)
        second = RUNNER.generate_records(self.manifest, second_oracle, second_config)
        self.assertEqual(RUNNER._signature(first), RUNNER._signature(second))
        self.assertEqual(RUNNER._signature(first), RUNNER._signature(self.records))

    def test_internal_cost_is_recomputed_from_the_same_frozen_oracle(self) -> None:
        oracle, _ = RUNNER.load_frozen_oracle()
        initial = set(self.manifest["initial_state"])
        for record in self.records:
            state = set(initial)
            cost = 0.0
            for node in record.sequence:
                cost += oracle.base_cost(node) / oracle.success_prob(node, state)
                state.add(node)
            self.assertEqual(packed(cost), packed(float(record.internal_cost)))

    def test_common_evaluator_accepts_every_record(self) -> None:
        scored = SequenceEvaluator.from_artifacts().score_records(self.records)
        self.assertTrue(all(result.valid for result in scored))
        with (RESULTS / "scored_sequences.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["valid"] == "True" for row in rows))
        self.assertEqual({row["method"] for row in rows}, {Method.BKT_SET_GREEDY.value})


if __name__ == "__main__":
    unittest.main()

