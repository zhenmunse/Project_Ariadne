"""Planning tests for DKT-derived Set Oracle + Greedy."""

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
ARTIFACTS = ROOT / "artifacts" / "dkt_set"
RESULTS = ROOT / "results" / "dkt_set_greedy"
RUNNER = importlib.import_module("experiments.15_run_dkt_set_greedy")


class DKTSetGreedyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.records = read_jsonl(RESULTS / "sequences.jsonl")
        with (ARTIFACTS / "surrogate_config.json").open(encoding="utf-8") as file:
            cls.config = json.load(file)

    def test_runner_loads_only_the_frozen_checkpoint(self) -> None:
        self.assertEqual(
            sha256_file(ARTIFACTS / "surrogate_checkpoint.pt"),
            RUNNER.EXPECTED_SURROGATE_CHECKPOINT_HASH,
        )
        source = (ROOT / "experiments" / "15_run_dkt_set_greedy.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("train_dkt_set_oracle", source)

    def test_records_cover_all_targets_and_match_provenance(self) -> None:
        self.assertEqual(len(self.records), 10)
        self.assertEqual({record.target_node for record in self.records}, set(self.manifest["targets"]))
        closures = {c["target_node"]: c for c in self.manifest["closures"]}
        for record in self.records:
            closure = closures[record.target_node]
            self.assertIs(record.method, Method.DKT_SET_GREEDY)
            self.assertEqual(record.run_id, 0)
            self.assertEqual(set(record.sequence), set(closure["sequence_nodes"]))
            self.assertEqual(record.sequence[-1], record.target_node)
            self.assertEqual(record.metadata["manifest_hash"], manifest_hash(self.manifest))
            self.assertEqual(record.metadata["closure_hash"], closure["closure_hash"])
            self.assertEqual(record.metadata["surrogate_checkpoint_hash"], RUNNER.EXPECTED_SURROGATE_CHECKPOINT_HASH)
            for key in (
                "evaluator_hash", "split_hash", "compression_config_hash",
                "zero_observation_nodes_hash", "training_observed_nodes_hash",
                "teacher_tensor_hash", "teacher_checkpoint_hash", "teacher_config_hash",
            ):
                self.assertEqual(record.metadata[key], self.config[key])
            self.assertEqual(record.metadata["distillation_table_hash"], self.config["tuple_collection_hash"])

    def test_independent_oracles_generate_identical_plans_and_costs(self) -> None:
        first_oracle, first_config = RUNNER.load_frozen_oracle()
        second_oracle, second_config = RUNNER.load_frozen_oracle()
        first = RUNNER.generate_records(self.manifest, first_oracle, first_config)
        second = RUNNER.generate_records(self.manifest, second_oracle, second_config)
        self.assertEqual(RUNNER._signature(first), RUNNER._signature(second))
        self.assertEqual(RUNNER._signature(first), RUNNER._signature(self.records))
        for record in self.records:
            state = set(self.manifest["initial_state"])
            cost = 0.0
            for node in record.sequence:
                cost += first_oracle.base_cost(node) / first_oracle.success_prob(node, state)
                state.add(node)
            self.assertEqual(struct.pack("!d", cost), struct.pack("!d", float(record.internal_cost)))

    def test_public_evaluator_accepts_all_records(self) -> None:
        scored = SequenceEvaluator.from_artifacts().score_records(self.records)
        self.assertTrue(all(result.valid for result in scored))
        with (RESULTS / "scored_sequences.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["valid"] == "True" for row in rows))
        self.assertEqual({row["method"] for row in rows}, {Method.DKT_SET_GREEDY.value})


if __name__ == "__main__":
    unittest.main()
