"""Tests for deterministic BKT fitting, inference, and pooled coverage."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.kt.artifacts import canonical_json_bytes, sha256_file
from src.oracle_core.bkt_teacher import (
    BKTParameters,
    BKTTeacher,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "bkt_set"
POOLED_NODES = {0, 1, 2, 5, 11, 32, 37, 51}


def packed(value: float) -> bytes:
    return struct.pack("!d", value)


class BKTFittingTests(unittest.TestCase):
    def test_small_fit_is_deterministic(self) -> None:
        script = """
import json
from src.oracle_core.bkt_teacher import fit_bkt_parameters
sequences = SEQUENCES
result = fit_bkt_parameters(sequences)
print(json.dumps({
    "objective": result.objective,
    "parameters": result.parameters.to_dict(),
    "selected_restart": result.selected_restart,
}, sort_keys=True))
"""
        forward = '{"a":[0,1,1,1],"b":[0,0,1],"c":[1,1,1]}'
        reverse = '{"c":[1,1,1],"b":[0,0,1],"a":[0,1,1,1]}'
        first = subprocess.check_output(
            [sys.executable, "-c", script.replace("SEQUENCES", forward)],
            cwd=ROOT,
            text=True,
        ).strip()
        second = subprocess.check_output(
            [sys.executable, "-c", script.replace("SEQUENCES", reverse)],
            cwd=ROOT,
            text=True,
        ).strip()
        self.assertEqual(first, second)


class BKTInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        shared = BKTParameters(prior=0.2, learn=0.1, guess=0.2, slip=0.1)
        other = BKTParameters(prior=0.4, learn=0.2, guess=0.1, slip=0.2)
        self.teacher = BKTTeacher({0: shared, 1: shared, 2: other})

    def test_unseen_concept_uses_prior_and_queries_are_read_only(self) -> None:
        state = self.teacher.new_student_state()
        before = state.posteriors
        first = state.query(0)
        second = state.query(0)
        self.assertEqual(packed(first), packed(second))
        self.assertEqual(state.posteriors, before)

    def test_query_order_does_not_change_results(self) -> None:
        first = self.teacher.new_student_state()
        forward = {node: first.query(node) for node in [0, 1, 2]}
        second = self.teacher.new_student_state()
        reverse = {node: second.query(node) for node in [2, 1, 0]}
        self.assertEqual(
            {node: packed(value) for node, value in forward.items()},
            {node: packed(value) for node, value in reverse.items()},
        )

    def test_current_heldout_label_cannot_change_current_prefix_query(self) -> None:
        correct_future = self.teacher.new_student_state()
        incorrect_future = self.teacher.new_student_state()
        self.assertEqual(packed(correct_future.query(0)), packed(incorrect_future.query(0)))
        correct_future.observe(0, 1)
        incorrect_future.observe(0, 0)
        self.assertNotEqual(packed(correct_future.query(0)), packed(incorrect_future.query(0)))

    def test_observation_updates_only_its_own_node(self) -> None:
        state = self.teacher.new_student_state()
        node_zero_before = state.query(0)
        node_one_before = state.query(1)
        state.observe(0, 1)
        self.assertNotEqual(packed(state.query(0)), packed(node_zero_before))
        self.assertEqual(packed(state.query(1)), packed(node_one_before))


class BKTArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ARTIFACTS / "bkt_parameters.json").open(encoding="utf-8") as file:
            cls.parameters = json.load(file)
        with (ARTIFACTS / "bkt_coverage.json").open(encoding="utf-8") as file:
            cls.coverage = json.load(file)
        with (ARTIFACTS / "pooled_bkt_parameters.json").open(encoding="utf-8") as file:
            cls.pooled = json.load(file)
        with (ARTIFACTS / "bkt_teacher_metadata.json").open(encoding="utf-8") as file:
            cls.metadata = json.load(file)

    def test_observed_and_pooled_parameter_sources_are_exact(self) -> None:
        entries = {entry["node_id"]: entry for entry in self.parameters["parameters"]}
        self.assertEqual(set(self.coverage["pooled_backoff_nodes"]), POOLED_NODES)
        self.assertEqual(set(self.coverage["missing_nodes"]), set())
        self.assertEqual(self.coverage["coverage_fraction"], 1.0)
        self.assertEqual(len(entries), self.coverage["required_node_count"])
        for node, entry in entries.items():
            if node in POOLED_NODES:
                self.assertEqual(entry["parameter_source"], "pooled_zero_observation_bkt")
                self.assertEqual(entry["train_observations"], 0)
            else:
                self.assertEqual(entry["parameter_source"], "concept_specific")
                self.assertGreater(entry["train_observations"], 0)

    def test_pooled_fit_uses_training_students_only(self) -> None:
        self.assertEqual(self.metadata["training_split"], "train")
        self.assertEqual(self.metadata["validation_students_used"], 0)
        self.assertEqual(self.metadata["test_students_used"], 0)
        self.assertEqual(self.metadata["train_student_count"], 236)

    def test_all_pooled_nodes_return_legal_probabilities(self) -> None:
        teacher = BKTTeacher.from_artifact(ARTIFACTS / "bkt_parameters.json")
        state = teacher.new_student_state()
        for node in POOLED_NODES:
            probability = state.query(node)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_numeric_value_hashes_are_separate_from_artifact_hashes(self) -> None:
        vector = self.pooled["parameters"]
        vector_hash = hashlib.sha256(canonical_json_bytes(vector)).hexdigest()
        self.assertEqual(self.pooled["pooled_parameter_vector_hash"], vector_hash)
        self.assertEqual(self.coverage["pooled_parameter_vector_hash"], vector_hash)
        self.assertEqual(
            self.coverage["pooled_parameter_artifact_hash"],
            sha256_file(ARTIFACTS / "pooled_bkt_parameters.json"),
        )

        values_payload = []
        for entry in self.parameters["parameters"]:
            entry_values = {
                name: entry[name] for name in ("guess", "learn", "prior", "slip")
            }
            self.assertEqual(
                entry["parameter_values_hash"],
                hashlib.sha256(canonical_json_bytes(entry_values)).hexdigest(),
            )
            values_payload.append(
                {
                    "node_id": entry["node_id"],
                    "parameter_source": entry["parameter_source"],
                    "prior": entry["prior"],
                    "learn": entry["learn"],
                    "guess": entry["guess"],
                    "slip": entry["slip"],
                }
            )
        values_hash = hashlib.sha256(
            canonical_json_bytes(values_payload)
        ).hexdigest()
        self.assertEqual(self.parameters["parameter_values_hash"], values_hash)
        self.assertEqual(self.coverage["parameter_values_hash"], values_hash)
        self.assertEqual(
            self.coverage["bkt_parameter_artifact_hash"],
            sha256_file(ARTIFACTS / "bkt_parameters.json"),
        )


if __name__ == "__main__":
    unittest.main()
