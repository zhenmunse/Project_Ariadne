"""Contract tests for the frozen DKT-derived Set Oracle."""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

from experiments.kt.artifacts import sha256_file
from src.oracle_core.dkt_set_oracle import DKTSetOracle


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "dkt_set"


def packed(value: float) -> bytes:
    return struct.pack("!d", value)


class DKTSetOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.oracle = DKTSetOracle.from_artifacts()
        with (ARTIFACTS / "surrogate_metrics.json").open(encoding="utf-8") as file:
            cls.metrics = json.load(file)

    def test_checkpoint_reload_and_provenance_identity(self) -> None:
        other = DKTSetOracle.from_artifacts()
        self.assertEqual(self.oracle.config_hash, other.config_hash)
        self.assertEqual(self.oracle.checkpoint_hash, other.checkpoint_hash)
        self.assertEqual(
            self.oracle.checkpoint_hash,
            sha256_file(ARTIFACTS / "surrogate_checkpoint.pt"),
        )
        self.assertEqual(self.oracle.teacher_tensor_hash, other.teacher_tensor_hash)
        self.assertEqual(self.oracle.tuple_collection_hash, other.tuple_collection_hash)

    def test_repeated_query_query_order_and_new_object_are_bitwise_identical(self) -> None:
        queries = [(6, frozenset()), (7, frozenset({0, 1, 2, 4, 5, 6}))]
        first = {(v, state): packed(self.oracle.success_prob(v, state)) for v, state in queries}
        repeated = {(v, state): packed(self.oracle.success_prob(v, state)) for v, state in queries}
        other = DKTSetOracle.from_artifacts()
        reversed_table = {
            (v, state): packed(other.success_prob(v, state))
            for v, state in reversed(queries)
        }
        self.assertEqual(first, repeated)
        self.assertEqual(first, reversed_table)

    def test_invalid_queries_match_shared_set_oracle_contract(self) -> None:
        with self.assertRaises(TypeError):
            self.oracle.success_prob(True, frozenset())
        with self.assertRaises(ValueError):
            self.oracle.success_prob(999, frozenset())
        with self.assertRaises(TypeError):
            self.oracle.success_prob(6, frozenset({True}))
        with self.assertRaises(ValueError):
            self.oracle.success_prob(6, frozenset({999}))
        with self.assertRaises(ValueError):
            self.oracle.success_prob(6, frozenset({6}))
        with self.assertRaises(ValueError):
            self.oracle.success_prob(6, frozenset({5}))
        with self.assertRaises(ValueError):
            DKTSetOracle.from_artifacts(device="cuda")

    def test_best_case_probability_and_state_dependence_gate(self) -> None:
        self.assertEqual(self.oracle.base_cost(6), 60.0)
        self.assertEqual(self.oracle.best_case_success_prob(6), 1.0)
        state = self.metrics["state_dependence"]
        self.assertTrue(self.metrics["go"])
        self.assertTrue(state["packed_outputs_differ"])
        self.assertGreaterEqual(state["max_state_effect"], 1e-6)


if __name__ == "__main__":
    unittest.main()
