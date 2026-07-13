"""Contract tests for the frozen planner-facing BKT-derived Set Oracle."""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

import pandas as pd

from experiments.common.manifest import load_dag
from experiments.kt.artifacts import sha256_file
from experiments.kt.mastery import ancestor_map
from src.oracle_core.bkt_set_oracle import BKTSetOracle


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "bkt_set"
RESULTS = ROOT / "results" / "bkt_set"


def packed(value: float) -> bytes:
    return struct.pack("!d", value)


class BKTSetOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        grouped = pd.read_parquet(ARTIFACTS / "validation_grouped_tuples.parquet")
        cls.queries = []
        for row in grouped.itertuples(index=False):
            state = frozenset(json.loads(row.mastery_state))
            target = int(row.target_node)
            if target not in state:
                cls.queries.append((target, state))
        cls.queries = cls.queries[:50]
        if not cls.queries:
            raise AssertionError("Need at least one valid validation query")
        nodes, edges = load_dag(ROOT / "data" / "ecs32a_dag_required_full_v1.json")
        cls.ancestors = ancestor_map(nodes, edges)

    def test_repeated_query_is_bitwise_identical(self) -> None:
        oracle = BKTSetOracle.from_artifacts()
        target, state = self.queries[0]
        self.assertEqual(
            packed(oracle.success_prob(target, state)),
            packed(oracle.success_prob(target, state)),
        )

    def test_query_order_is_identical_across_independent_objects(self) -> None:
        first = BKTSetOracle.from_artifacts()
        forward = {
            query: packed(first.success_prob(*query)) for query in self.queries
        }
        second = BKTSetOracle.from_artifacts()
        reverse = {
            query: packed(second.success_prob(*query))
            for query in reversed(self.queries)
        }
        self.assertEqual(forward, reverse)

    def test_new_object_reload_is_bitwise_identical(self) -> None:
        first = BKTSetOracle.from_artifacts()
        second = BKTSetOracle.from_artifacts()
        for query in self.queries:
            self.assertEqual(
                packed(first.success_prob(*query)),
                packed(second.success_prob(*query)),
            )

    def test_empty_and_warm_cache_are_identical(self) -> None:
        oracle = BKTSetOracle.from_artifacts()
        target, state = self.queries[-1]
        cold = oracle.success_prob(target, state)
        warm = oracle.success_prob(target, state)
        oracle._cache.clear()
        cold_again = oracle.success_prob(target, state)
        self.assertEqual(packed(cold), packed(warm))
        self.assertEqual(packed(cold), packed(cold_again))

    def test_model_is_cpu_eval_and_gradient_frozen(self) -> None:
        oracle = BKTSetOracle.from_artifacts()
        self.assertFalse(oracle.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in oracle.model.parameters()))
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in oracle.model.parameters()))

    def test_base_cost_and_heuristic_bound(self) -> None:
        oracle = BKTSetOracle.from_artifacts()
        self.assertEqual(oracle.base_cost(6), 60.0)
        self.assertEqual(oracle.best_case_success_prob(6), 1.0)

    def test_invalid_queries_are_rejected(self) -> None:
        oracle = BKTSetOracle.from_artifacts()
        with self.assertRaises(TypeError):
            oracle.success_prob(True, frozenset())
        with self.assertRaises(ValueError):
            oracle.success_prob(999, frozenset())
        with self.assertRaises(ValueError):
            oracle.success_prob(9, frozenset())
        with self.assertRaises(TypeError):
            oracle.success_prob(6, frozenset({True}))
        with self.assertRaises(ValueError):
            oracle.success_prob(6, frozenset({999}))
        with self.assertRaises(ValueError):
            oracle.success_prob(6, frozenset({6}))
        non_root = next(node for node, ancestors in self.ancestors.items() if ancestors)
        target = next(node for node in oracle.supported_targets if node != non_root)
        with self.assertRaisesRegex(ValueError, "prerequisite-closed"):
            oracle.success_prob(target, frozenset({non_root}))

    def test_state_dependence_gate_and_provenance(self) -> None:
        with (ARTIFACTS / "surrogate_metrics.json").open(encoding="utf-8") as file:
            metrics = json.load(file)
        dependence = metrics["state_dependence"]
        self.assertTrue(metrics["go"])
        self.assertTrue(dependence["packed_outputs_differ"])
        self.assertGreaterEqual(dependence["max_state_effect"], 1e-6)
        self.assertEqual(dependence["targets_with_multiple_states"], 24)
        self.assertEqual(dependence["targets_with_single_state"], 3)
        self.assertEqual(
            metrics["checkpoint_hash"],
            sha256_file(ARTIFACTS / "surrogate_checkpoint.pt"),
        )
        effects = pd.read_csv(RESULTS / "state_dependence.csv")
        multiple = effects[effects["validation_state_count"] >= 2]
        single = effects[effects["validation_state_count"] < 2]
        self.assertEqual(len(multiple), 24)
        self.assertEqual(set(single["target_node"]), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
