"""Regression coverage for the shared experiment protocol (Tasks 1-5)."""

from __future__ import annotations

import struct
import unittest

from experiments.common.evaluator import REGRET_TOLERANCE, SequenceEvaluator
from experiments.common.frozen_oracle import FrozenMonotonicOracle
from experiments.common.manifest import load_manifest
from experiments.common.schema import Method, SequenceRecord
from src.planner_engine.heuristics import sum_heuristic
from src.planner_engine.solver import DAGPlannerDP


EXPECTED_TARGETS = [6, 7, 12, 18, 29, 36, 39, 42, 46, 52]


def _float_bits(value: float) -> bytes:
    return struct.pack("!d", value)


class ManifestTests(unittest.TestCase):
    def test_manifest_targets_are_fixed(self) -> None:
        manifest = load_manifest()
        self.assertEqual(manifest["targets"], EXPECTED_TARGETS)

    def test_closures_are_deterministic_and_have_unique_target_sink(self) -> None:
        first = load_manifest()
        second = load_manifest()
        self.assertEqual(first, second)

        for closure in first["closures"]:
            sources = {src for src, _ in closure["edges"]}
            sinks = set(closure["nodes"]) - sources
            self.assertEqual(sinks, {closure["target_node"]})
            self.assertEqual(
                closure["sequence_nodes"],
                [
                    node
                    for node in closure["nodes"]
                    if node not in first["initial_state"]
                ],
            )


class FrozenOracleTests(unittest.TestCase):
    QUERIES = [
        (6, frozenset()),
        (7, frozenset({0, 1, 2})),
        (12, frozenset({0, 1, 2, 4, 5})),
        (42, frozenset({0, 1, 2, 3, 4, 11})),
    ]

    def test_repeated_calls_are_bitwise_identical(self) -> None:
        oracle = FrozenMonotonicOracle.from_artifacts(device="cpu")
        for node, state in self.QUERIES:
            first = oracle.success_prob(node, state)
            second = oracle.success_prob(node, state)
            self.assertEqual(_float_bits(first), _float_bits(second))

    def test_query_order_produces_identical_table(self) -> None:
        forward_oracle = FrozenMonotonicOracle.from_artifacts(device="cpu")
        reverse_oracle = FrozenMonotonicOracle.from_artifacts(device="cpu")
        forward = {
            query: _float_bits(forward_oracle.success_prob(*query))
            for query in self.QUERIES
        }
        reverse = {
            query: _float_bits(reverse_oracle.success_prob(*query))
            for query in reversed(self.QUERIES)
        }
        self.assertEqual(forward, reverse)

    def test_new_oracle_objects_produce_bitwise_identical_probability(self) -> None:
        # Two independent constructors are intentional: this must not pass by
        # hitting one object's content-addressed cache.
        first_oracle = FrozenMonotonicOracle.from_artifacts(device="cpu")
        second_oracle = FrozenMonotonicOracle.from_artifacts(device="cpu")
        for query in self.QUERIES:
            first = first_oracle.success_prob(*query)
            second = second_oracle.success_prob(*query)
            self.assertEqual(_float_bits(first), _float_bits(second))


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = SequenceEvaluator.from_artifacts()
        cls.exact_cost, cls.exact_path = cls.evaluator.exact_optimum(6)

    def test_exact_dp_path_scores_as_valid_with_zero_regret(self) -> None:
        record = SequenceRecord(
            method=Method.EXACT_DP,
            target_node=6,
            run_id=0,
            sequence=self.exact_path,
            internal_cost=self.exact_cost,
            metadata={},
        )
        scored = self.evaluator.score(record)
        self.assertTrue(scored.valid)
        self.assertEqual(scored.normalized_regret, 0.0)
        self.assertIsNotNone(scored.evaluation_cost)
        self.assertLess(
            abs(scored.evaluation_cost - self.exact_cost),
            REGRET_TOLERANCE * max(1.0, self.exact_cost),
        )

    def test_same_sequence_has_identical_cost_across_methods(self) -> None:
        records = [
            SequenceRecord(Method.EXACT_DP, 6, 0, self.exact_path, None, {}),
            SequenceRecord(Method.ARIADNE_LAO, 6, 0, self.exact_path, None, {}),
            SequenceRecord(Method.FREQUENCY_GREEDY, 6, 0, self.exact_path, None, {}),
        ]
        scored = self.evaluator.score_records(records)
        self.assertEqual(len({result.evaluation_cost for result in scored}), 1)
        self.assertEqual(len({result.sequence_hash for result in scored}), 1)

    def test_invalid_prerequisite_order_is_rejected(self) -> None:
        record = SequenceRecord(
            Method.ARIADNE_GREEDY,
            6,
            0,
            [1, 0, 2, 4, 5, 6],
            None,
            {},
        )
        scored = self.evaluator.score(record)
        self.assertFalse(scored.valid)
        self.assertTrue(scored.invalid_reason.startswith("prerequisites_not_mastered"))

    def test_duplicate_node_is_rejected_by_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SequenceRecord(
                Method.ARIADNE_GREEDY,
                6,
                0,
                [0, 1, 2, 4, 5, 5, 6],
                None,
                {},
            )

    def test_missing_node_is_rejected(self) -> None:
        record = SequenceRecord(
            Method.ARIADNE_GREEDY,
            6,
            0,
            [0, 1, 2, 4, 6],
            None,
            {},
        )
        scored = self.evaluator.score(record)
        self.assertFalse(scored.valid)
        self.assertTrue(scored.invalid_reason.startswith("sequence_missing_required_nodes"))

    def test_target_must_be_final(self) -> None:
        record = SequenceRecord(
            Method.ARIADNE_GREEDY,
            6,
            0,
            [0, 1, 2, 4, 6, 5],
            None,
            {},
        )
        scored = self.evaluator.score(record)
        self.assertFalse(scored.valid)
        self.assertEqual(scored.invalid_reason, "target_must_be_final_sequence_node")

    def test_duplicate_experiment_identity_is_rejected(self) -> None:
        record = SequenceRecord(Method.EXACT_DP, 6, 0, self.exact_path, None, {})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.evaluator.score_records([record, record])

    def test_sum_heuristic_with_unit_probability_bound_is_admissible(self) -> None:
        closure = self.evaluator.closures[6]
        graph = self.evaluator._closure_graph(6)
        target = frozenset(closure["nodes"])

        # Enumerate every prerequisite-closed mastery state for this closure,
        # rather than checking only the initial state.
        nodes = closure["nodes"]
        tested_states = 0
        for mask in range(1 << len(nodes)):
            state = frozenset(
                node for index, node in enumerate(nodes) if mask & (1 << index)
            )
            if any(
                not set(graph.predecessors(node)) <= state
                for node in state
            ):
                continue

            planner = DAGPlannerDP(
                oracle=self.evaluator.oracle,
                nx_graph=graph,
                config={"planner": {"base_cost": self.evaluator.manifest["base_cost"]}},
                edge_index=self.evaluator.oracle.edge_index,
                num_nodes=self.evaluator.oracle.model.num_nodes,
            )
            exact_remaining, _ = planner.solve(set(state), set(target))
            heuristic = sum_heuristic(state, target, self.evaluator.oracle)
            self.assertLessEqual(
                heuristic,
                exact_remaining + 1e-9,
                msg=f"inadmissible at state {sorted(state)}",
            )
            tested_states += 1

        self.assertGreater(tested_states, 1)


if __name__ == "__main__":
    unittest.main()
