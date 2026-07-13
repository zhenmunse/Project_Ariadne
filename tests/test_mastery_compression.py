"""Regression tests for the frozen prefix-to-mastery protocol."""

from __future__ import annotations

import unittest

from experiments.kt.mastery import (
    ConceptSession,
    ancestor_map,
    canonical_mastery_tuple,
    mastery_state_before_prefix,
    raw_mastery_state_before_prefix,
    zero_observation_nodes,
    zero_observation_prerequisite_completion,
)


def sessions(*values: tuple[int, float]) -> list[ConceptSession]:
    return [ConceptSession(node, score) for node, score in values]


class MasteryCompressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ancestors = ancestor_map([0, 1, 2, 3], [(0, 1), (1, 2)])

    def test_fewer_than_three_successful_sessions_is_not_mastery(self) -> None:
        history = sessions((0, 0.8), (0, 1.0))
        self.assertEqual(raw_mastery_state_before_prefix(history, 2), frozenset())

    def test_three_consecutive_sessions_reaches_mastery(self) -> None:
        history = sessions((0, 0.8), (0, 0.9), (0, 1.0))
        self.assertEqual(raw_mastery_state_before_prefix(history, 3), frozenset({0}))

    def test_mastery_is_irreversible_after_a_low_score(self) -> None:
        history = sessions((0, 1.0), (0, 0.9), (0, 0.8), (0, 0.0))
        self.assertEqual(raw_mastery_state_before_prefix(history, 4), frozenset({0}))

    def test_consecutive_counts_are_per_concept(self) -> None:
        history = sessions(
            (0, 1.0),
            (1, 1.0),
            (0, 1.0),
            (1, 0.0),
            (0, 1.0),
        )
        self.assertEqual(raw_mastery_state_before_prefix(history, 5), frozenset({0}))

    def test_current_session_does_not_leak_into_prefix(self) -> None:
        history = sessions((0, 1.0), (0, 1.0), (0, 1.0))
        self.assertEqual(raw_mastery_state_before_prefix(history, 2), frozenset())
        self.assertEqual(raw_mastery_state_before_prefix(history, 3), frozenset({0}))

    def test_empty_raw_mastery_remains_empty(self) -> None:
        result = zero_observation_prerequisite_completion(
            set(), self.ancestors, {2, 3}
        )
        self.assertEqual(result.state, frozenset())

    def test_zero_observation_ancestor_chain_is_completed(self) -> None:
        result = zero_observation_prerequisite_completion(
            {2}, self.ancestors, {2, 3}
        )
        self.assertEqual(result.retained_mastery, frozenset({2}))
        self.assertEqual(result.completed_ancestors, frozenset({0, 1}))
        self.assertEqual(result.state, frozenset({0, 1, 2}))

    def test_missing_observed_ancestor_removes_descendant(self) -> None:
        result = zero_observation_prerequisite_completion(
            {2}, self.ancestors, {1, 2, 3}
        )
        self.assertEqual(result.state, frozenset())

    def test_observed_ancestors_are_retained_and_zero_gap_is_completed(self) -> None:
        chain = ancestor_map([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
        result = zero_observation_prerequisite_completion(
            {0, 2, 3}, chain, {0, 2, 3}
        )
        self.assertEqual(result.completed_ancestors, frozenset({1}))
        self.assertEqual(result.state, frozenset({0, 1, 2, 3}))

    def test_unrelated_zero_observation_node_is_not_added(self) -> None:
        graph = ancestor_map([0, 1, 2, 3], [(0, 1), (2, 3)])
        result = zero_observation_prerequisite_completion({1}, graph, {1, 3})
        self.assertEqual(result.state, frozenset({0, 1}))
        self.assertNotIn(2, result.state)

    def test_completed_state_is_always_prerequisite_closed(self) -> None:
        result = zero_observation_prerequisite_completion(
            {0, 2}, self.ancestors, {0, 2, 3}
        )
        self.assertTrue(
            all(self.ancestors[node].issubset(result.state) for node in result.state)
        )

    def test_zero_observation_set_uses_training_nodes_only(self) -> None:
        nodes = [0, 1, 2, 3]
        train_observed = {1, 3}
        validation_observed = {0, 2}
        first = zero_observation_nodes(nodes, train_observed)
        second = zero_observation_nodes(nodes, train_observed)
        self.assertEqual(first, frozenset({0, 2}))
        self.assertEqual(first, second)
        self.assertNotEqual(first, zero_observation_nodes(nodes, validation_observed))

    def test_same_prefix_has_same_sorted_serialization(self) -> None:
        history = sessions(
            (3, 1.0), (0, 1.0), (3, 1.0), (0, 1.0), (3, 1.0), (0, 1.0)
        )
        first = mastery_state_before_prefix(
            history, 6, ancestors=self.ancestors, training_observed_nodes={0, 1, 2, 3}
        )
        second = mastery_state_before_prefix(
            history, 6, ancestors=self.ancestors, training_observed_nodes={0, 1, 2, 3}
        )
        self.assertEqual(canonical_mastery_tuple(first), (0, 3))
        self.assertEqual(canonical_mastery_tuple(first), canonical_mastery_tuple(second))


if __name__ == "__main__":
    unittest.main()

