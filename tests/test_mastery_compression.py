"""Regression tests for the frozen prefix-to-mastery protocol."""

from __future__ import annotations

import unittest

from experiments.kt.mastery import (
    ConceptSession,
    ancestor_map,
    canonical_mastery_tuple,
    mastery_state_before_prefix,
    prerequisite_closed_projection,
    raw_mastery_state_before_prefix,
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

    def test_raw_mastered_node_without_ancestor_is_removed(self) -> None:
        self.assertEqual(
            prerequisite_closed_projection({1, 2}, self.ancestors),
            frozenset(),
        )

    def test_raw_mastered_node_with_all_ancestors_is_retained(self) -> None:
        state = prerequisite_closed_projection({0, 1, 2}, self.ancestors)
        self.assertEqual(state, frozenset({0, 1, 2}))
        self.assertTrue(all(self.ancestors[node].issubset(state) for node in state))

    def test_same_prefix_has_same_sorted_serialization(self) -> None:
        history = sessions(
            (3, 1.0), (0, 1.0), (3, 1.0), (0, 1.0), (3, 1.0), (0, 1.0)
        )
        first = mastery_state_before_prefix(history, 6, ancestors=self.ancestors)
        second = mastery_state_before_prefix(history, 6, ancestors=self.ancestors)
        self.assertEqual(canonical_mastery_tuple(first), (0, 3))
        self.assertEqual(canonical_mastery_tuple(first), canonical_mastery_tuple(second))


if __name__ == "__main__":
    unittest.main()

