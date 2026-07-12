"""Tests for BKT prefix expansion and deterministic tuple aggregation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from experiments.build_bkt_distillation_data import (
    build_prefix_examples,
    group_prefix_examples,
)
from experiments.kt.mastery import ancestor_map
from src.oracle_core.bkt_teacher import BKTParameters, BKTTeacher


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "bkt_set"


class BKTDistillationUnitTests(unittest.TestCase):
    def test_current_label_is_not_used_in_current_prefix(self) -> None:
        parameters = BKTParameters(prior=0.2, learn=0.1, guess=0.2, slip=0.1)
        teacher = BKTTeacher({0: parameters})
        base = {
            "student_id": "a",
            "session_index": 0,
            "source_order": 0,
            "timestamp": "2026-01-01T00:00:00Z",
            "target_node": 0,
            "session_score": 1.0,
            "split": "train",
        }
        correct = pd.DataFrame([{**base, "correct": 1}])
        incorrect = pd.DataFrame([{**base, "correct": 0}])
        ancestors = ancestor_map([0], [])
        first = build_prefix_examples(
            correct,
            split_name="train",
            teacher=teacher,
            required_nodes=[0],
            dag_nodes=[0],
            ancestors=ancestors,
        )
        second = build_prefix_examples(
            incorrect,
            split_name="train",
            teacher=teacher,
            required_nodes=[0],
            dag_nodes=[0],
            ancestors=ancestors,
        )
        pd.testing.assert_frame_equal(first, second, check_exact=True)

    def test_grouped_mean_counts_and_order_are_exact(self) -> None:
        raw = pd.DataFrame(
            [
                ["a", 0, "train", "[]", "00", 1, 0.2],
                ["b", 0, "train", "[]", "00", 1, 0.4],
                ["a", 1, "train", "[0]", "10", 0, 0.8],
            ],
            columns=[
                "student_id",
                "prefix_index",
                "split",
                "mastery_state",
                "mastery_mask",
                "target_node",
                "teacher_probability",
            ],
        )
        grouped = group_prefix_examples(raw)
        self.assertEqual(grouped["mastery_mask"].tolist(), ["00", "10"])
        self.assertEqual(grouped["target_node"].tolist(), [1, 0])
        self.assertEqual(grouped["count"].tolist(), [2, 1])
        self.assertAlmostEqual(grouped.iloc[0]["teacher_probability_mean"], 0.3)


class BKTDistillationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_raw = pd.read_parquet(ARTIFACTS / "train_prefix_examples.parquet")
        cls.validation_raw = pd.read_parquet(
            ARTIFACTS / "validation_prefix_examples.parquet"
        )
        cls.train_grouped = pd.read_parquet(ARTIFACTS / "train_grouped_tuples.parquet")
        cls.validation_grouped = pd.read_parquet(
            ARTIFACTS / "validation_grouped_tuples.parquet"
        )
        with (ARTIFACTS / "distillation_metadata.json").open(encoding="utf-8") as file:
            cls.metadata = json.load(file)
        with (ROOT / "data" / "kt_set" / "student_split.json").open(encoding="utf-8") as file:
            cls.split = json.load(file)

    def test_raw_and_grouped_counts_and_means_are_consistent(self) -> None:
        for raw, grouped in (
            (self.train_raw, self.train_grouped),
            (self.validation_raw, self.validation_grouped),
        ):
            self.assertEqual(int(grouped["count"].sum()), len(raw))
            recomputed = group_prefix_examples(raw)
            pd.testing.assert_frame_equal(grouped, recomputed, check_exact=True)

    def test_no_test_students_and_all_targets_are_covered(self) -> None:
        test_students = set(self.split["test"])
        required = set(self.metadata["required_nodes"])
        for raw in (self.train_raw, self.validation_raw):
            self.assertFalse(set(raw["student_id"]) & test_students)
            self.assertEqual(set(raw["target_node"]), required)
        self.assertEqual(self.metadata["test_students_used"], 0)

    def test_tuple_order_is_deterministic_and_states_are_valid(self) -> None:
        for raw, grouped in (
            (self.train_raw, self.train_grouped),
            (self.validation_raw, self.validation_grouped),
        ):
            ordered_raw = raw.sort_values(
                ["student_id", "prefix_index", "target_node"], kind="mergesort"
            ).reset_index(drop=True)
            pd.testing.assert_frame_equal(raw, ordered_raw, check_exact=True)
            ordered_grouped = grouped.sort_values(
                ["mastery_mask", "target_node"], kind="mergesort"
            ).reset_index(drop=True)
            pd.testing.assert_frame_equal(grouped, ordered_grouped, check_exact=True)
            for state in raw["mastery_state"].drop_duplicates():
                self.assertIsInstance(json.loads(state), list)


if __name__ == "__main__":
    unittest.main()

