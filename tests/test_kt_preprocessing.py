"""Tests for canonical KT sessions, student splits, and artifact stability."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.kt.artifacts import sha256_file
from experiments.kt.prepare_kt_data import (
    ROOT,
    SESSION_COLUMNS,
    _canonical_interactions,
    aggregate_concept_sessions,
    canonical_student_split,
    prepare_kt_data,
    validate_sessions,
)


class KTPreprocessingTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self) -> None:
        students = [f"student_{index:03d}" for index in range(30)]
        first = canonical_student_split(students)
        second = canonical_student_split(students)
        self.assertEqual(first, second)
        split_sets = {name: set(first[name]) for name in ("train", "validation", "test")}
        self.assertFalse(split_sets["train"] & split_sets["validation"])
        self.assertFalse(split_sets["train"] & split_sets["test"])
        self.assertFalse(split_sets["validation"] & split_sets["test"])
        self.assertEqual(set().union(*split_sets.values()), set(students))

    def test_session_aggregation_uses_continuous_score_and_frozen_binary_rule(self) -> None:
        interactions = pd.DataFrame(
            [
                {
                    "student_id": "a",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "item_id": 1,
                    "source_order": 0,
                    "target_node": 6,
                    "is_correct": 1,
                },
                {
                    "student_id": "a",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "item_id": 2,
                    "source_order": 1,
                    "target_node": 6,
                    "is_correct": 0,
                },
                {
                    "student_id": "a",
                    "timestamp": "2026-01-01T00:00:02Z",
                    "item_id": 3,
                    "source_order": 2,
                    "target_node": 7,
                    "is_correct": 1,
                },
            ]
        )
        result = aggregate_concept_sessions(interactions)
        self.assertEqual(result["session_index"].tolist(), [0, 1])
        self.assertEqual(result["source_order"].tolist(), [0, 2])
        self.assertEqual(result["session_score"].tolist(), [0.5, 1.0])
        self.assertEqual(result["correct"].tolist(), [0, 1])

    def test_equal_timestamps_preserve_source_order_not_item_order(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            interactions_path = Path(temp_dir) / "interactions.csv"
            pd.DataFrame(
                [
                    {
                        "user_id": "a",
                        "item_id": 20,
                        "is_correct": 1,
                        "timestamp": "2026-01-01T00:00:00Z",
                    },
                    {
                        "user_id": "a",
                        "item_id": 10,
                        "is_correct": 1,
                        "timestamp": "2026-01-01T00:00:00Z",
                    },
                ]
            ).to_csv(interactions_path, index=False)

            ordered = _canonical_interactions(interactions_path, {20: 2, 10: 1})
            result = aggregate_concept_sessions(ordered)
            self.assertEqual(ordered["item_id"].tolist(), [20, 10])
            self.assertEqual(result["target_node"].tolist(), [2, 1])
            self.assertEqual(result["source_order"].tolist(), [0, 1])

    def test_validation_rejects_non_chronological_sessions(self) -> None:
        sessions = pd.DataFrame(
            [
                ["a", 0, 0, "2026-01-02T00:00:00Z", 0, 1.0, 1, "train"],
                ["a", 1, 1, "2026-01-01T00:00:00Z", 0, 1.0, 1, "train"],
            ],
            columns=SESSION_COLUMNS,
        )
        split = {"seed": 42, "train": ["a"], "validation": [], "test": []}
        with self.assertRaisesRegex(ValueError, "chronological timestamp"):
            validate_sessions(sessions, split, {0})

    def test_repository_artifacts_are_byte_stable_across_repeated_builds(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            output = Path(temp_dir) / "kt_set"
            first_manifest = prepare_kt_data(output_dir=output)
            first_hashes = {
                name: sha256_file(output / name)
                for name in (
                    "student_split.json",
                    "concept_sessions.parquet",
                    "preprocessing_manifest.json",
                )
            }
            second_manifest = prepare_kt_data(output_dir=output)
            second_hashes = {
                name: sha256_file(output / name)
                for name in first_hashes
            }

            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(first_manifest, second_manifest)
            sessions = pd.read_parquet(output / "concept_sessions.parquet")
            self.assertEqual(sessions.columns.tolist(), SESSION_COLUMNS)
            self.assertEqual(len(sessions), first_manifest["counts"]["sessions"])
            self.assertEqual(sessions.groupby("student_id")["split"].nunique().max(), 1)

            split = json.loads((output / "student_split.json").read_text(encoding="utf-8"))
            session_students = set(sessions["student_id"])
            self.assertEqual(
                set(split["train"]) | set(split["validation"]) | set(split["test"]),
                session_students,
            )


if __name__ == "__main__":
    unittest.main()
