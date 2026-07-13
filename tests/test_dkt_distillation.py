"""Artifact and aggregation tests for DKT teacher distillation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from experiments.kt.artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "dkt_set"
BKT_ARTIFACTS = ROOT / "artifacts" / "bkt_set"


class DKTDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ARTIFACTS / "distillation_metadata.json").open(encoding="utf-8") as file:
            cls.metadata = json.load(file)
        with (BKT_ARTIFACTS / "surrogate_config.json").open(encoding="utf-8") as file:
            cls.bkt_config = json.load(file)
        cls.train = pd.read_parquet(ARTIFACTS / "train_prefix_examples.parquet")
        cls.validation = pd.read_parquet(
            ARTIFACTS / "validation_prefix_examples.parquet"
        )

    def test_every_prefix_queries_all_required_targets(self) -> None:
        required = set(self.metadata["required_nodes"])
        for frame in (self.train, self.validation):
            counts = frame.groupby(["student_id", "prefix_index"])["target_node"].agg(
                ["count", lambda values: set(values)]
            )
            self.assertTrue((counts["count"] == len(required)).all())
            self.assertTrue(all(value == required for value in counts["<lambda_0>"]))

    def test_train_validation_separation_and_no_test_usage(self) -> None:
        self.assertEqual(set(self.train["split"]), {"train"})
        self.assertEqual(set(self.validation["split"]), {"validation"})
        self.assertTrue(set(self.train["student_id"]).isdisjoint(self.validation["student_id"]))
        self.assertEqual(self.metadata["test_students_used"], 0)

    def test_grouped_counts_and_means_reconcile(self) -> None:
        for name, raw in (("train", self.train), ("validation", self.validation)):
            grouped = pd.read_parquet(ARTIFACTS / f"{name}_grouped_tuples.parquet")
            self.assertEqual(int(grouped["count"].sum()), len(raw))
            recomputed = (
                raw.groupby(
                    ["split", "mastery_mask", "mastery_state", "target_node"],
                    sort=False,
                    as_index=False,
                )["teacher_probability"]
                .agg(teacher_probability_mean="mean", count="size")
                .sort_values(["mastery_mask", "target_node"], kind="mergesort")
                .reset_index(drop=True)
            )
            recomputed["count"] = recomputed["count"].astype("int64")
            pd.testing.assert_frame_equal(grouped, recomputed, check_exact=True)

    def test_mastery_protocol_is_identical_to_bkt(self) -> None:
        mastery = self.metadata["mastery"]
        self.assertEqual(
            mastery["compression_config_hash"],
            self.bkt_config["compression_config_hash"],
        )
        self.assertEqual(
            mastery["zero_observation_nodes_hash"],
            "47ca6d6d085a531a2ce866021b51c4b1bd95f647190e5c676a5c258b33358992",
        )
        self.assertEqual(mastery["projection"], "zero-observation prerequisite completion")

    def test_metadata_hashes_match_artifacts(self) -> None:
        for name, entry in self.metadata["artifacts"].items():
            self.assertEqual(entry["sha256"], sha256_file(ROOT / entry["path"]), name)
        self.assertTrue(self.metadata["prefix_protocol"]["current_session_excluded"])
        self.assertTrue(self.metadata["prefix_protocol"]["hidden_state_advanced_after_query"])


if __name__ == "__main__":
    unittest.main()
