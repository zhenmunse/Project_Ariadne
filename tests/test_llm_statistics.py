"""Train-only aggregate-statistics contract for LLM-Full."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from experiments.llm.statistics import build_full_statistics, compute_statistics_values


ROOT = Path(__file__).resolve().parents[1]
STATISTICS = ROOT / "experiments/llm/generated/full_statistics.json"
SESSIONS = ROOT / "data/kt_set/concept_sessions.parquet"


class LLMStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = json.loads(STATISTICS.read_text(encoding="utf-8"))
        cls.sessions = pd.read_parquet(SESSIONS)

    def test_rebuild_values_and_counts_match_artifact(self) -> None:
        rebuilt = build_full_statistics()
        self.assertEqual(rebuilt["nodes"], self.artifact["nodes"])
        self.assertEqual(rebuilt["statistics_values_hash"], self.artifact["statistics_values_hash"])
        self.assertEqual(sum(row["attempt_count"] for row in rebuilt["nodes"]), 25089)

    def test_attempts_are_canonical_train_sessions_only(self) -> None:
        train = self.sessions[self.sessions["split"] == "train"]
        expected = train.groupby("target_node").size().to_dict()
        for row in self.artifact["nodes"]:
            self.assertEqual(row["attempt_count"], int(expected.get(row["real_node_id"], 0)))

    def test_binary_success_rate_reconciles_at_threshold(self) -> None:
        train = self.sessions[self.sessions["split"] == "train"]
        for row in self.artifact["nodes"]:
            values = train.loc[train["target_node"] == row["real_node_id"], "session_score"]
            if values.empty:
                self.assertIsNone(row["success_rate"])
            else:
                self.assertEqual(row["success_rate"], float((values >= 0.8).mean()))

    def test_zero_observation_is_null_without_imputation(self) -> None:
        zero = [row for row in self.artifact["nodes"] if row["attempt_count"] == 0]
        self.assertEqual(len(zero), 27)
        self.assertTrue(all(row["success_rate"] is None for row in zero))

    def test_validation_and_test_are_not_counted(self) -> None:
        self.assertEqual(self.artifact["split"], "train")
        self.assertEqual(self.artifact["train_session_count"], 25089)
        self.assertNotEqual(len(self.sessions), self.artifact["train_session_count"])

    def test_mutating_validation_and_test_does_not_change_statistical_values(self) -> None:
        nodes = [row["real_node_id"] for row in self.artifact["nodes"]]
        mutated = self.sessions.copy()
        mask = mutated["split"].isin(["validation", "test"])
        mutated.loc[mask, "session_score"] = 1.0 - mutated.loc[mask, "session_score"]
        self.assertEqual(
            compute_statistics_values(self.sessions, nodes, 0.8),
            compute_statistics_values(mutated, nodes, 0.8),
        )
