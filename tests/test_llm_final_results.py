"""Task 17-2 final status, scoring, aggregation, and provenance contract."""

from __future__ import annotations

import csv
import json
import statistics
import unittest
from collections import Counter
from pathlib import Path

from experiments.llm.artifacts import sha256_file, value_hash


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "llm"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


class LLMFinalResultsTests(unittest.TestCase):
    def test_exact_terminal_counts_and_denominators(self) -> None:
        statuses = jsonl(RESULTS / "run_status.jsonl")
        self.assertEqual(len(statuses), 800)
        self.assertEqual(
            Counter(row["terminal_status"] for row in statuses),
            {"valid": 792, "model_invalid": 7, "transport_ambiguous": 1},
        )
        responses = sum(row["provider_response_obtained"] for row in statuses)
        valid = sum(row["terminal_status"] == "valid" for row in statuses)
        self.assertEqual((responses, valid), (799, 792))
        self.assertAlmostEqual(valid / responses, 792 / 799)
        self.assertEqual(valid / len(statuses), 0.99)

    def test_valid_invalid_and_transport_outputs_are_disjoint(self) -> None:
        valid = jsonl(RESULTS / "valid_sequences.jsonl")
        invalid = jsonl(RESULTS / "invalid_runs.jsonl")
        transport = jsonl(RESULTS / "transport_failures.jsonl")
        self.assertEqual((len(valid), len(invalid), len(transport)), (792, 7, 1))
        valid_keys = {row["metadata"]["logical_run_key"] for row in valid}
        invalid_keys = {row["logical_run_key"] for row in invalid}
        transport_keys = {row["logical_run_key"] for row in transport}
        self.assertFalse(valid_keys & invalid_keys)
        self.assertFalse(valid_keys & transport_keys)
        self.assertFalse(invalid_keys & transport_keys)
        self.assertEqual(transport_keys, {"open_weight/full/39/10"})

    def test_public_evaluator_scored_every_valid_sequence(self) -> None:
        scored = csv_rows(RESULTS / "scored_valid_sequences.csv")
        self.assertEqual(len(scored), 792)
        self.assertEqual({row["valid"] for row in scored}, {"True"})
        self.assertTrue(all(row["evaluation_cost"] for row in scored))
        self.assertTrue(all(row["normalized_regret"] for row in scored))

    def test_target_then_equal_target_aggregation(self) -> None:
        per_target = csv_rows(RESULTS / "per_target.csv")
        main = csv_rows(RESULTS / "main_table.csv")
        self.assertEqual((len(per_target), len(main)), (40, 4))
        self.assertEqual({int(row["planned_runs"]) for row in per_target}, {20})
        for summary in main:
            rows = [
                row for row in per_target
                if row["model_key"] == summary["model_key"]
                and row["condition"] == summary["condition"]
            ]
            self.assertEqual(len(rows), 10)
            expected = statistics.fmean(
                float(row["mean_normalized_regret_valid_runs"]) for row in rows
            )
            self.assertAlmostEqual(
                float(summary["target_equal_mean_normalized_regret_valid_runs"]),
                expected,
            )

    def test_formal_manifest_binds_all_inputs_and_outputs(self) -> None:
        manifest = json.loads(
            (RESULTS / "formal_run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            (manifest["planned_runs"], manifest["provider_responses"],
             manifest["valid_sequences"], manifest["model_invalid_responses"],
             manifest["transport_ambiguous_runs"]),
            (800, 799, 792, 7, 1),
        )
        self.assertEqual(manifest["inputs"]["requests"]["count"], 800)
        self.assertEqual(manifest["inputs"]["raw"]["count"], 799)
        self.assertEqual(manifest["inputs"]["parsed"]["count"], 799)
        for output in manifest["outputs"].values():
            self.assertEqual(output["sha256"], sha256_file(ROOT / output["path"]))
        payload_hash = manifest.pop("manifest_payload_hash")
        self.assertEqual(payload_hash, value_hash(manifest))


if __name__ == "__main__":
    unittest.main()
