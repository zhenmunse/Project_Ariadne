"""Task 18 unified aggregation, inference, and provenance contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from experiments.common.manifest import manifest_hash, load_manifest, sha256_file


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"


def csv_rows(name: str) -> list[dict[str, str]]:
    with (FINAL / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (FINAL / name).read_text(encoding="utf-8").splitlines()]


def value_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


class FinalFreezeTests(unittest.TestCase):
    def test_final_grid_and_llm_terminal_counts(self) -> None:
        statuses = jsonl("all_run_status.jsonl")
        sequences = jsonl("all_sequences.jsonl")
        self.assertEqual((len(statuses), len(sequences)), (1890, 1882))
        llm = [row for row in statuses if row["method"].startswith(("gpt56_", "deepseek_"))]
        self.assertEqual(len(llm), 800)
        self.assertEqual(
            Counter(row["terminal_status"] for row in llm),
            {"valid": 792, "model_invalid": 7, "transport_ambiguous": 1},
        )

    def test_main_and_validity_tables_have_fourteen_methods(self) -> None:
        main = csv_rows("main_table.csv")
        validity = csv_rows("validity_table.csv")
        self.assertEqual((len(main), len(validity)), (14, 14))
        by_method = {row["method"]: row for row in main}
        self.assertEqual(int(by_method["random_frontier"]["planned_runs"]), 1000)
        self.assertEqual(int(by_method["gpt56_sol_full"]["planned_runs"]), 200)
        self.assertEqual(int(by_method["deepseek_v4_full"]["provider_responses"]), 199)
        self.assertEqual(int(by_method["deepseek_v4_full"]["valid_runs"]), 197)
        self.assertAlmostEqual(float(by_method["deepseek_v4_full"]["model_validity_rate"]), 197 / 199)
        self.assertAlmostEqual(float(by_method["deepseek_v4_full"]["validity_rate"]), 197 / 200)
        self.assertAlmostEqual(float(by_method["deepseek_v4_full"]["pipeline_yield"]), 197 / 200)

    def test_statistics_are_target_paired_and_full_minus_zero(self) -> None:
        rows = csv_rows("statistical_tests.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["sample_unit"] for row in rows}, {"target_level_mean_normalized_regret"})
        self.assertTrue(all(int(row["targets"]) == 10 for row in rows))
        self.assertTrue(all(float(row["paired_mean_difference"]) < 0 for row in rows))
        self.assertTrue(all(int(row["bootstrap_replicates"]) == 100000 for row in rows))
        self.assertTrue(all(float(row["permutation_p_value_holm"]) >= float(row["permutation_p_value"]) for row in rows))
        self.assertTrue(all(float(row["wilcoxon_p_value_holm"]) >= float(row["wilcoxon_p_value"]) for row in rows))

    def test_manifest_recomputes_every_file_and_collection_hash(self) -> None:
        manifest = json.loads((FINAL / "final_freeze_manifest.json").read_text(encoding="utf-8"))
        payload_hash = manifest.pop("manifest_payload_hash")
        self.assertEqual(payload_hash, value_hash(manifest))
        self.assertEqual(manifest["shared_protocol"]["materialized_manifest_hash"], manifest_hash(load_manifest()))

        def visit(value):
            if isinstance(value, dict):
                if set(("path", "sha256")).issubset(value):
                    self.assertEqual(value["sha256"], sha256_file(ROOT / value["path"]))
                if set(("files", "collection_hash")).issubset(value):
                    self.assertEqual(value["count"], len(value["files"]))
                    self.assertEqual(value["collection_hash"], value_hash(value["files"]))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(manifest)

    def test_formal_llm_collections_exclude_smoke_and_pilots(self) -> None:
        manifest = json.loads((FINAL / "final_freeze_manifest.json").read_text(encoding="utf-8"))
        llm = manifest["llm"]
        self.assertEqual(llm["formal_requests"]["count"], 800)
        self.assertEqual(llm["formal_raw_outputs"]["count"], 799)
        self.assertEqual(llm["formal_parsed_outputs"]["count"], 799)
        paths = [
            item["path"]
            for key in ("formal_requests", "formal_raw_outputs", "formal_parsed_outputs")
            for item in llm[key]["files"]
        ]
        self.assertFalse(any("smoke" in path or "pilot_" in path for path in paths))

    def test_model_configuration_is_exactly_frozen(self) -> None:
        manifest = json.loads((FINAL / "final_freeze_manifest.json").read_text(encoding="utf-8"))
        models = manifest["llm"]["models"]
        self.assertEqual(models["closed_frontier"]["requested_model_ids"], ["gpt-5.6-sol"])
        self.assertEqual(models["open_weight"]["requested_model_ids"], ["deepseek-v4-pro"])
        for model in models.values():
            self.assertEqual(model["requested_model_ids"], model["response_model_ids"])
            self.assertEqual(model["reasoning"], "medium")
            self.assertEqual(model["max_output_tokens"], 32768)
            self.assertIsNone(model["temperature"])
            self.assertIsNone(model["top_p"])
            self.assertEqual(model["repetitions"], 20)


if __name__ == "__main__":
    unittest.main()
