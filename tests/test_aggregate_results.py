"""Contract tests for the Task 14 unified result aggregator."""

from __future__ import annotations

import csv
import importlib
import json
import statistics
import tempfile
import unittest
from pathlib import Path

from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
AGGREGATOR = importlib.import_module("experiments.aggregate_results")


def read_csv(name: str) -> list[dict[str, str]]:
    with (FINAL / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


class AggregateResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = read_jsonl(FINAL / "all_sequences.jsonl")
        cls.scored = read_csv("scored_sequences.csv")
        cls.per_target = read_csv("per_target.csv")
        cls.main_table = read_csv("main_table.csv")
        cls.oracle_metrics = read_csv("oracle_metrics.csv")

    def test_all_fourteen_approved_conditions_and_only_them_are_present(self) -> None:
        expected = set(AGGREGATOR.ALL_METHODS)
        self.assertEqual({record.method for record in self.records}, expected)
        self.assertNotIn(Method.LLM_ZERO, expected)
        self.assertNotIn(Method.LLM_FULL, expected)
        self.assertEqual(len(self.records), 1882)
        self.assertEqual(len(self.scored), 1882)

    def test_identity_grid_is_complete_and_unique(self) -> None:
        targets = set(load_manifest()["targets"])
        identities = {
            (record.method, record.target_node, record.run_id) for record in self.records
        }
        self.assertEqual(len(identities), len(self.records))
        for method, _, runs in AGGREGATOR.CONDITIONS:
            method_records = [record for record in self.records if record.method is method]
            self.assertEqual(len(method_records), len(targets) * runs)
            self.assertEqual(
                {(record.target_node, record.run_id) for record in method_records},
                {(target, run) for target in targets for run in range(runs)},
            )
        expected_llm_valid = {
            Method.GPT56_SOL_ZERO: 200,
            Method.GPT56_SOL_FULL: 200,
            Method.DEEPSEEK_V4_ZERO: 195,
            Method.DEEPSEEK_V4_FULL: 197,
        }
        for method, expected_count in expected_llm_valid.items():
            method_records = [record for record in self.records if record.method is method]
            self.assertEqual(len(method_records), expected_count)

    def test_every_record_uses_current_manifest_and_closure(self) -> None:
        manifest = load_manifest()
        expected_manifest = manifest_hash(manifest)
        closures = {item["target_node"]: item["closure_hash"] for item in manifest["closures"]}
        for record in self.records:
            self.assertEqual(record.metadata["manifest_hash"], expected_manifest)
            self.assertEqual(record.metadata["closure_hash"], closures[record.target_node])
        self.assertGreater(
            len({record.metadata.get("evaluator_hash") for record in self.records}),
            1,
        )

    def test_all_public_scores_are_valid_and_nonnegative(self) -> None:
        self.assertTrue(all(row["valid"] == "True" for row in self.scored))
        self.assertTrue(all(row["invalid_reason"] == "" for row in self.scored))
        self.assertTrue(all(float(row["normalized_regret"]) >= 0.0 for row in self.scored))

    def test_per_target_and_main_table_have_expected_granularity(self) -> None:
        self.assertEqual(len(self.per_target), 140)
        self.assertEqual(len(self.main_table), 14)
        self.assertTrue(all(int(row["targets"]) == 10 for row in self.main_table))
        by_method = {row["method"]: row for row in self.main_table}
        self.assertEqual(int(by_method[Method.RANDOM_FRONTIER.value]["planned_runs"]), 1000)
        for method, _, _ in AGGREGATOR.CONDITIONS:
            expected = 1000 if method is Method.RANDOM_FRONTIER else 10
            self.assertEqual(int(by_method[method.value]["valid_runs"]), expected)
        self.assertEqual(int(by_method[Method.GPT56_SOL_ZERO.value]["planned_runs"]), 200)
        self.assertEqual(int(by_method[Method.DEEPSEEK_V4_FULL.value]["valid_runs"]), 197)

    def test_main_table_weights_targets_equally(self) -> None:
        for main in self.main_table:
            values = [
                float(row["mean_normalized_regret"])
                for row in self.per_target
                if row["method"] == main["method"]
            ]
            self.assertEqual(len(values), 10)
            self.assertEqual(
                float(main["mean_normalized_regret"]),
                statistics.fmean(values),
            )

    def test_oracle_metrics_are_validation_only_and_provenance_bound(self) -> None:
        expected_oracles = {
            "FrozenMonotonicOracle",
            "FrequencyOracle",
            "BKT-derived Set Oracle",
            "DKT-derived Set Oracle",
        }
        self.assertEqual({row["oracle"] for row in self.oracle_metrics}, expected_oracles)
        self.assertEqual({row["split"] for row in self.oracle_metrics}, {"validation"})
        expected_manifest = manifest_hash(load_manifest())
        expected_evaluator = sha256_file(ROOT / "experiments/common/evaluator.py")
        for row in self.oracle_metrics:
            source = ROOT / row["source_path"]
            self.assertEqual(row["source_sha256"], sha256_file(source))
            self.assertEqual(row["manifest_hash"], expected_manifest)
            self.assertEqual(row["aggregation_evaluator_hash"], expected_evaluator)
            with source.open(encoding="utf-8", newline="") as file:
                source_rows = list(csv.DictReader(file))
            matching = [
                item for item in source_rows
                if item.get("split", "validation") == row["split"]
            ]
            self.assertEqual(len(matching), 1)
            self.assertEqual(
                row["source_evaluator_hash"], matching[0].get("evaluator_hash", "")
            )

    def test_canonical_all_sequences_jsonl_is_reparseable(self) -> None:
        first = json.loads((FINAL / "all_sequences.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(first["method"], Method.ARIADNE_GREEDY.value)
        self.assertEqual(self.records[0].method, Method.ARIADNE_GREEDY)

    def test_aggregation_manifest_binds_inputs_outputs_and_current_scorer(self) -> None:
        with (FINAL / "aggregation_manifest.json").open(encoding="utf-8") as file:
            aggregate = json.load(file)
        self.assertEqual(aggregate["sorting"], ["method_order", "target_node", "run_id"])
        self.assertEqual(
            aggregate["evaluator_hash"],
            sha256_file(ROOT / "experiments/common/evaluator.py"),
        )
        self.assertEqual(set(aggregate["inputs"]), {method.value for method in AGGREGATOR.ALL_METHODS})
        for item in aggregate["inputs"].values():
            self.assertEqual(item["sha256"], sha256_file(ROOT / item["path"]))
        for item in aggregate["outputs"].values():
            self.assertEqual(item["sha256"], sha256_file(ROOT / item["path"]))

    def _temporary_condition(self, mutator):
        source = read_jsonl(ROOT / "results/ariadne_greedy/sequences.jsonl")
        values = [record.to_dict() for record in source]
        mutator(values)
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "records.jsonl"
        write_jsonl(path, values)
        return temporary, ((Method.ARIADNE_GREEDY, path, 1),)

    def test_missing_target_is_rejected(self) -> None:
        temporary, conditions = self._temporary_condition(lambda values: values.pop())
        with temporary, self.assertRaises(ValueError):
            AGGREGATOR.load_and_validate_records(conditions)

    def test_duplicate_identity_is_rejected(self) -> None:
        temporary, conditions = self._temporary_condition(
            lambda values: values.append(dict(values[0]))
        )
        with temporary, self.assertRaises(ValueError):
            AGGREGATOR.load_and_validate_records(conditions)

    def test_method_path_contract_is_enforced(self) -> None:
        def mutate(values):
            values[0]["method"] = Method.FREQUENCY_GREEDY.value
        temporary, conditions = self._temporary_condition(mutate)
        with temporary, self.assertRaises(ValueError):
            AGGREGATOR.load_and_validate_records(conditions)

    def test_manifest_mismatch_is_rejected(self) -> None:
        def mutate(values):
            values[0]["metadata"]["manifest_hash"] = "0" * 64
        temporary, conditions = self._temporary_condition(mutate)
        with temporary, self.assertRaises(ValueError):
            AGGREGATOR.load_and_validate_records(conditions)

    def test_closure_mismatch_is_rejected(self) -> None:
        def mutate(values):
            values[0]["metadata"]["closure_hash"] = "0" * 64
        temporary, conditions = self._temporary_condition(mutate)
        with temporary, self.assertRaises(ValueError):
            AGGREGATOR.load_and_validate_records(conditions)

    def test_shuffled_input_is_returned_in_canonical_order(self) -> None:
        temporary, conditions = self._temporary_condition(lambda values: values.reverse())
        with temporary:
            records = AGGREGATOR.load_and_validate_records(conditions)
        self.assertEqual(
            [(record.target_node, record.run_id) for record in records],
            sorted((record.target_node, record.run_id) for record in records),
        )


if __name__ == "__main__":
    unittest.main()
