"""Deterministic anonymization contract for Task 16-1."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.common.manifest import load_manifest
from experiments.llm.anonymize import build_mappings, clean_concept_name
from experiments.llm.artifacts import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
MAPPINGS = ROOT / "experiments/llm/generated/mappings.json"


class LLMAnonymizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated = build_mappings()
        cls.committed = json.loads(MAPPINGS.read_text(encoding="utf-8"))
        cls.manifest = load_manifest()

    def test_rebuild_is_byte_identical(self) -> None:
        self.assertEqual(canonical_json_bytes(self.generated), MAPPINGS.read_bytes())

    def test_all_ten_closures_have_exact_bijections(self) -> None:
        self.assertEqual(len(self.generated["targets"]), 10)
        closures = {item["target_node"]: set(item["nodes"]) for item in self.manifest["closures"]}
        for bundle in self.generated["targets"]:
            real = {int(node) for node in bundle["real_to_opaque"]}
            opaque = set(bundle["opaque_to_real"])
            self.assertEqual(real, closures[bundle["target_node"]])
            self.assertEqual(len(real), len(opaque))
            self.assertEqual(
                {int(value) for value in bundle["opaque_to_real"].values()}, real
            )
            self.assertEqual(set(bundle["concept_order"]), opaque)
            self.assertEqual(bundle["target_opaque_id"], bundle["real_to_opaque"][str(bundle["target_node"])])

    def test_orders_are_fixed_and_edges_use_only_opaque_ids(self) -> None:
        rebuilt = {item["target_node"]: item for item in build_mappings()["targets"]}
        for bundle in self.generated["targets"]:
            other = rebuilt[bundle["target_node"]]
            self.assertEqual(bundle["mapping_hash"], other["mapping_hash"])
            self.assertEqual(bundle["concept_order"], other["concept_order"])
            self.assertEqual(bundle["edge_order"], other["edge_order"])
            allowed = set(bundle["opaque_to_real"])
            self.assertTrue(all(source in allowed and target in allowed for source, target in bundle["edge_order"]))

    def test_different_targets_use_distinct_mapping_hashes(self) -> None:
        hashes = {item["mapping_hash"] for item in self.generated["targets"]}
        self.assertEqual(len(hashes), 10)

    def test_order_prefix_cleaning_preserves_semantics(self) -> None:
        self.assertEqual(clean_concept_name("Unit 3: Boolean expressions"), "Boolean expressions")
        self.assertEqual(clean_concept_name("Week 5: string_parsing"), "string parsing")
        self.assertEqual(clean_concept_name("03_variable_assignment"), "variable assignment")
        self.assertEqual(clean_concept_name("U4-loop_semantics"), "loop semantics")
