"""Task S1 serialization coverage for future synthetic-oracle parameters."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from experiments.synthetic.config import OracleParameterConfig, canonical_json_bytes, value_hash


class SyntheticOracleScaffoldTests(unittest.TestCase):
    def test_oracle_parameters_are_canonical_and_hashable(self) -> None:
        first = OracleParameterConfig(
            oracle_id="oracle-1", graph_id="graph-1", transfer_id="transfer-1",
            bound_target=2, bound_closure_hash="a" * 64,
            beta=0.5, alpha_by_node={2: -0.25, 1: 0.75},
        )
        second = OracleParameterConfig(
            oracle_id="oracle-1", graph_id="graph-1", transfer_id="transfer-1",
            bound_target=2, bound_closure_hash="a" * 64,
            beta=0.5, alpha_by_node={1: 0.75, 2: -0.25},
        )
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(value_hash(first), value_hash(second))

    def test_negative_beta_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "beta"):
            OracleParameterConfig("oracle", "graph", "transfer", 1, "a" * 64, -0.1, {})

    def test_oracle_artifact_round_trip_preserves_schema_and_hash(self) -> None:
        oracle = OracleParameterConfig(
            "oracle", "graph", "transfer", 2, "a" * 64, 1.0, {2: -0.5, 1: 0.25},
            metadata={"difficulty_source": "fixed", "samples": [1, 2]},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "oracle.json"
            oracle.write(path)
            loaded = OracleParameterConfig.load(path)
        self.assertEqual(loaded.to_dict(), oracle.to_dict())
        self.assertEqual(loaded.artifact_hash, oracle.artifact_hash)

    def test_nested_parameters_are_defensively_frozen(self) -> None:
        alpha = {1: 0.5}
        metadata = {"nested": [1, 2]}
        oracle = OracleParameterConfig(
            "oracle", "graph", "transfer", 1, "a" * 64, 1.0, alpha, metadata=metadata
        )
        original_hash = oracle.artifact_hash
        alpha[1] = 99.0
        metadata["nested"].append(3)
        self.assertEqual(oracle.artifact_hash, original_hash)
        self.assertEqual(oracle.to_dict()["alpha_by_node"], [[1, 0.5]])

    def test_metadata_json_types_round_trip_without_shape_guessing(self) -> None:
        cases = (
            {"empty_list": []},
            {"empty_object": {}},
            {"pairs": [["a", 1], ["b", 2]]},
            {"nested": {"items": [], "options": {}}},
        )
        for metadata in cases:
            oracle = OracleParameterConfig(
                "oracle", "graph", "transfer", 1, "a" * 64, 0.0, {1: 0.5},
                metadata=metadata,
            )
            self.assertEqual(oracle.to_dict()["metadata"], metadata)


if __name__ == "__main__":
    unittest.main()
