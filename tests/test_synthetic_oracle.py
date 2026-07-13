"""Task S1 serialization coverage for future synthetic-oracle parameters."""

from __future__ import annotations

import unittest

from experiments.synthetic.config import OracleParameterConfig, canonical_json_bytes, value_hash


class SyntheticOracleScaffoldTests(unittest.TestCase):
    def test_oracle_parameters_are_canonical_and_hashable(self) -> None:
        first = OracleParameterConfig(
            oracle_id="oracle-1",
            graph_id="graph-1",
            transfer_id="transfer-1",
            beta=0.5,
            alpha_by_node={2: -0.25, 1: 0.75},
        )
        second = OracleParameterConfig(
            oracle_id="oracle-1",
            graph_id="graph-1",
            transfer_id="transfer-1",
            beta=0.5,
            alpha_by_node={1: 0.75, 2: -0.25},
        )
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(value_hash(first), value_hash(second))

    def test_negative_beta_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "beta"):
            OracleParameterConfig("oracle", "graph", "transfer", -0.1, {})


if __name__ == "__main__":
    unittest.main()
