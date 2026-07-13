"""Strict parser and no-repair structural validation tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import networkx as nx

from experiments.llm.models import ParseResult
from experiments.llm.parse_output import parse_output
from experiments.llm.validate_sequence import validate_sequence


ROOT = Path(__file__).resolve().parents[1]
MAPPINGS = ROOT / "experiments/llm/generated/mappings.json"


class LLMParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
        cls.bundle = next(item for item in mappings["targets"] if item["target_node"] == 42)
        graph = nx.DiGraph()
        graph.add_nodes_from(cls.bundle["opaque_to_real"])
        graph.add_edges_from(tuple(edge) for edge in cls.bundle["edge_order"])
        graph_without_target = graph.copy()
        graph_without_target.remove_node(cls.bundle["target_opaque_id"])
        cls.valid_sequence = tuple(nx.lexicographical_topological_sort(graph_without_target)) + (
            cls.bundle["target_opaque_id"],
        )

    def test_plain_json_and_single_outer_fence_are_accepted(self) -> None:
        raw = json.dumps({"sequence": list(self.valid_sequence)})
        plain = parse_output(raw)
        fenced = parse_output(f"```json\n{raw}\n```")
        self.assertTrue(plain.schema_valid)
        self.assertEqual(plain.opaque_sequence, self.valid_sequence)
        self.assertEqual(fenced.opaque_sequence, self.valid_sequence)

    def test_outer_prose_multiple_objects_and_single_quotes_are_rejected(self) -> None:
        cases = {
            "outer_text_present": 'Here: {"sequence": []}',
            "multiple_json_objects": '{"sequence": []} {"sequence": []}',
            "invalid_json": "{'sequence': []}",
        }
        for expected, raw in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(parse_output(raw).parse_error_code, expected)

    def test_schema_failures_are_distinct(self) -> None:
        self.assertEqual(parse_output("").parse_error_code, "empty_response")
        self.assertEqual(parse_output("[]").parse_error_code, "outer_text_present")
        self.assertEqual(parse_output("{}").parse_error_code, "missing_sequence")
        self.assertEqual(
            parse_output('{"sequence": [], "reason": "x"}').parse_error_code,
            "unexpected_fields",
        )
        self.assertEqual(parse_output('{"sequence": "C01"}').parse_error_code, "sequence_not_array")
        self.assertEqual(parse_output('{"sequence": [1]}').parse_error_code, "sequence_item_not_string")

    def test_valid_sequence_maps_to_real_ids(self) -> None:
        parsed = ParseResult(True, True, self.valid_sequence, None, None)
        result = validate_sequence(parsed, self.bundle)
        self.assertTrue(result.valid)
        self.assertEqual(set(result.real_sequence), {int(v) for v in self.bundle["opaque_to_real"].values()})
        self.assertEqual(result.real_sequence[-1], 42)

    def test_unknown_duplicate_missing_prerequisite_and_target_errors_are_not_repaired(self) -> None:
        valid = list(self.valid_sequence)
        cases = []
        unknown = valid.copy(); unknown[0] = "ZZZ"; cases.append(("unknown_opaque_id", unknown))
        duplicate = valid.copy(); duplicate[1] = duplicate[0]; cases.append(("duplicate_opaque_id", duplicate))
        cases.append(("missing_opaque_id", valid[:-1]))
        target_early = valid.copy(); target_early[-1], target_early[-2] = target_early[-2], target_early[-1]
        cases.append(("target_not_final", target_early))
        reversed_sequence = list(reversed(valid[:-1])) + [valid[-1]]
        cases.append(("prerequisite_violation", reversed_sequence))
        for expected, sequence in cases:
            with self.subTest(expected=expected):
                result = validate_sequence(ParseResult(True, True, tuple(sequence), None, None), self.bundle)
                self.assertFalse(result.valid)
                self.assertEqual(result.error_code, expected)
                self.assertIsNone(result.real_sequence)
