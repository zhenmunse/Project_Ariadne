"""Terminal classification for dispatched requests with no durable response."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.llm.artifacts import atomic_write_json, load_json
from experiments.llm.finalize_transport_ambiguous import finalize


class LLMTransportFinalizationTests(unittest.TestCase):
    def test_terminal_ambiguous_is_excluded_from_model_validity_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "requests/open_weight/full/39/10/000.json"
            atomic_write_json(request, {
                "logical_run_key": "open_weight/full/39/10",
                "attempt": 0,
                "status": "request_dispatched",
            })
            finalize(
                output_root=root,
                logical_run_key="open_weight/full/39/10",
                attempt=0,
                error_class="ConnectionResetError",
                error_message="WinError 10054",
            )
            artifact = load_json(request)
        error = artifact["transport_error"]
        self.assertEqual(artifact["status"], "transport_ambiguous_terminal")
        self.assertFalse(error["retryable"])
        self.assertTrue(error["excluded_from_model_validity_denominator"])
        self.assertTrue(error["included_in_pipeline_yield_denominator"])


if __name__ == "__main__":
    unittest.main()
