"""Resume, retry, invalid-output, force-rerun and crash-recovery tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.llm.artifacts import load_json, sha256_file
from experiments.llm.harness import (
    AmbiguousRunState,
    SimulatedCrash,
    assert_secret_free,
    execute_run,
)
from experiments.llm.providers.mock import MockProvider


ROOT = Path(__file__).resolve().parents[1]
RUNS = json.loads((ROOT / "experiments/llm/generated/run_manifest.json").read_text(encoding="utf-8"))["runs"]


class LLMRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = next(run for run in RUNS if run["model_key"] == "closed_frontier" and run["condition"] == "zero" and run["target_node"] == 42 and run["run_id"] == 0)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.commit_patch = patch("experiments.llm.harness.repository_commit_sha", return_value="test-commit")
        self.commit_patch.start()

    def tearDown(self) -> None:
        self.commit_patch.stop()
        self.temp.cleanup()

    def paths(self, attempt: int):
        key = Path(*self.run["logical_run_key"].split("/"))
        suffix = f"{attempt:03d}.json"
        return (
            self.root / "requests" / key / suffix,
            self.root / "raw" / key / suffix,
            self.root / "parsed" / key / suffix,
        )

    def test_valid_response_writes_three_layers_and_preserves_raw_payload(self) -> None:
        provider = MockProvider()
        result = execute_run(self.run, provider, output_root=self.root)
        request_path, raw_path, parsed_path = self.paths(0)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(request_path.is_file() and raw_path.is_file() and parsed_path.is_file())
        raw = load_json(raw_path)
        parsed = load_json(parsed_path)
        self.assertEqual(raw["provider_response"]["raw_provider_payload"]["response_text"], raw["provider_response"]["response_text"])
        self.assertEqual(raw["request_artifact_hash"], sha256_file(request_path))
        self.assertTrue(parsed["sequence_validation"]["valid"])
        self.assertEqual(parsed["raw_response_hash"], sha256_file(raw_path))
        self.assertTrue(parsed["selected_for_analysis"])

    def test_successful_invalid_response_is_not_retried(self) -> None:
        provider = MockProvider(fixture="duplicate_node")
        result = execute_run(self.run, provider, output_root=self.root, max_transport_attempts=3)
        self.assertEqual(len(provider.calls), 1)
        self.assertFalse(result["parsed"]["sequence_validation"]["valid"])
        self.assertEqual(result["parsed"]["sequence_validation"]["error_code"], "duplicate_opaque_id")
        self.assertEqual(len(list((self.root / "requests").rglob("*.json"))), 1)

    def test_transport_retry_preserves_logical_run_and_records_every_attempt(self) -> None:
        provider = MockProvider(transport_failures_before_success=2)
        result = execute_run(self.run, provider, output_root=self.root, max_transport_attempts=3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(provider.calls, [self.run["logical_run_key"]] * 3)
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(load_json(self.paths(0)[0])["status"], "transport_error")
        self.assertEqual(load_json(self.paths(1)[0])["status"], "transport_error")
        self.assertTrue(self.paths(2)[1].is_file())

    def test_exhausted_transport_budget_is_not_reset_by_resume(self) -> None:
        provider = MockProvider(fixture="timeout")
        first = execute_run(self.run, provider, output_root=self.root, max_transport_attempts=3)
        self.assertEqual(first["status"], "transport_failed")
        self.assertEqual(len(provider.calls), 3)
        resumed = MockProvider()
        second = execute_run(self.run, resumed, output_root=self.root, max_transport_attempts=3)
        self.assertEqual(second["status"], "transport_failed_exhausted")
        self.assertEqual(resumed.calls, [])

    def test_nonretryable_transport_error_remains_terminal_after_resume(self) -> None:
        provider = MockProvider(fixture="fatal_transport")
        first = execute_run(self.run, provider, output_root=self.root)
        self.assertEqual(first["status"], "transport_failed")
        self.assertEqual(len(provider.calls), 1)
        resumed = MockProvider()
        second = execute_run(self.run, resumed, output_root=self.root)
        self.assertEqual(second["status"], "transport_failed_nonretryable")
        self.assertEqual(resumed.calls, [])

    def test_resume_skips_completed_without_provider_call(self) -> None:
        execute_run(self.run, MockProvider(), output_root=self.root)
        second = MockProvider()
        result = execute_run(self.run, second, output_root=self.root)
        self.assertEqual(result["status"], "skipped_completed")
        self.assertEqual(second.calls, [])

    def test_force_rerun_creates_new_attempt_without_overwrite(self) -> None:
        execute_run(self.run, MockProvider(), output_root=self.root)
        first_hash = sha256_file(self.paths(0)[1])
        result = execute_run(self.run, MockProvider(), output_root=self.root, force_rerun=True)
        self.assertEqual(result["attempt"], 1)
        self.assertEqual(sha256_file(self.paths(0)[1]), first_hash)
        self.assertTrue(self.paths(1)[1].is_file())
        self.assertFalse(result["parsed"]["selected_for_analysis"])

    def test_crash_before_request_leaves_no_artifact(self) -> None:
        with self.assertRaises(SimulatedCrash):
            execute_run(self.run, MockProvider(), output_root=self.root, crash_at="before_request")
        self.assertFalse(self.paths(0)[0].exists())

    def test_post_provider_pre_raw_crash_fails_closed_on_resume(self) -> None:
        with self.assertRaises(SimulatedCrash):
            execute_run(self.run, MockProvider(), output_root=self.root, crash_at="after_provider_before_raw")
        resumed = MockProvider()
        with self.assertRaises(AmbiguousRunState):
            execute_run(self.run, resumed, output_root=self.root)
        self.assertEqual(resumed.calls, [])

    def test_post_raw_crash_recovers_parse_without_provider_call(self) -> None:
        with self.assertRaises(SimulatedCrash):
            execute_run(self.run, MockProvider(), output_root=self.root, crash_at="after_raw_before_parse")
        resumed = MockProvider()
        result = execute_run(self.run, resumed, output_root=self.root)
        self.assertEqual(result["status"], "recovered_parse")
        self.assertEqual(resumed.calls, [])
        self.assertTrue(result["parsed"]["sequence_validation"]["valid"])

    def test_sensitive_keys_are_rejected_recursively(self) -> None:
        with self.assertRaises(ValueError):
            assert_secret_free({"nested": {"Authorization": "secret"}})


class LLMFullMatrixMockTests(unittest.TestCase):
    def test_mock_provider_completes_all_800_planned_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "experiments.llm.harness.repository_commit_sha",
            return_value="test-commit",
        ):
            root = Path(directory)
            provider = MockProvider()
            results = [execute_run(run, provider, output_root=root) for run in RUNS]
            self.assertEqual(len(results), 800)
            self.assertEqual(len(provider.calls), 800)
            self.assertEqual({result["status"] for result in results}, {"completed"})
            self.assertTrue(
                all(result["parsed"]["sequence_validation"]["valid"] for result in results)
            )
            self.assertEqual(len(list((root / "requests").rglob("*.json"))), 800)
            self.assertEqual(len(list((root / "raw").rglob("*.json"))), 800)
            self.assertEqual(len(list((root / "parsed").rglob("*.json"))), 800)


if __name__ == "__main__":
    unittest.main()
