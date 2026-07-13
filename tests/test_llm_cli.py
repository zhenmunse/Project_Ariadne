"""Command-line safety contract for selecting one LLM experiment."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.llm.harness import select_runs
from experiments.llm.models import ProviderRequest
from experiments.llm.providers.mock import MockProvider
from experiments.llm.run_llm import (
    execute_selected_runs,
    parse_args,
    validate_single_run_args,
    validate_workers,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = json.loads(
    (ROOT / "experiments/llm/generated/run_manifest.json").read_text(encoding="utf-8")
)["runs"]


class LLMSingleRunCLITests(unittest.TestCase):
    def test_explicit_selectors_choose_exactly_one_run(self) -> None:
        args = parse_args([
            "--single-run",
            "--provider", "open_weight",
            "--model", "open_weight",
            "--condition", "zero",
            "--target", "6",
            "--run-id", "0",
        ])
        validate_single_run_args(args)
        selected = select_runs(
            RUNS,
            model=args.model,
            condition=args.condition,
            target=args.target,
            run_id=args.run_id,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["logical_run_key"], "open_weight/zero/6/00")

    def test_single_run_rejects_any_missing_identity_selector(self) -> None:
        complete = [
            "--single-run",
            "--provider", "open_weight",
            "--model", "open_weight",
            "--condition", "full",
            "--target", "42",
            "--run-id", "7",
        ]
        for flag in ("--provider", "--model", "--condition", "--target", "--run-id"):
            with self.subTest(flag=flag):
                index = complete.index(flag)
                args = parse_args(complete[:index] + complete[index + 2:])
                with self.assertRaisesRegex(SystemExit, flag):
                    validate_single_run_args(args)

    def test_target_and_run_id_remain_user_configurable(self) -> None:
        selected = select_runs(
            RUNS,
            model="closed_frontier",
            condition="full",
            target=42,
            run_id=19,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["target_node"], 42)
        self.assertEqual(selected[0]["run_id"], 19)

    def test_worker_count_must_be_positive(self) -> None:
        validate_workers(1)
        with self.assertRaisesRegex(SystemExit, "at least 1"):
            validate_workers(0)

    def test_workers_overlap_distinct_runs_and_preserve_artifacts(self) -> None:
        class ObservedMockProvider(MockProvider):
            def __init__(self) -> None:
                super().__init__()
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def complete(self, request: ProviderRequest):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    time.sleep(0.05)
                    return super().complete(request)
                finally:
                    with self.lock:
                        self.active -= 1

        selected = select_runs(
            RUNS,
            model="closed_frontier",
            condition="zero",
            target=6,
        )[:4]
        provider = ObservedMockProvider()
        with tempfile.TemporaryDirectory() as directory, patch(
            "experiments.llm.harness.repository_commit_sha",
            return_value="test-commit",
        ):
            root = Path(directory)
            counts = execute_selected_runs(
                selected,
                provider,
                output_root=root,
                force_rerun=False,
                max_transport_attempts=3,
                workers=4,
            )
            self.assertEqual(counts, {"completed": 4})
            self.assertGreater(provider.max_active, 1)
            self.assertEqual(len(list((root / "requests").rglob("*.json"))), 4)
            self.assertEqual(len(list((root / "raw").rglob("*.json"))), 4)
            self.assertEqual(len(list((root / "parsed").rglob("*.json"))), 4)


if __name__ == "__main__":
    unittest.main()
