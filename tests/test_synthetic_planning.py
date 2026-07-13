"""Task S1 package, artifact, and per-run provenance coverage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.synthetic.config import (
    GraphArtifactConfig,
    SyntheticExperimentConfig,
    TransferArtifactConfig,
    build_run_record,
    ensure_output_directories,
    verify_run_record,
    write_jsonl,
)


class SyntheticPlanningScaffoldTests(unittest.TestCase):
    def test_graph_and_transfer_payloads_are_separate(self) -> None:
        graph = GraphArtifactConfig(
            graph_id="g",
            family="layered",
            nodes=(0, 1),
            edges=((0, 1),),
            target=1,
            seed=1,
        )
        transfer = TransferArtifactConfig(
            transfer_id="h",
            graph_id=graph.graph_id,
            policy="mixed_transfer",
            weights=((1, 0, 0.25),),
            seed=2,
        )
        self.assertEqual(transfer.graph_id, graph.graph_id)
        self.assertNotEqual(transfer.weights, graph.edges)

    def test_output_directories_are_isolated_under_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "synthetic"
            directories = ensure_output_directories(SyntheticExperimentConfig(output_root=root))
            self.assertEqual(directories["root"], root)
            self.assertTrue(all(path.is_dir() for path in directories.values()))

    def test_run_record_is_self_hashing_and_tamper_evident(self) -> None:
        config = SyntheticExperimentConfig()
        record = build_run_record(
            run_identity={"family": "test", "seed": 1},
            result={"cost": 120.0},
            config=config,
            commit_sha="0" * 40,
        )
        self.assertTrue(verify_run_record(record))
        changed = {**record, "result": {"cost": 121.0}}
        self.assertFalse(verify_run_record(changed))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "runs.jsonl"
            write_jsonl(output, [record])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            with self.assertRaisesRegex(ValueError, "provenance"):
                write_jsonl(output, [changed])


if __name__ == "__main__":
    unittest.main()
