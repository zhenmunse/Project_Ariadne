"""Task S1 artifact and mandatory provenance coverage."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.synthetic.config import (
    GraphArtifactConfig,
    OracleParameterConfig,
    RunProvenanceInputs,
    SyntheticExperimentConfig,
    TransferArtifactConfig,
    build_run_record,
    ensure_output_directories,
    repository_state,
    value_hash,
    verify_run_record,
    write_jsonl,
)


class SyntheticPlanningScaffoldTests(unittest.TestCase):
    def _artifacts(
        self, root: Path
    ) -> tuple[SyntheticExperimentConfig, RunProvenanceInputs]:
        graph = GraphArtifactConfig(
            graph_id="g", family="layered", nodes=(0, 1), edges=((0, 1),),
            target=1, seed=1, metadata={"width": 1},
        )
        transfer = TransferArtifactConfig(
            transfer_id="h", graph_id="g", policy="mixed_transfer",
            weights=((1, 0, 0.25),), seed=2, metadata={"density": 0.5},
        )
        oracle = OracleParameterConfig(
            "o", "g", "h", 1, graph.artifact_hash, 0.5, {0: 0.1, 1: 0.2}
        )
        config = SyntheticExperimentConfig(output_root=root / "output")
        paths = {
            "graph": root / "graph.json",
            "transfer": root / "transfer.json",
            "oracle": root / "oracle.json",
            "config": root / "config.json",
            "runner_a": root / "runner_a.py",
            "runner_b": root / "runner_b.py",
        }
        graph.write(paths["graph"])
        transfer.write(paths["transfer"])
        oracle.write(paths["oracle"])
        config.write(paths["config"])
        paths["runner_a"].write_text("A = 1\n", encoding="utf-8")
        paths["runner_b"].write_text("B = 2\n", encoding="utf-8")
        code_artifacts = {
            "runner": (paths["runner_a"],),
            "oracle": (paths["runner_b"],),
            "graph_factory": (paths["runner_a"],),
            "transfer_factory": (paths["runner_b"],),
            "planner": (paths["runner_a"],),
            "solver": (paths["runner_b"],),
        }
        inputs = RunProvenanceInputs(
            graph_artifact=paths["graph"], transfer_artifact=paths["transfer"],
            oracle_artifact=paths["oracle"], experiment_config=paths["config"],
            code_artifacts=code_artifacts,
        )
        return config, inputs

    def _record(
        self, config: SyntheticExperimentConfig, inputs: RunProvenanceInputs
    ) -> dict:
        state = {
            "repository_commit_sha": "0" * 40,
            "repository_dirty": False,
            "git_diff_hash": "1" * 64,
        }
        with patch("experiments.synthetic.config.repository_state", return_value=state):
            return build_run_record(
                run_identity={"family": "test", "seed": 1},
                result={"cost": 120.0}, config=config,
                provenance_inputs=inputs, commit_sha="0" * 40,
            )

    def test_graph_transfer_and_oracle_artifacts_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, inputs = self._artifacts(Path(temporary_directory))
            pairs = (
                (GraphArtifactConfig, inputs.graph_artifact),
                (TransferArtifactConfig, inputs.transfer_artifact),
                (OracleParameterConfig, inputs.oracle_artifact),
            )
            for artifact_type, path in pairs:
                loaded = artifact_type.load(path)
                before = loaded.artifact_hash
                loaded.write(path)
                self.assertEqual(artifact_type.load(path).artifact_hash, before)

    def test_output_directories_are_isolated_under_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "synthetic"
            directories = ensure_output_directories(SyntheticExperimentConfig(output_root=root))
            self.assertEqual(directories["root"], root)
            self.assertTrue(all(path.is_dir() for path in directories.values()))

    def test_missing_mandatory_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, inputs = self._artifacts(Path(temporary_directory))
            missing = RunProvenanceInputs(
                Path(temporary_directory) / "missing.json", inputs.transfer_artifact,
                inputs.oracle_artifact, inputs.experiment_config, inputs.code_artifacts,
            )
            with self.assertRaises(FileNotFoundError):
                self._record(config, missing)

    def test_missing_code_categories_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "required code artifact categories"):
            RunProvenanceInputs(Path("g"), Path("h"), Path("o"), Path("c"), ())

    def test_dirty_repository_is_rejected_unless_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, inputs = self._artifacts(Path(temporary_directory))
            state = {"repository_commit_sha": "1" * 40, "repository_dirty": True, "git_diff_hash": "2" * 64}
            with patch("experiments.synthetic.config.repository_state", return_value=state):
                with self.assertRaisesRegex(RuntimeError, "clean repository"):
                    build_run_record(
                        run_identity={}, result={}, config=config,
                        provenance_inputs=inputs,
                    )
                record = build_run_record(
                    run_identity={}, result={}, config=config,
                    provenance_inputs=inputs, allow_dirty_repository=True,
                )
            self.assertTrue(record["repository_dirty"])
            self.assertEqual(record["git_diff_hash"], "2" * 64)

    def test_repository_dirty_detection_matches_git_status(self) -> None:
        expected = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]
            )
        )
        state = repository_state()
        self.assertEqual(state["repository_dirty"], expected)
        self.assertRegex(state["repository_commit_sha"], r"^[0-9a-f]{40}$")
        self.assertRegex(state["git_diff_hash"], r"^[0-9a-f]{64}$")

    def test_invalid_manual_commit_sha_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, inputs = self._artifacts(Path(temporary_directory))
            with self.assertRaisesRegex(ValueError, "40 lowercase"):
                build_run_record(
                    run_identity={}, result={}, config=config,
                    provenance_inputs=inputs, allow_dirty_repository=True,
                    commit_sha="ABC",
                )

    def test_valid_but_incorrect_manual_commit_sha_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, inputs = self._artifacts(Path(temporary_directory))
            state = {"repository_commit_sha": "0" * 40, "repository_dirty": False, "git_diff_hash": "2" * 64}
            with patch("experiments.synthetic.config.repository_state", return_value=state):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    build_run_record(
                        run_identity={}, result={}, config=config,
                        provenance_inputs=inputs, commit_sha="1" * 40,
                    )

    def test_config_and_artifact_reference_tampering_breaks_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, inputs = self._artifacts(Path(temporary_directory))
            record = self._record(config, inputs)
            self.assertTrue(verify_run_record(record))
            changed_config = copy.deepcopy(record)
            changed_config["config"]["base_cost"] = 99.0
            changed_config["provenance_hash"] = value_hash(
                {key: value for key, value in changed_config.items() if key != "provenance_hash"}
            )
            self.assertFalse(verify_run_record(changed_config))
            changed_ref = copy.deepcopy(record)
            changed_ref["artifacts"]["graph"]["sha256"] = "f" * 64
            changed_ref["provenance_hash"] = value_hash(
                {key: value for key, value in changed_ref.items() if key != "provenance_hash"}
            )
            self.assertFalse(verify_run_record(changed_ref))

    def test_changed_input_file_content_breaks_artifact_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, inputs = self._artifacts(Path(temporary_directory))
            record = self._record(config, inputs)
            self.assertTrue(verify_run_record(record))
            inputs.graph_artifact.write_text("{}\n", encoding="utf-8")
            self.assertFalse(verify_run_record(record))
            self.assertTrue(verify_run_record(record, verify_artifacts=False))

    def test_input_order_does_not_change_provenance_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, inputs = self._artifacts(Path(temporary_directory))
            reverse = RunProvenanceInputs(
                inputs.graph_artifact, inputs.transfer_artifact, inputs.oracle_artifact,
                inputs.experiment_config,
                tuple(
                    (category, tuple(reversed(paths)))
                    for category, paths in reversed(inputs.code_artifacts)
                ),
            )
            first = self._record(config, inputs)
            second = self._record(config, reverse)
            self.assertEqual(first["provenance_hash"], second["provenance_hash"])

    def test_write_jsonl_requires_current_artifacts_and_valid_self_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, inputs = self._artifacts(root)
            record = self._record(config, inputs)
            output = root / "runs.jsonl"
            write_jsonl(output, [record])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            changed = {**record, "result": {"cost": 121.0}}
            with self.assertRaisesRegex(ValueError, "provenance"):
                write_jsonl(output, [changed])

    def test_experiment_config_artifact_must_match_materialized_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, inputs = self._artifacts(Path(temporary_directory))
            payload = json.loads(inputs.experiment_config.read_text(encoding="utf-8"))
            payload["base_cost"] = 61.0
            inputs.experiment_config.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                self._record(config, inputs)


if __name__ == "__main__":
    unittest.main()
