"""Contract tests for the canonical deterministic DKT teacher."""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

import torch

from experiments.kt.artifacts import sha256_file
from src.oracle_core.dkt_teacher import (
    DKTSequence,
    DKTTeacherModel,
    FrozenDKTTeacher,
    masked_next_target_bce,
    pad_sequences,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "dkt_set"


class DKTTeacherUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = DKTTeacherModel()
        self.model.eval()

    def test_output_shape_and_current_outcome_exclusion(self) -> None:
        first = torch.tensor([[6, 9, 12]], dtype=torch.long)
        changed_current = torch.tensor([[6, 8, 12]], dtype=torch.long)
        with torch.inference_mode():
            first_values = self.model.prefix_probabilities(first)
            changed_values = self.model.prefix_probabilities(changed_current)
        self.assertEqual(tuple(first_values.shape), (1, 3, 61))
        self.assertTrue(torch.equal(first_values[:, 0], changed_values[:, 0]))
        self.assertTrue(torch.equal(first_values[:, 1], changed_values[:, 1]))
        self.assertFalse(torch.equal(first_values[:, 2], changed_values[:, 2]))
        self.assertTrue(torch.isfinite(first_values).all())
        self.assertTrue(torch.all((first_values >= 0) & (first_values <= 1)))

    def test_padding_is_masked_from_loss(self) -> None:
        sequence = DKTSequence("a", (1, 2), (0, 1), (1, 0))
        longer = DKTSequence("b", (3, 4, 5), (1, 2, 2), (1, 0, 1))
        tokens, targets, outcomes, mask = pad_sequences([sequence, longer])
        baseline = masked_next_target_bce(
            self.model, tokens, targets, outcomes, mask
        )
        changed_outcomes = outcomes.clone()
        changed_targets = targets.clone()
        changed_outcomes[~mask] = 1.0
        changed_targets[~mask] = 60
        changed = masked_next_target_bce(
            self.model, tokens, changed_targets, changed_outcomes, mask
        )
        self.assertEqual(
            struct.pack("!d", float(baseline.detach())),
            struct.pack("!d", float(changed.detach())),
        )


class FrozenDKTTeacherArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.teacher = FrozenDKTTeacher.from_artifacts(
            config_path=ARTIFACTS / "dkt_config.json",
            checkpoint_path=ARTIFACTS / "dkt_checkpoint.pt",
        )
        with (ARTIFACTS / "dkt_training_metrics.json").open(encoding="utf-8") as file:
            cls.metrics = json.load(file)
        with (ARTIFACTS / "dkt_input_metadata.json").open(encoding="utf-8") as file:
            cls.input_metadata = json.load(file)

    def test_training_is_train_only_and_uses_canonical_inputs(self) -> None:
        self.assertFalse(self.input_metadata["training_uses_test_students"])
        self.assertTrue(self.input_metadata["current_label_excluded"])
        self.assertTrue(self.input_metadata["empty_prefix_prediction"])
        self.assertEqual(self.input_metadata["node_vocabulary_size"], 61)
        self.assertEqual(self.input_metadata["interaction_vocabulary_size"], 122)
        self.assertEqual(self.input_metadata["statistics"]["train"]["students"], 236)
        self.assertEqual(self.input_metadata["statistics"]["validation"]["students"], 29)
        self.assertEqual(self.input_metadata["statistics"]["test"]["students"], 29)

    def test_checkpoint_identity_and_recorded_metrics(self) -> None:
        self.assertEqual(
            self.metrics["teacher_checkpoint_artifact_hash"],
            sha256_file(ARTIFACTS / "dkt_checkpoint.pt"),
        )
        self.assertEqual(self.metrics["test_students_used"], 0)
        self.assertGreater(self.metrics["validation"]["auc"], 0.5)
        self.assertGreater(self.metrics["selected_epoch"], 0)

    def test_repeated_inference_and_independent_objects_are_identical(self) -> None:
        tokens = [self.teacher.token(3, 1), self.teacher.token(4, 0)]
        first = self.teacher.probability_table(tokens)
        repeated = self.teacher.probability_table(tokens)
        other = FrozenDKTTeacher.from_artifacts(
            config_path=ARTIFACTS / "dkt_config.json",
            checkpoint_path=ARTIFACTS / "dkt_checkpoint.pt",
        ).probability_table(tokens)
        self.assertTrue(torch.equal(first, repeated))
        self.assertTrue(torch.equal(first, other))

    def test_student_sequences_do_not_share_hidden_state(self) -> None:
        history_a = [self.teacher.token(3, 1), self.teacher.token(4, 1)]
        history_b = [self.teacher.token(7, 0)]
        b_before = self.teacher.probability_table(history_b)
        self.teacher.probability_table(history_a)
        b_after = self.teacher.probability_table(history_b)
        self.assertTrue(torch.equal(b_before, b_after))


if __name__ == "__main__":
    unittest.main()
