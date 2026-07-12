"""Tests for the shared deterministic set-oracle surrogate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from experiments.kt.artifacts import sha256_file
from src.oracle_core.set_oracle_surrogate import (
    SetOracleSurrogate,
    load_deterministic_checkpoint,
    save_deterministic_checkpoint,
)


class SetOracleSurrogateTests(unittest.TestCase):
    def test_architecture_is_exact(self) -> None:
        model = SetOracleSurrogate(61)
        layers = list(model.network)
        self.assertEqual(
            [type(layer) for layer in layers],
            [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear, nn.Sigmoid],
        )
        self.assertEqual((layers[0].in_features, layers[0].out_features), (122, 128))
        self.assertEqual((layers[2].in_features, layers[2].out_features), (128, 64))
        self.assertEqual((layers[4].in_features, layers[4].out_features), (64, 1))

    def test_forward_has_one_probability_per_input(self) -> None:
        model = SetOracleSurrogate(61)
        output = model(torch.zeros(3, 61), torch.tensor([0, 6, 52]))
        self.assertEqual(tuple(output.shape), (3,))
        self.assertTrue(torch.all((output > 0.0) & (output < 1.0)))

    def test_checkpoint_format_is_byte_stable_and_round_trips(self) -> None:
        torch.manual_seed(42)
        model = SetOracleSurrogate(61)
        metadata = {"config_hash": "abc", "selected_epoch": 3}
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.pt"
            second = Path(temp_dir) / "second.pt"
            save_deterministic_checkpoint(
                first, state_dict=model.state_dict(), metadata=metadata
            )
            save_deterministic_checkpoint(
                second, state_dict=model.state_dict(), metadata=metadata
            )
            self.assertEqual(sha256_file(first), sha256_file(second))
            loaded = load_deterministic_checkpoint(first)
            self.assertEqual(loaded["config_hash"], "abc")
            self.assertEqual(loaded["selected_epoch"], 3)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded["state_dict"][name]))


if __name__ == "__main__":
    unittest.main()

