"""Task S1 configuration coverage for later calibration work."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.synthetic.config import SeedConfig, SyntheticExperimentConfig, value_hash


class SyntheticCalibrationScaffoldTests(unittest.TestCase):
    def test_calibration_and_evaluation_seeds_must_differ(self) -> None:
        with self.assertRaisesRegex(ValueError, "different"):
            SeedConfig(calibration=7, evaluation=7)

    def test_config_round_trip_preserves_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            config = SyntheticExperimentConfig(output_root=Path(temporary_directory) / "out")
            config.write(path)
            loaded = SyntheticExperimentConfig.load(path)
            self.assertEqual(config.to_dict(), loaded.to_dict())
            self.assertEqual(value_hash(config), value_hash(loaded))


if __name__ == "__main__":
    unittest.main()
