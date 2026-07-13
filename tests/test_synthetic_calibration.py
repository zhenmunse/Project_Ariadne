"""Task S1 configuration coverage for later calibration work."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.synthetic.config import (
    CalibrationConfig,
    DependenceSweepConfig,
    LayeredGraphFamilyConfig,
    SeedConfig,
    SyntheticExperimentConfig,
    TrapFamilyConfig,
    value_hash,
)


class SyntheticCalibrationScaffoldTests(unittest.TestCase):
    def test_calibration_and_evaluation_seeds_must_differ(self) -> None:
        with self.assertRaisesRegex(ValueError, "different"):
            SeedConfig(calibration=7, evaluation=7)

    def test_all_seed_namespaces_must_be_distinct(self) -> None:
        with self.assertRaisesRegex(ValueError, "all be different"):
            SeedConfig(graph=5, transfer=5, calibration=6, evaluation=7)

    def test_config_round_trip_preserves_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            config = SyntheticExperimentConfig(output_root=Path(temporary_directory) / "out")
            config.write(path)
            loaded = SyntheticExperimentConfig.load(path)
            self.assertEqual(config.to_dict(), loaded.to_dict())
            self.assertEqual(value_hash(config), value_hash(loaded))

    def test_runner_specific_configs_round_trip_without_shared_field_limit(self) -> None:
        configs = (
            DependenceSweepConfig(beta_grid=(0.0, 0.5, 2.0), random_frontier_runs=17),
            TrapFamilyConfig(delta_grid=(0.01, 0.02), tau_grid=(0.2, 0.4), k_values=(2, 8)),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            for index, config in enumerate(configs):
                path = Path(temporary_directory) / f"config-{index}.json"
                config.write(path)
                loaded = type(config).load(path)
                self.assertEqual(loaded.to_dict(), config.to_dict())
                self.assertEqual(value_hash(loaded), value_hash(config))

    def test_invalid_task_specific_config_is_rejected_at_construction(self) -> None:
        invalid_constructors = (
            lambda: LayeredGraphFamilyConfig(edge_density=-5),
            lambda: LayeredGraphFamilyConfig(max_order_ideals=-1),
            lambda: DependenceSweepConfig(beta_grid=(0.0, -1.0)),
            lambda: DependenceSweepConfig(beta_grid=(0.0, 0.5, 0.5)),
            lambda: DependenceSweepConfig(random_frontier_runs=0),
            lambda: DependenceSweepConfig(solver_tolerance=-1.0),
            lambda: TrapFamilyConfig(p_a=1.2),
            lambda: TrapFamilyConfig(q=0.0),
            lambda: TrapFamilyConfig(p_a=0.2, delta_grid=(0.0, 0.2)),
            lambda: TrapFamilyConfig(tau_grid=(-0.1, 0.0)),
            lambda: TrapFamilyConfig(solver_tolerance=0.0),
            lambda: CalibrationConfig(samples=0),
            lambda: CalibrationConfig(maximum_iterations=0),
        )
        for constructor in invalid_constructors:
            with self.subTest(constructor=constructor), self.assertRaises(
                (TypeError, ValueError)
            ):
                constructor()


if __name__ == "__main__":
    unittest.main()
