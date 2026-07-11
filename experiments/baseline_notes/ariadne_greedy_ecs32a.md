# Ariadne + Greedy on ECS32A

## Protocol

This condition uses the shared experiment manifest, the canonical CPU
`FrozenMonotonicOracle`, and the repository's one-step `GreedyPlanner`.
Targets and prerequisite closures are read from
`experiments/common/manifest.json`; the runner does not sample targets.

The method emits only standard sequence records plus its internal planning
cost. Public evaluation cost and normalized regret are produced separately by
`experiments/score_sequences.py` under the same frozen evaluator used for all
conditions.

## Oracle validation modes

| Metric | Full feature | Planning mode (`x=0`) |
|---|---:|---:|
| Samples | 3,103 | 3,103 |
| Binary samples | 2,170 | 2,170 |
| AUC | 0.774873 | 0.610952 |
| Accuracy | 0.798618 | 0.402765 |
| RMSE | 0.351672 | 0.500678 |
| MAE | 0.303330 | 0.436528 |

The planner uses planning mode. Full-feature validation measures the complete
predictor with historical features and must not be presented as the metric for
the zero-history planning interface.

## Planning and common scoring

| Metric | Value |
|---|---:|
| Targets | 10 |
| Valid sequences | 10 / 10 |
| Mean evaluation cost | 1789.669432 |
| Mean normalized regret | 0.000000593392 |
| Median normalized regret | 0.0 |
| Maximum normalized regret | 0.000005933923 (target 39) |

Two independent runner invocations, each with a newly constructed frozen
Oracle, produced identical sequences and internal costs. Runtime metadata is
diagnostic and is not expected to be bitwise identical.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/17_run_ariadne_greedy.py` | Generates standard sequences and both validation modes. |
| `results/ariadne_greedy/sequences.jsonl` | Method output records. |
| `results/ariadne_greedy/oracle_valid_metrics.csv` | Full-feature and planning-mode metrics. |
| `results/ariadne_greedy/scored_sequences.csv` | Output from the independent common scorer. |

```powershell
python experiments\09_prepare_oracle_data.py
python experiments\17_run_ariadne_greedy.py
python experiments\score_sequences.py `
  results\ariadne_greedy\sequences.jsonl `
  --output results\ariadne_greedy\scored_sequences.csv
```
