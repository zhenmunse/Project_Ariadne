# BKT + LAO* on ECS32A

## Scope

Condition 6 of the planning matrix. This experiment uses the fitted ECS32A
BKT parameters, the same train/validation sessions, DAG, seed, targets, and
cost model as `BKT + Greedy`; only the Solver changes to LAO*.

The planner adapter uses all four BKT parameters. It computes the expected
number of attempts until the first correct response while updating the latent
mastery belief after each failed attempt, then multiplies that expectation by
the 60-second base attempt cost. The adapter is population-level and does not
track a particular student's posterior history.

## Results

### BKT validation

| Metric | Value |
|---|---:|
| Samples | 3,103 |
| Binary samples | 2,170 |
| AUC | 0.591798 |
| Accuracy | 0.586175 |
| RMSE | 0.421051 |
| MAE | 0.374297 |

### Planning

The same ten targets were used with seed 42:
`[6, 7, 12, 18, 29, 36, 39, 42, 46, 52]`.

| Planning metric | Value |
|---|---:|
| Mean expected total cost | 1144.680158 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Mean expanded states | 1689.7 |
| Total planning time for 10 targets | 55.73 s |
| Converged targets | 10 / 10 |
| Prerequisite-valid paths | 10 / 10 |

LAO* reduces the mean expected cost relative to `BKT + Greedy` by selecting a
long-horizon policy over the prerequisite DAG. All selected paths contain only
required ancestors and the target for this run.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/16_run_bkt_lao.py` | Loads BKT parameters and runs LAO*. |
| `results/bkt_lao/oracle_valid_metrics.csv` | BKT validation metrics. |
| `results/bkt_lao/planner_trajectories.csv` | Per-target paths and LAO* diagnostics. |
| `results/bkt_lao/summary.json` | Aggregate planning metrics. |

```powershell
.\.venv\Scripts\python.exe experiments\16_run_bkt_lao.py
```

The script includes a toy assertion for the BKT expected-attempt calculation
and asserts convergence and prerequisite validity for every target.
