# DKT + LAO* on ECS32A

## Scope

Condition 5 of the planning matrix. This experiment uses the same DKT
checkpoint, training-fold probability adapter, validation fold, DAG, seed,
targets, and cost model as `DKT + Greedy`; only the Solver changes to LAO*.

The DKT validation metrics are unchanged because the Oracle is unchanged:

| Metric | Value |
|---|---:|
| Validation AUC | 0.730437 |
| Validation accuracy | 0.710614 |
| RMSE | 0.435532 |
| Samples | 12,022 |

The planner uses training-fold DKT predictions averaged by target concept. For
concepts observed in that fold this avoids validation leakage, but it is a
population-level adapter rather than a student-specific DKT hidden-state
planner.

## First-round status

This is the first-round integration run. For DAG concepts absent from the
training-fold DKT predictions, the current adapter falls back to the mean
validation prediction. The fallback is recorded for transparency and will be
corrected and rerun after the matrix PRs are reviewed. Therefore, these
planning numbers are intermediate results, not the final leakage-free result.

## Planning result

The same ten targets were used with seed 42:
`[6, 7, 12, 18, 29, 36, 39, 42, 46, 52]`.

| Planning metric | Value |
|---|---:|
| Mean expected total cost | 1061.071168 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Mean expanded states | 1248.3 |
| Total planning time for 10 targets | 93.12 s |
| Converged targets | 10 / 10 |
| Prerequisite-valid paths | 10 / 10 |

Compared with `DKT + Greedy`, LAO* reduces mean expected cost from
`2258.932923` to `1061.071168` and mean path length from `25.3` to `11.7`.
The difference is attributable to Solver lookahead: LAO* avoids the Greedy
planner's off-target actions. The trade-off is substantially higher search
time for targets 18, 36, and 39.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/15_run_dkt_lao.py` | Loads DKT and runs LAO*. |
| `results/dkt_lao/oracle_valid_metrics.csv` | DKT validation metrics. |
| `results/dkt_lao/planner_trajectories.csv` | Per-target paths and LAO* diagnostics. |
| `results/dkt_lao/summary.json` | Aggregate planning metrics. |

```powershell
..\pykt-toolkit\.venv\Scripts\python.exe experiments\15_run_dkt_lao.py
```

All ten targets converged and produced prerequisite-valid paths.
